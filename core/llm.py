"""GPT-5.6 integration with validation, retry, mock mode, and local fallback.

Only this module touches the OpenAI API. It receives either consented headers
and a few sample rows, or aggregate cluster statistics—never an uploaded file.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any

_CLUSTER_NUMBER_PATTERN = re.compile(r"\bcluster\s*#?\s*\d", re.IGNORECASE)

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-terra")
MAX_OUTPUT_TOKENS = 3000
MAPPING_KEYS = [
    "date_col",
    "description_col",
    "amount_col",
    "debit_col",
    "credit_col",
    "category_col",
    "date_format",
    "decimal_separator",
    "expenses_are",
]


class LLMError(Exception):
    """Raised when an AI response cannot be used safely."""


def _mock() -> bool:
    return os.getenv("MOCK_LLM", "").lower() in {"1", "true", "yes"}


def is_configured() -> bool:
    """Return whether an API key is available, without exposing its value."""
    return _mock() or bool(os.getenv("OPENAI_API_KEY"))


def _chat_json(system: str, user: str) -> str:
    """Call the Responses API and return its JSON text output."""
    from openai import BadRequestError, OpenAI

    kwargs: dict[str, Any] = dict(
        model=MODEL,
        instructions=system,
        input=user,
        text={"format": {"type": "json_object"}},
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    # Classification and phrasing need little deliberation; low effort cuts
    # latency substantially on reasoning models.
    effort = os.getenv("OPENAI_REASONING_EFFORT", "low")
    if effort:
        kwargs["reasoning"] = {"effort": effort}
    try:
        response = OpenAI().responses.create(**kwargs)
    except BadRequestError:
        if "reasoning" not in kwargs:
            raise
        kwargs.pop("reasoning")  # model rejects the parameter: retry without
        response = OpenAI().responses.create(**kwargs)
    if not response.output_text:
        raise ValueError("The model returned no text output")
    return response.output_text


def _json_with_retry(
    system: str,
    user: str,
    validate: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    feedback = ""
    last_error: Exception | None = None
    for _ in range(2):
        try:
            response = _chat_json(system, user + feedback)
            data = json.loads(response)
            if not isinstance(data, dict):
                raise ValueError("response must be a JSON object")
            validate(data)
            return data
        except Exception as error:
            last_error = error
            feedback = (
                f"\n\nYour previous response was invalid: {error}. "
                "Return valid JSON only and follow the requested shape exactly."
            )
    raise LLMError(f"AI request failed twice. Last error: {last_error}")


MAPPING_SYSTEM = """You map bank-export spreadsheet columns to a schema.
Reply with JSON only. Include these keys: date_col, description_col, amount_col,
debit_col, credit_col, category_col, date_format (a strptime format such as
%d.%m.%Y), decimal_separator ('.' or ','), and expenses_are ('negative',
'positive', or 'debit_col'). Use null for fields that do not apply. Copy all
column names exactly from the provided headers."""

ANALYSIS_SYSTEM = """You are a personal-finance analyst. You receive only
aggregate spending-cluster statistics for one user, never a raw transaction
table. Reply with JSON only in this shape:
{"clusters": [{"cluster_id": <int>, "name": "<short descriptive name>",
"category": "<concise semantic category>", "emoji": "<one emoji>",
"merchant_behaviours": {"<merchant>": "<habit phrase>"}}]}.

Naming rules:
- Classify clusters generatively rather than from a fixed category list.
- Base every label strictly on the supplied evidence. Never invent merchants,
  amounts, or trends.
- When evidence is insufficient, use "Unclear spending pattern".
- Names must be clearly distinct from each other: no two patterns may read
  almost the same (bad: "Dining, travel and shopping" alongside "Travel,
  home and shopping"). Give each pattern one recognizable identity.
- emoji is exactly one emoji character that represents the general kind of
  spending (e.g. groceries, transport, dining, subscriptions), not a specific
  brand or logo. Pick the single most fitting emoji; do not combine several.

Merchant behaviour rules:
- For every merchant listed in a cluster's merchant_features
  ([name, purchases_per_month, avg_amount, interval_regularity,
  amount_stability]), write a short lowercase habit phrase (max 40
  characters, no digits), e.g. "weekly grocery run", "recurring
  subscription", "occasional big-ticket purchase".
- Describe the buying habit the numbers show; never invent specifics."""

SUMMARY_SYSTEM = """You are a personal-finance analyst. You receive aggregate
spending statistics for the analysis window the user selected, never a raw
transaction table. Each cluster carries its already-assigned pattern "name":
use those names verbatim; never rename, merge, or invent patterns.
The payload's "recurring_charges" lists every merchant charging a stable
amount on a regular monthly rhythm, as [name, exact_monthly_price]. It is
the only source for the subscriptions bullet, and it is computed from the
user's full history, so it stays exact and complete even when the selected
window is short.
Reply with JSON only in this shape:
{"summary": "<exactly five labeled markdown bullets as specified below,
each on its own line starting with '- '>"}.

Summary rules — the summary is the product's centerpiece. Write it like a
sharp friend who just read the user's statement, not like a report:
- Format: exactly five markdown bullets, in this order, each starting with
  "- " and the bold label shown, each label followed by 1-2 sentences.
- "- **The biggest change:** " the single biggest movement in the period and
  what actually caused it. Always distinguish a one-off event (a trip, a
  large purchase) from a changed habit. If spending was basically flat, say
  so plainly instead of inventing a trend.
- "- **Your subscriptions:** " the true subscriptions from
  recurring_charges: things the user signed up for and could cancel today
  (streaming, memberships, apps, delivery clubs). Exclude rent, utilities,
  and telecom bills from this bullet (mention those under habits as part of
  the fixed base instead). Name each subscription with its exact price from
  recurring_charges, then their monthly total. If recurring_charges is
  empty, say plainly that no subscriptions were detected.
- "- **Your habits:** " the repeated everyday spending (groceries, delivery,
  coffee): which habits are stable and which are quietly drifting up or
  down, with the size and pace of the drift.
- "- **One-offs:** " the events that distorted individual months, named by
  what they were using the top merchants ("a car rental and a hotel in
  April, about €600"), never by pattern names. Do not retell the biggest-
  change bullet; this bullet itemizes, that one interprets.
- "- **Worth a look:** " one specific, quantified observation the user can
  act on, chosen because it moves the most money. A statement, never a
  question, never generic advice such as "consider budgeting", and never a
  repeat of a finding already stated in an earlier bullet.
- Voice: second person ("your"), plain everyday words, no filler ("overall",
  "it is worth noting"), no exclamation marks, and no questions anywhere.
- Merchant names: write them in their natural brand form, never the
  uppercase bank form ("Netflix", not "NETFLIX COM"; "Apple iCloud", not
  "APPLE ICLOUD"; "El Corte Inglés", not "EL CORTE INGLES-D").
- Numbers: every amount carries the currency symbol given in the period's
  "currency" field (omit only if that field is empty). Itemized recurring
  charges use exact prices ("€22.99"), never "about €3". Aggregates and
  changes use rounded whole amounts ("about €570 (+33%)"). At most one
  number or number pair per claim; never narrate month-by-month totals.
  Every claim must be computable from the supplied statistics.
- Never write the word "cluster" followed by a number (e.g. "Cluster 3").
  Refer to each pattern only by its given name.
- If the window covers fewer than 2 full months, still write all five
  bullets from what IS knowable. Subscriptions stay exact and itemized
  (recurring_charges covers the full history). The biggest change compares
  within the month or says plainly that one month cannot show a trend.
  Habits and one-offs describe what the month actually contained. Never pad
  multiple bullets with "more data needed" boilerplate."""

SUMMARY_LABELS = ("**The biggest change:**", "**Your subscriptions:**",
                  "**Your habits:**", "**One-offs:**", "**Worth a look:**")


def map_columns(headers: list[str], sample_rows: list[list[Any]]) -> dict[str, Any]:
    """Map spreadsheet columns after the user has consented to the request."""
    if _mock():
        import pandas as pd

        from core import parser

        guess = parser.guess_mapping(pd.DataFrame(sample_rows, columns=headers))
        if guess is None:
            raise LLMError("Mock mapping could not identify the required columns.")
        return {key: getattr(guess, key) for key in MAPPING_KEYS}

    def validate(data: dict[str, Any]) -> None:
        columns = set(headers)
        if data.get("date_col") not in columns:
            raise ValueError("date_col is not one of the uploaded headers")
        if data.get("description_col") not in columns:
            raise ValueError("description_col is not one of the uploaded headers")
        has_amount = data.get("amount_col") in columns
        has_debit_credit = (
            data.get("debit_col") in columns and data.get("credit_col") in columns
        )
        if not (has_amount or has_debit_credit):
            raise ValueError("no valid amount or debit/credit mapping")

    payload = json.dumps({"headers": headers, "sample_rows": sample_rows[:5]})
    data = _json_with_retry(MAPPING_SYSTEM, payload, validate)
    return {key: data.get(key) for key in MAPPING_KEYS}


def _clean_behaviours(raw: Any, merchant_features: list[list]) -> dict[str, str]:
    """Keep only valid AI behaviour phrases: known merchant, str, <=40 chars,
    digit-free (prevents the model echoing amounts into the table)."""
    if not isinstance(raw, dict):
        return {}
    known = {row[0] for row in merchant_features}
    return {
        merchant: phrase.strip()
        for merchant, phrase in raw.items()
        if merchant in known
        and isinstance(phrase, str)
        and 0 < len(phrase.strip()) <= 40
        and not any(ch.isdigit() for ch in phrase)
    }


def summarize_period(
    stats: list[dict[str, Any]],
    period: dict[str, Any],
    recurring_charges: list[list] | None = None,
) -> str:
    """Write the five-bullet summary for the user's selected analysis window.

    Each stats item must carry the pattern "name" already assigned by
    analyze_clusters, so period switches never rename patterns.
    recurring_charges ([merchant, exact monthly price] pairs, from
    features.recurring_charges) grounds the subscriptions bullet.
    """
    if _mock():
        return (
            "- **The biggest change:** Mock analysis: spending is stable.\n"
            "- **Your subscriptions:** Mock analysis: none detected.\n"
            "- **Your habits:** Mock analysis: steady repeated spending.\n"
            "- **One-offs:** Mock analysis: no unusual months.\n"
            "- **Worth a look:** Mock analysis: nothing stands out."
        )

    def validate(data: dict[str, Any]) -> None:
        summary = data.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("summary is missing or empty")
        if _CLUSTER_NUMBER_PATTERN.search(summary):
            raise ValueError(
                "summary must refer to patterns by their assigned name, never "
                "by cluster_id number (do not write 'Cluster 3' or similar)"
            )
        if not any(line.strip().startswith("- ") for line in summary.splitlines()):
            raise ValueError(
                "summary must be a markdown bullet list with each line "
                "starting with '- '"
            )
        if "?" in summary:
            raise ValueError(
                "summary must contain plain statements only, never questions"
            )
        missing = [label for label in SUMMARY_LABELS if label not in summary]
        if missing:
            raise ValueError(
                "summary must contain all five labeled bullets; missing: "
                + ", ".join(missing)
            )

    payload = json.dumps({
        "period": period,
        "recurring_charges": recurring_charges or [],
        "clusters": stats,
    })
    return _json_with_retry(SUMMARY_SYSTEM, payload, validate)["summary"]


def analyze_clusters(stats: list[dict[str, Any]], period: dict[str, Any]) -> dict[str, Any]:
    """Name the patterns and phrase merchant habits from aggregate statistics."""
    if _mock():
        from core.features import behaviour_label

        return {
            "clusters": [
                {
                    "cluster_id": item["cluster_id"],
                    "name": f"Mock pattern {item['cluster_id']}",
                    "category": "Mock",
                    "emoji": "📊",
                    "merchant_behaviours": {
                        name: behaviour_label(tpm, avg, reg, stab)
                        for name, tpm, avg, reg, stab in item.get("merchant_features", [])
                    },
                }
                for item in stats
            ],
        }

    expected_ids = {item["cluster_id"] for item in stats}

    def validate(data: dict[str, Any]) -> None:
        clusters = data.get("clusters")
        if not isinstance(clusters, list):
            raise ValueError("clusters is missing")
        returned_ids = {item.get("cluster_id") for item in clusters if isinstance(item, dict)}
        if returned_ids != expected_ids or len(clusters) != len(expected_ids):
            raise ValueError("cluster ids do not match the supplied clusters")
        for cluster in clusters:
            if not isinstance(cluster.get("name"), str) or not cluster["name"].strip():
                raise ValueError("cluster name is missing")
            if not isinstance(cluster.get("category"), str) or not cluster["category"].strip():
                raise ValueError("cluster category is missing")
            if not isinstance(cluster.get("emoji"), str) or not cluster["emoji"].strip():
                raise ValueError("cluster emoji is missing")
        # Behaviour phrases are best-effort: sanitize instead of failing the
        # whole response over one bad phrase; missing ones fall back to rules.
        features_by_id = {
            item["cluster_id"]: item.get("merchant_features", []) for item in stats
        }
        for cluster in clusters:
            cluster["merchant_behaviours"] = _clean_behaviours(
                cluster.get("merchant_behaviours"),
                features_by_id.get(cluster.get("cluster_id"), []),
            )

    payload = json.dumps({"period": period, "clusters": stats})
    return _json_with_retry(ANALYSIS_SYSTEM, payload, validate)


def generic_labels(stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Return usable local labels when AI analysis is unavailable."""
    return {
        "clusters": [
            {
                "cluster_id": item["cluster_id"],
                "name": f"Spending pattern {index + 1}",
                "category": "Uncategorized",
                "emoji": "💳",
            }
            for index, item in enumerate(stats)
        ],
        "summary": (
            "AI analysis is temporarily unavailable. The patterns below were "
            "detected locally from your data."
        ),
    }
