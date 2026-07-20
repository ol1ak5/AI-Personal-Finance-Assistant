import pytest

import core.llm as llm
from core.llm import (
    LLMError, analyze_clusters, generic_labels, map_columns, summarize_period,
)

STATS = [
    {
        "cluster_id": 0,
        "n_transactions": 12,
        "total_spend": 240.0,
        "avg_amount": 20.0,
        "top_merchants": ["SUPERMARKET GREENFIELD"],
        "example_descriptions": ["SUPERMARKET GREENFIELD 123"],
        "weekend_share": 0.2,
        "monthly_totals": {"2026-01": 120.0, "2026-02": 120.0},
    }
]
PERIOD = {"start": "2026-01-01", "end": "2026-02-28", "months": 2, "currency": "€"}
SUMMARY_STATS = [{**STATS[0], "name": "Groceries"}]

# A summary that satisfies the five-label contract, for fixtures that test
# other validation rules.
VALID_SUMMARY = (
    "- **The biggest change:** Spending held steady across both months.\\n"
    "- **Your subscriptions:** No recurring subscriptions were detected.\\n"
    "- **Your habits:** Groceries repeated weekly at a stable level.\\n"
    "- **One-offs:** Nothing unusual distorted either month.\\n"
    "- **Worth a look:** Groceries are the only place spending could shift."
)


def test_mock_map_columns_uses_heuristics(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "true")
    result = map_columns(
        ["Date", "Description", "Amount"],
        [["2026-01-05", "SHOP", "-10.00"]],
    )
    assert result["date_col"] == "Date"
    assert result["amount_col"] == "Amount"


def test_mock_analysis_is_deterministic(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "true")
    first = analyze_clusters(STATS, PERIOD)
    second = analyze_clusters(STATS, PERIOD)
    assert first == second
    assert first["clusters"][0]["cluster_id"] == 0
    assert first["clusters"][0]["emoji"]


def test_invalid_json_retries_then_raises(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")
    calls = []

    def bad_chat(system, user):
        calls.append(1)
        return "not json at all"

    monkeypatch.setattr(llm, "_chat_json", bad_chat)
    with pytest.raises(LLMError):
        analyze_clusters(STATS, PERIOD)
    assert len(calls) == 2


def test_valid_response_passes_validation(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setattr(
        llm,
        "_chat_json",
        lambda system, user: (
            '{"clusters": [{"cluster_id": 0, "name": "Groceries", '
            '"category": "Food", "emoji": "🛒"}]}'
        ),
    )
    result = analyze_clusters(STATS, PERIOD)
    assert result["clusters"][0]["name"] == "Groceries"
    assert result["clusters"][0]["emoji"] == "🛒"


def test_prose_summary_without_bullets_rejected(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setattr(
        llm,
        "_chat_json",
        lambda system, user: (
            '{"summary": "Steady grocery spending across the period."}'
        ),
    )
    with pytest.raises(LLMError):
        summarize_period(SUMMARY_STATS, PERIOD)


def test_missing_emoji_rejected(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setattr(
        llm,
        "_chat_json",
        lambda system, user: (
            '{"clusters": [{"cluster_id": 0, "name": "Groceries", '
            '"category": "Food"}], "summary": "Steady grocery spending."}'
        ),
    )
    with pytest.raises(LLMError):
        analyze_clusters(STATS, PERIOD)


def test_missing_keys_rejected(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setattr(llm, "_chat_json", lambda system, user: '{"clusters": []}')
    with pytest.raises(LLMError):
        analyze_clusters(STATS, PERIOD)


def test_summary_referencing_cluster_number_is_rejected(monkeypatch):
    """The model must reference clusters by their assigned name, never by
    cluster_id (e.g. 'Cluster 3') — a real leak seen in production output."""
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setattr(
        llm,
        "_chat_json",
        lambda system, user: (
            f'{{"summary": "{VALID_SUMMARY.replace("Groceries repeated", "Cluster 0 repeated")}"}}'
        ),
    )
    with pytest.raises(LLMError):
        summarize_period(SUMMARY_STATS, PERIOD)


def test_summary_using_cluster_name_is_accepted(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setattr(
        llm,
        "_chat_json",
        lambda system, user: (
            f'{{"summary": "{VALID_SUMMARY}"}}'
        ),
    )
    result = summarize_period(SUMMARY_STATS, PERIOD)
    assert "Groceries" in result


def test_generic_labels_never_fail():
    result = generic_labels(STATS)
    assert result["clusters"][0]["name"] == "Spending pattern 1"
    assert result["clusters"][0]["emoji"]
    assert "summary" in result


def test_mock_analysis_returns_merchant_behaviours(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "true")
    stats = [{
        "cluster_id": 0,
        "merchant_features": [["NETFLIX COM", 1.0, 13.99, 0.96, 1.0],
                              ["MERCADONA", 4.0, 69.02, 0.88, 0.91]],
    }]
    result = analyze_clusters(stats, {"months": 6})
    behaviours = result["clusters"][0]["merchant_behaviours"]
    assert behaviours["NETFLIX COM"] == "recurring subscription"
    assert behaviours["MERCADONA"] == "frequent shopping"


def test_behaviour_validation_rejects_bad_phrases():
    features = [["NETFLIX COM", 1.0, 13.99, 0.96, 1.0]]
    ok = llm._clean_behaviours({"NETFLIX COM": "recurring subscription"}, features)
    assert ok == {"NETFLIX COM": "recurring subscription"}
    # too long
    assert llm._clean_behaviours({"NETFLIX COM": "x" * 41}, features) == {}
    # digits could leak amounts into the table
    assert llm._clean_behaviours({"NETFLIX COM": "pays 13.99 monthly"}, features) == {}
    # unknown merchant ignored
    assert llm._clean_behaviours({"SPOTIFY": "recurring subscription"}, features) == {}
    # non-string ignored
    assert llm._clean_behaviours({"NETFLIX COM": 7}, features) == {}


def test_summary_missing_labels_rejected(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setattr(
        llm,
        "_chat_json",
        lambda system, user: (
            '{"summary": "- **The biggest change:** Spending held steady."}'
        ),
    )
    with pytest.raises(LLMError):
        summarize_period(SUMMARY_STATS, PERIOD)


def test_summary_with_question_rejected(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "false")
    questioned = VALID_SUMMARY.replace(
        "Groceries are the only place spending could shift.",
        "What caused groceries to rise?",
    )
    monkeypatch.setattr(
        llm,
        "_chat_json",
        lambda system, user: (
            f'{{"summary": "{questioned}"}}'
        ),
    )
    with pytest.raises(LLMError):
        summarize_period(SUMMARY_STATS, PERIOD)


def test_mock_summary_has_all_labels(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "true")
    summary = summarize_period(SUMMARY_STATS, PERIOD)
    for label in llm.SUMMARY_LABELS:
        assert label in summary
