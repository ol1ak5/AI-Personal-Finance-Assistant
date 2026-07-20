# Merchant Habits Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A collapsible "Merchant habits" section under "Patterns in detail" that lists every merchant grouped by spending pattern with frequency, average purchase, monthly spend, and an AI-written (rule-based fallback) behaviour phrase.

**Architecture:** Pure formatting/labeling helpers go in `core/features.py`. The GPT-5.6 phrase generation rides inside the existing single `analyze_clusters` call in `core/llm.py` (stats dicts are enriched with per-merchant features; the response gains `merchant_behaviours` per cluster). `app.py` joins the feature matrix, cluster labels, pattern names, and behaviours into HTML `<details>` groups styled like the existing pattern cards.

**Tech Stack:** Python 3, pandas, Streamlit, OpenAI Responses API (mocked via `MOCK_LLM`). Run everything with `uv run`. Tests with `MOCK_LLM=true uv run pytest`.

## Global Constraints

- Spec: `docs/merchant-habits-design.md`. UI chrome is minimal: header plus one short caption, no legend, no long explanation.
- Behaviour phrases: strings, max 40 chars, no digits. Unknown merchant keys ignored. Missing merchants fall back to local rules.
- No second API call; `MAX_OUTPUT_TOKENS` goes 1200 -> 2000.
- The no-key path must render the full section using rule-based phrases.
- Merchant names in stats/LLM payloads are the normalized uppercase identifiers already used everywhere.

---

### Task 1: Frequency formatter and behaviour rules in core/features.py

**Files:**
- Modify: `core/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces: `format_frequency(tx_per_month: float) -> str` and
  `behaviour_label(tx_per_month: float, avg_amount: float, interval_regularity: float, amount_stability: float) -> str`. Task 2's mock mode and Task 3's rendering call both.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_features.py`)

```python
from core.features import behaviour_label, format_frequency


def test_format_frequency_buckets():
    assert format_frequency(4.0) == "4/month"
    assert format_frequency(1.0) == "monthly"
    assert format_frequency(0.5) == "every 2 months"
    assert format_frequency(0.33) == "every 3 months"
    assert format_frequency(0.17) == "1-2 times in 6 months"


def test_behaviour_label_rules():
    # order: tx_per_month, avg_amount, interval_regularity, amount_stability
    assert behaviour_label(4.0, 10.0, 0.9, 0.9) == "frequent small purchases"
    assert behaviour_label(4.0, 69.0, 0.88, 0.91) == "frequent shopping"
    assert behaviour_label(2.0, 47.7, 0.92, 0.88) == "regular shopping"
    assert behaviour_label(2.0, 47.7, 0.3, 0.88) == "repeat purchases"
    assert behaviour_label(1.0, 13.99, 0.96, 1.0) == "recurring subscription"
    assert behaviour_label(1.0, 58.65, 0.96, 0.94) == "steady monthly purchase"
    assert behaviour_label(0.5, 18.6, 0.98, 1.0) == "recurring bimonthly bill"
    assert behaviour_label(0.5, 132.3, 0.87, 0.91) == "occasional shopping trips"
    assert behaviour_label(0.33, 242.0, 0.0, 0.85) == "occasional big-ticket purchase"
    assert behaviour_label(0.17, 14.2, 0.0, 1.0) == "one-off purchases"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `MOCK_LLM=true uv run pytest tests/test_features.py -v -k "frequency or behaviour"`
Expected: FAIL with ImportError (`format_frequency` not defined).

- [ ] **Step 3: Implement in `core/features.py`** (append after `build_features`)

```python
def format_frequency(tx_per_month: float) -> str:
    """Human wording for a purchase rate; never a raw '0.53/month'."""
    if tx_per_month >= 1.5:
        return f"{round(tx_per_month)}/month"
    if tx_per_month >= 0.8:
        return "monthly"
    if tx_per_month >= 0.4:
        return "every 2 months"
    if tx_per_month >= 0.28:
        return "every 3 months"
    return "1-2 times in 6 months"


def behaviour_label(tx_per_month: float, avg_amount: float,
                    interval_regularity: float, amount_stability: float) -> str:
    """Rule-based habit phrase; the local fallback for AI-written behaviours."""
    if tx_per_month >= 3.5:
        return "frequent small purchases" if avg_amount < 15 else "frequent shopping"
    if tx_per_month >= 1.5:
        return "regular shopping" if interval_regularity >= 0.6 else "repeat purchases"
    if 0.8 <= tx_per_month <= 1.3 and interval_regularity >= 0.8:
        return ("recurring subscription" if amount_stability >= 0.97
                else "steady monthly purchase")
    if 0.4 <= tx_per_month < 0.8 and interval_regularity >= 0.9:
        return "recurring bimonthly bill"
    if 0.4 <= tx_per_month < 0.8 and interval_regularity >= 0.6:
        return "occasional shopping trips"
    if avg_amount >= 100:
        return "occasional big-ticket purchase"
    return "one-off purchases"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `MOCK_LLM=true uv run pytest tests/test_features.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add core/features.py tests/test_features.py
git commit -m "feat: frequency formatter and behaviour-label rules"
```

---

### Task 2: merchant_behaviours in the analyze_clusters contract

**Files:**
- Modify: `core/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `behaviour_label` and `format_frequency` are NOT used here except `behaviour_label` in mock mode.
- Consumes (payload shape): each stats item MAY carry
  `"merchant_features": [[name, tx_per_month, avg_amount, interval_regularity, amount_stability], ...]` (added by Task 3 in `app.py`).
- Produces: validated `analyze_clusters` responses where each cluster dict may include `"merchant_behaviours": {<merchant>: <phrase>}`; phrases are str, <= 40 chars, digit-free; keys not present in that cluster's `merchant_features` names are dropped. Mock mode returns rule-based phrases.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_llm.py`; follow the file's existing MOCK_LLM fixture style if one exists, otherwise use `monkeypatch.setenv("MOCK_LLM", "true")`)

```python
def test_mock_analysis_returns_merchant_behaviours(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "true")
    stats = [{
        "cluster_id": 0,
        "merchant_features": [["NETFLIX COM", 1.0, 13.99, 0.96, 1.0],
                              ["MERCADONA", 4.0, 69.02, 0.88, 0.91]],
    }]
    result = llm.analyze_clusters(stats, {"months": 6})
    behaviours = result["clusters"][0]["merchant_behaviours"]
    assert behaviours["NETFLIX COM"] == "recurring subscription"
    assert behaviours["MERCADONA"] == "frequent shopping"


def test_behaviour_validation_rejects_bad_phrases():
    features = [["NETFLIX COM", 1.0, 13.99, 0.96, 1.0]]
    ok = llm._clean_behaviours({"NETFLIX COM": "recurring subscription"}, features)
    assert ok == {"NETFLIX COM": "recurring subscription"}
    # too long
    assert llm._clean_behaviours({"NETFLIX COM": "x" * 41}, features) == {}
    # digits leak amounts
    assert llm._clean_behaviours({"NETFLIX COM": "pays 13.99 monthly"}, features) == {}
    # unknown merchant ignored
    assert llm._clean_behaviours({"SPOTIFY": "recurring subscription"}, features) == {}
    # non-string ignored
    assert llm._clean_behaviours({"NETFLIX COM": 7}, features) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `MOCK_LLM=true uv run pytest tests/test_llm.py -v -k behaviour`
Expected: FAIL (`_clean_behaviours` missing; mock lacks `merchant_behaviours`).

- [ ] **Step 3: Implement in `core/llm.py`**

3a. `MAX_OUTPUT_TOKENS = 1200` becomes `MAX_OUTPUT_TOKENS = 2000`.

3b. Add near the validation helpers:

```python
def _clean_behaviours(raw, merchant_features) -> dict[str, str]:
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
```

3c. In the mock branch of `analyze_clusters`, add behaviours per cluster:

```python
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
            "summary": "- Mock summary: spending is stable across the analysis period.",
        }
```

3d. In `ANALYSIS_SYSTEM`, extend the reply shape line to:

```
{"clusters": [{"cluster_id": <int>, "name": "<short descriptive name>",
"category": "<concise semantic category>", "emoji": "<one emoji>",
"merchant_behaviours": {"<merchant>": "<habit phrase>"}}],
"summary": "<a markdown bullet list: 4-6 lines, each starting with '- '>"}.
```

and append one rule block after the naming rules:

```
Merchant behaviour rules:
- For every merchant listed in a cluster's merchant_features
  ([name, purchases_per_month, avg_amount, interval_regularity,
  amount_stability]), write a short lowercase habit phrase (max 40
  characters, no digits), e.g. "weekly grocery run", "recurring
  subscription", "occasional big-ticket purchase".
- Describe the buying habit the numbers show; never invent specifics.
```

3e. In the real-mode `validate` inside `analyze_clusters`, after the emoji check, sanitize in place (validation must not fail the whole response over one bad phrase; cleaning enforces the contract):

```python
        features_by_id = {
            item["cluster_id"]: item.get("merchant_features", []) for item in stats
        }
        for cluster in clusters:
            cluster["merchant_behaviours"] = _clean_behaviours(
                cluster.get("merchant_behaviours"),
                features_by_id.get(cluster.get("cluster_id"), []),
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `MOCK_LLM=true uv run pytest tests/test_llm.py -v`
Expected: all PASS (existing tests too; the mock's new key must not break them).

- [ ] **Step 5: Commit**

```bash
git add core/llm.py tests/test_llm.py
git commit -m "feat: AI merchant behaviour phrases in the analysis contract"
```

---

### Task 3: Merchant habits section in app.py

**Files:**
- Modify: `app.py` (constant `DEMO_PATH` area untouched; changes at the analysis call ~line 593 and after the pattern-cards grid ~line 780)

**Interfaces:**
- Consumes: `features.format_frequency`, `features.behaviour_label`,
  `feats` (DataFrame indexed by merchant with `tx_per_month`, `avg_amount`,
  `monthly_spend`, `interval_regularity`, `amount_stability`),
  `cluster_result.labels` (Series merchant -> cluster id), `stats` (ordered
  list of cluster dicts), `names`, `analysis["clusters"]`, `DONUT_COLORS`,
  `format_amount`, `currency`.
- Produces: `merchant_habits_html(...) -> str` and the rendered section.

- [ ] **Step 1: Enrich the LLM payload.** Where `cached_analysis` is called with `json.dumps(all_stats, ...)`, build an enriched copy first (just above the `if llm.is_configured():` line):

```python
        merchant_features_by_cluster = {
            int(cluster_id): [
                [
                    merchant,
                    round(float(row["tx_per_month"]), 2),
                    round(float(row["avg_amount"]), 2),
                    round(float(row["interval_regularity"]), 2),
                    round(float(row["amount_stability"]), 2),
                ]
                for merchant, row in feats[cluster_result.labels == cluster_id]
                .sort_values("monthly_spend", ascending=False)
                .iterrows()
            ]
            for cluster_id in cluster_result.labels.unique()
        }
        stats_for_ai = [
            {**item, "merchant_features": merchant_features_by_cluster.get(item["cluster_id"], [])}
            for item in all_stats
        ]
```

and pass `json.dumps(stats_for_ai, sort_keys=True)` to `cached_analysis` instead of `json.dumps(all_stats, sort_keys=True)`.

- [ ] **Step 2: Collect behaviours with fallback.** After `names`/`emojis` are built:

```python
        ai_behaviours: dict[str, str] = {}
        for item in analysis["clusters"]:
            ai_behaviours.update(item.get("merchant_behaviours", {}))

        def merchant_behaviour(merchant: str) -> str:
            if merchant in ai_behaviours:
                return ai_behaviours[merchant]
            row = feats.loc[merchant]
            return features.behaviour_label(
                row["tx_per_month"], row["avg_amount"],
                row["interval_regularity"], row["amount_stability"],
            )
```

- [ ] **Step 3: Add the rendering function** next to the other HTML builders (after `rounded_donut_svg`); reuse the card palette (`#171027`, `#3C2E5A`, `#241A38`, `#8B7FA8`, `#C9BFE0`):

```python
def merchant_habits_html(stats, names, feats, labels, behaviour_for, currency) -> str:
    """Collapsible per-pattern merchant tables (<details>, no JS, no rerun)."""
    css = (
        "<style>"
        ".habits details{background:#171027;border:1px solid #3C2E5A;"
        "border-radius:16px;margin:0 0 12px;overflow:hidden;}"
        ".habits summary{display:flex;align-items:center;gap:9px;cursor:pointer;"
        "padding:13px 18px;list-style:none;flex-wrap:wrap;}"
        ".habits summary::-webkit-details-marker{display:none;}"
        ".habits summary:hover{background:#241A38;}"
        ".habits .dot{width:10px;height:10px;border-radius:50%;flex:none;}"
        ".habits .pname{font-weight:700;font-size:15px;}"
        ".habits .pmeta{color:#8B7FA8;font-size:12px;}"
        ".habits .ptotal{margin-left:auto;font-weight:700;font-size:12.5px;"
        "color:#C9BFE0;background:#241A38;border-radius:999px;padding:4px 10px;"
        "white-space:nowrap;}"
        ".habits .chev{color:#8B7FA8;transition:transform .15s;}"
        ".habits details[open] .chev{transform:rotate(180deg);}"
        ".habits table{width:100%;border-collapse:collapse;font-size:13px;}"
        ".habits th{text-align:left;font-size:10.5px;text-transform:uppercase;"
        "letter-spacing:.06em;color:#8B7FA8;padding:9px 18px 7px;}"
        ".habits td{padding:8px 18px;border-top:1px solid rgba(60,46,90,.55);}"
        ".habits td.num,.habits th.num{text-align:right;"
        "font-variant-numeric:tabular-nums;white-space:nowrap;}"
        ".habits td.beh{color:#C9BFE0;}"
        "</style>"
    )
    groups = []
    for idx, item in enumerate(stats):
        cluster_id = item["cluster_id"]
        members = feats[labels == cluster_id].sort_values(
            "monthly_spend", ascending=False
        )
        if members.empty:
            continue
        color = DONUT_COLORS[idx % len(DONUT_COLORS)]
        monthly_total = float(members["monthly_spend"].sum())
        rows = "".join(
            "<tr>"
            f"<td style='font-weight:600;'>{html.escape(merchant.title())}</td>"
            f"<td>{features.format_frequency(row['tx_per_month'])}</td>"
            f"<td class='num'>{format_amount(row['avg_amount'], currency)}</td>"
            f"<td class='num'>{format_amount(row['monthly_spend'], currency)}</td>"
            f"<td class='beh'>{html.escape(behaviour_for(merchant))}</td>"
            "</tr>"
            for merchant, row in members.iterrows()
        )
        groups.append(
            "<details>"
            "<summary>"
            f"<span class='dot' style='background:{color};'></span>"
            f"<span class='pname'>{html.escape(names[cluster_id])}</span>"
            f"<span class='pmeta'>{len(members)} merchants</span>"
            f"<span class='ptotal'>{format_amount(monthly_total, currency)}/month</span>"
            "<span class='chev'>▾</span>"
            "</summary>"
            "<table>"
            "<thead><tr><th>Merchant</th><th>Frequency</th>"
            "<th class='num'>Avg purchase</th><th class='num'>Monthly spend</th>"
            "<th>Behaviour</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            "</details>"
        )
    return css + "<div class='habits'>" + "".join(groups) + "</div>"
```

- [ ] **Step 4: Render the section** right after the pattern-cards `st.markdown(...)` grid:

```python
        st.subheader("Merchant habits")
        st.caption("Every merchant in each pattern, with its buying rhythm.")
        st.markdown(
            merchant_habits_html(
                stats, names, feats, cluster_result.labels,
                merchant_behaviour, currency,
            ),
            unsafe_allow_html=True,
        )
```

- [ ] **Step 5: Full test suite**

Run: `MOCK_LLM=true uv run pytest -q`
Expected: all PASS.

- [ ] **Step 6: Verify in the running app** (dev server `plainfinance` from `.claude/launch.json`): load demo data, confirm the section renders under "Patterns in detail", groups collapse/expand without a rerun, all merchants present, phrases sensible, no console errors. Screenshot for the user.

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "feat: collapsible merchant-habits section on the dashboard"
```

---

### Task 4: End-to-end guard on the demo data

**Files:**
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: the real `data/spending_demo.csv` through `parser`, `merchants`, `features` (same chain the app uses).

- [ ] **Step 1: Write the test** (append to `tests/test_features.py`)

```python
def test_demo_merchants_get_sensible_behaviours():
    from pathlib import Path

    from core import merchants, parser

    data = (Path(__file__).parent.parent / "data" / "spending_demo.csv").read_bytes()
    frames = parser.load_frames(data, "spending_demo.csv")
    df = list(frames.values())[0]
    result = parser.apply_mapping(df, parser.guess_mapping(df))
    feats = build_features(merchants.add_merchant_column(result.df))

    labels = {
        merchant: behaviour_label(
            row["tx_per_month"], row["avg_amount"],
            row["interval_regularity"], row["amount_stability"],
        )
        for merchant, row in feats.iterrows()
    }
    assert set(labels) == set(feats.index)          # every merchant labeled
    assert labels["MERCADONA"] == "frequent shopping"
    assert labels["NETFLIX COM"] == "recurring subscription"
    assert labels["CANAL ISABEL II"] == "recurring bimonthly bill"
    assert labels["BOOKING COM"] == "occasional big-ticket purchase"
```

(`build_features` and `behaviour_label` are already imported at the top of the file from Task 1; add `build_features` to that import if missing.)

- [ ] **Step 2: Run the test**

Run: `MOCK_LLM=true uv run pytest tests/test_features.py -v`
Expected: all PASS. If a threshold misses one of the four named merchants, tune the rule thresholds in `behaviour_label` (spec allows tuning), rerun Task 1 tests, and only then adjust this test if the phrase legitimately changed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features.py
git commit -m "test: demo-data guard for behaviour labels"
```
