# Merchant habits section: design

Date: 2026-07-19. Status: agreed in conversation, pending final review.
Visual mockup (real demo-data numbers): claude.ai artifact "Merchant habits mockup".

## Goal

The dashboard groups merchants into spending patterns but never shows the
per-merchant detail behind them. A new "Merchant habits" section lists every
merchant, organized by pattern, with the KPIs the clustering already computes:
how often you buy, what a typical purchase costs, monthly spend, and a short
plain-language description of the habit. It doubles as an explainability view:
you can see exactly which merchants form each pattern and why they belong
together.

## UI

- New section under "Patterns in detail", rendered as custom HTML via
  `st.markdown(unsafe_allow_html=True)`, matching the pattern-card styling
  (same colors, borders, and the donut color per pattern).
- Chrome is minimal: the section header plus one short caption. No long
  explanation paragraph and no legend table of behaviour rules in the UI
  (those existed only in the design mockup).
- One collapsible group per pattern, using native `<details>` and `<summary>`
  elements. No JavaScript, no Streamlit rerun on toggle.
- The `<summary>` header shows: pattern color dot, pattern name, category,
  merchant count, and total monthly spend. All groups start collapsed, so the
  section is only a few lines tall until the user opens one.
- Inside: a table with columns Merchant, Frequency, Avg purchase,
  Monthly spend, Behaviour. Merchants sorted by monthly spend, descending.
  All merchants are shown (no top-N cutoff); the collapse handles length.
- Frequency is formatted for humans, never as a raw rate:
  - 1.5+ per month: "4/month" (rounded)
  - 0.8 to 1.5: "monthly"
  - 0.4 to 0.8: "every 2 months"
  - 0.28 to 0.4: "every 3 months"
  - below: "1-2 times in 6 months"

## Data flow

Everything numeric already exists. `build_pipeline` in `app.py` returns the
per-merchant feature matrix (`features.build_features`: `tx_per_month`,
`avg_amount`, `monthly_spend`, `interval_regularity`, `amount_stability`) and
the cluster labels. The new section joins those with the pattern names from
the analysis result. No new computation beyond formatting.

## Behaviour phrases

Two sources, same contract as cluster names today (AI when available, local
fallback otherwise):

1. **GPT-5.6 (primary).** Extend the existing single `analyze_clusters` call
   in `core/llm.py`; do not add a second API call. Each cluster's stats gain a
   compact merchant feature table: name, tx_per_month, avg_amount,
   interval_regularity, amount_stability, all rounded. This stays within the
   module's privacy promise (aggregates only, never raw transactions; merchant
   names already appear in `top_merchants`). The response schema gains, per
   cluster, `"merchant_behaviours": {merchant: phrase}`.
   - Validation, alongside the existing response validation: each phrase is a
     string of at most 40 characters, no digits (prevents the model echoing
     amounts), lowercase-ish free text. Unknown merchant keys are ignored.
   - `MAX_OUTPUT_TOKENS` rises from 1200 to 2000 to fit ~40 short phrases.
   - The mock mode (`MOCK_LLM`) returns rule-based phrases so tests are
     deterministic and offline.
2. **Local rules (fallback).** A pure function `behaviour_label(row)` in
   `core/features.py`, applied when the key is missing, the call fails, or a
   merchant is absent from the response. Rules, checked in order:
   - tx_per_month >= 3.5: "frequent small purchases" if avg < 15 else
     "frequent shopping"
   - tx_per_month >= 1.5: "regular shopping" if regularity >= 0.6 else
     "repeat purchases"
   - ~monthly (0.8 to 1.3) and regularity >= 0.8: "regular fixed payment"
     if avg >= 100 (rent-sized), else "recurring subscription" if
     amount_stability >= 0.97, else "steady monthly purchase"
   - every 2 months (0.4 to 0.8) and regularity >= 0.9:
     "recurring bimonthly bill"
   - every 2 months and regularity >= 0.6: "occasional shopping trips"
   - avg >= 100: "occasional big-ticket purchase"
   - otherwise: "one-off purchases"
   Thresholds may be tuned during implementation against the demo data; the
   demo file must produce sensible labels for Mercadona (frequent), Netflix
   (subscription), Canal Isabel II (bimonthly), and Booking.com (big-ticket).

## Error handling

- LLM path unavailable or invalid: silent per-merchant fallback to rules.
  No new warning banner; the existing "generic labels" notice already covers
  degraded AI mode.
- A merchant with a single transaction has regularity 0 and lands in the
  one-off/big-ticket rules, which is correct.

## Testing

- Unit tests for `behaviour_label` and the frequency formatter, one per rule
  branch, in `tests/test_features.py` (new file if none exists).
- `core/llm.py` validation tests: overlong phrase rejected, digits rejected,
  unknown merchants ignored, missing merchants filled by rules.
- The existing end-to-end demo test stays green; add an assertion that the
  pipeline produces a behaviour label for every merchant.

## Out of scope

- Sorting or filtering controls in the table.
- Per-merchant drill-down (transaction lists).
- Changing the clustering itself or the existing cards.
