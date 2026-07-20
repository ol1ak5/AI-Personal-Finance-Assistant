# AI Personal Finance Assistant — Design Spec

**Date:** 2026-07-15
**Context:** OpenAI Build Week 2026
**Track:** Apps for Your Life
**Submission deadline:** July 21, 2026, 5:00 pm PDT

## What it is

AI Personal Finance Assistant is a personal-finance dashboard for understanding spending behaviour.

A user uploads a common bank transaction export (`.xlsx` or `.csv`). The app cleans and normalizes the data, identifies behavioural spending patterns with K-means clustering, uses GPT-5.6 to name those patterns and explain trends in plain English, and lets the user download a PDF report with their dashboard and summary.

The first release focuses on **expenses only**.

## Product promise

> Upload a bank export and understand the habits behind your spending—not just where the money went.

Examples of outputs:

- “Groceries”
- “Recurring subscriptions”
- “Weekend food delivery”
- “Taxi”
- “Your food-delivery spending increased 28% compared with the previous month.”

## Privacy and data handling

AI Personal Finance Assistant is privacy-conscious, but not fully browser-only in its hosted version.

- The public app runs on Streamlit Community Cloud.
- Uploaded files are processed transiently in the app’s server memory.
- AI Personal Finance Assistant does not save uploads, raw transactions, or reports to a database or disk.
- Users never provide an API key.
- The app’s OpenAI API key is stored privately in Streamlit Community Cloud Secrets.
- GPT-5.6 receives only limited information needed for its tasks:
  - Optional AI-assisted column mapping: headers and up to five sample rows, after explicit user consent.
  - Cluster naming and analysis: aggregated cluster statistics only.
- Users can skip AI-assisted mapping and select their columns manually.
- The public demo and submission video use synthetic data only.

The UI must clearly explain this before upload and before any optional AI-assisted mapping request.

## Supported input

Supported file types:

- `.csv`
- `.xlsx`

AI Personal Finance Assistant supports **common bank exports with guided column mapping**. It does not promise to parse every possible spreadsheet format.

Required fields:

- Transaction date
- Transaction description / merchant
- Amount

Optional fields:

- Bank-provided category
- Account
- Currency

The app supports either one signed amount column or separate debit and credit columns. Only expense transactions are included in the dashboard analysis.

## Stack

- Python 3
- Streamlit
- pandas
- scikit-learn
- Plotly
- OpenAI Python SDK using GPT-5.6
- fpdf2
- kaleido
- pytest
- openpyxl for `.xlsx`

React/Vite is intentionally out of scope: this is a seven-day solo build and the Python data stack is a better fit.

## Architecture

Each module is independent and testable without the Streamlit UI.

```text
parser.py
uploaded file → validated, normalized transaction DataFrame

merchants.py
raw transaction descriptions → cleaned merchant identifiers

features.py
normalized DataFrame → feature matrix

clustering.py
feature matrix → K-means clusters and aggregate cluster statistics

llm.py
validated, structured GPT-5.6 requests and responses

pdf.py
dashboard figures + summary → PDF bytes

app.py
Streamlit interface and application orchestration
```

### `parser.py`

Responsibilities:

- Read `.csv` and `.xlsx` files in memory.
- Detect likely headers and sheets.
- Offer local column-name heuristics.
- Optionally request GPT-assisted mapping after user consent.
- Provide manual dropdown mappings if automatic mapping is uncertain.
- Parse dates, amounts, decimal separators, and sign conventions.
- Support European formats such as `1.234,56`.
- Filter transactions to expenses only.
- Drop invalid rows with a visible skipped-row count.
- Return a standard DataFrame:

```text
date | amount | description | category | merchant
```

### `merchants.py`

Responsibilities:

- Clean descriptions such as:

```text
SUPERMARKET 1234 CITY → cleaned merchant identifier
RIDE-SHARE TRIP REFERENCE → cleaned merchant identifier
```

- Remove reference numbers, location noise, and repeated payment metadata.
- Group obvious variations of the same merchant without assigning them to a fixed spending category.
- Preserve the original description for transparency in the dashboard.
- Stay deliberately simple: strip digits and reference codes, collapse whitespace, uppercase, prefix-match. No fuzzy matching, no external merchant databases. Timeboxed to half a day.

### `features.py`

Features are calculated **per merchant**, aggregated over the entire uploaded period — clustering runs once for the whole export, not per month. Monthly behaviour enters as features (frequency, trend) and as the dashboard's monthly breakdown, not as separate clustering runs.

Because uploads cover different period lengths, volume features are **normalized per month** so a 2-month export and a 12-month export produce comparable feature spaces. Recurrence is computed **across months** — a monthly subscription is only detectable over multiple months (within a single month, a Netflix charge and a one-off purchase look identical).

Features include:

- Average monthly spend (normalized, not raw total)
- Transactions per month (normalized, not raw count)
- Average transaction amount
- Amount variation
- Log-transformed average amount
- Day-of-week distribution
- Weekend share
- Recurrence signals: interval regularity between purchases (~weekly / ~monthly cadence) and amount stability
- Optional bank category as a reporting hint

If the export covers fewer than 2 full months, recurrence features are unreliable: the UI must show the detected analysis period and note that subscription detection needs at least 2 months of data.

Features are standardized before clustering.

### `clustering.py`

Responsibilities:

- Cluster **merchants** (not individual transactions) on their aggregated features.
- Select the cluster count using silhouette score, testing `k = 2` through `k_max = min(7, number_of_merchants // 3)`.
- Gracefully fall back to basic merchant/category grouping if any of these hold:
  - fewer than 10 valid expense transactions,
  - fewer than 6 distinct merchants,
  - no valid `k` exists,
  - the best silhouette score is below **0.15** (clusters too weak to be meaningful — K-means always produces groups on demand, and naming noise would mislead the user).

Each cluster includes:

- Number of transactions
- Total spending
- Average transaction amount
- Top normalized merchants
- Top original descriptions
- Time pattern
- Recurrence information
- Monthly trend

K-means identifies behavioural patterns; it is not a fixed category classifier. GPT-5.6 performs the semantic classification step using the calculated cluster evidence and representative cleaned descriptions. It returns a concise, human-readable category and a descriptive cluster name, such as “Groceries,” “Taxi,” or “Weekend food delivery,” without relying on a closed taxonomy. It must not invent merchants, amounts, or trends.

### `llm.py`

This is the only module that calls the OpenAI API. It uses structured JSON responses and validates every response before it reaches the app.

Calls:

1. **Optional column mapping**
   - Sent only after user consent.
   - Input: headers and up to five sample rows.
   - Output: date, amount, description, optional category mapping; date format; sign convention; decimal separator.
2. **Cluster naming and trend summary**
   - Input: aggregate statistics only.
   - Output: a flexible semantic category, short descriptive cluster name, plain-English trends, and evidence-based observations.
   - Classification is generative rather than chosen from a fixed category list. The prompt must require the model to base every label on the supplied cluster evidence and use “Unclear spending pattern” when the evidence is insufficient.
   - Naming and trends should be combined into one call when practical to reduce latency and cost.

`MOCK_LLM=true` returns deterministic canned responses for development and tests.

Failure behaviour:

- Invalid column mapping: one retry with validation feedback, then manual mapping UI.
- Invalid cluster output: one retry, then generic local labels such as “Spending pattern 1.”
- API failure: dashboard still works with local analysis and a clear message.

## Data flow

1. User uploads `.csv` or `.xlsx`.
2. The file is processed in memory.
3. The app uses local header heuristics to identify columns.
4. If uncertain, the user can map columns manually or explicitly opt in to GPT-assisted mapping.
5. The mapping is validated: at least 90% of relevant rows must produce a valid date and numeric amount.
6. pandas normalizes data and filters to expenses only.
7. The app shows skipped-row counts and data-quality warnings.
8. Merchant descriptions are normalized.
9. Merchant-level features are calculated over the full period and merchants are clustered with K-means.
10. Aggregate cluster statistics are sent to GPT-5.6.
11. GPT-5.6 semantically classifies each cluster, returns descriptive names, and writes an evidence-based trends summary.
12. The dashboard renders locally in Streamlit.
13. The user downloads a PDF containing charts and summary, without raw transaction rows.

## Dashboard

The dashboard includes:

- Total spending
- Number of expense transactions
- Average transaction amount
- Largest spending pattern
- Estimated recurring spending
- Cluster breakdown donut or bubble chart
- Monthly spend stacked by cluster
- Cluster table with AI label, total, transaction count, average amount, and top merchants
- Plain-language AI summary
- Download PDF button

The dashboard must make every AI label understandable by showing the merchants and calculated metrics behind it.

## PDF export

The PDF includes:

- Report title and analysis period
- Key spending metrics
- Dashboard charts
- Cluster breakdown
- AI-generated summary
- Privacy note

The PDF excludes raw transaction rows and the original uploaded file.

## Error handling

- Unsupported or malformed file: clear error message and downloadable sample template.
- Empty file: clear error message.
- Fewer than 10 valid expense transactions or fewer than 6 distinct merchants: show a friendly “not enough data” state.
- Invalid date or amount values: skip safely and show the skipped-row count.
- Multiple Excel sheets: choose the best candidate when clear; otherwise ask the user to select a sheet.
- Invalid model JSON: retry once, then use manual/local fallback.
- PDF failure: preserve the dashboard and show a retryable export error.
- No successful GPT response: show local cluster results without AI summary.

## Public-access and cost safeguards

Users do not need API keys, accounts, or logins.

To protect the public app and API budget:

- Limit upload file size.
- Limit total rows per upload.
- Limit analyses per Streamlit session.
- Set strict maximum model output tokens.
- Combine GPT calls where possible.
- Validate all incoming files and all model outputs.
- Never expose or log the OpenAI API key.
- Do not log raw transaction data.
- Configure OpenAI usage alerts and a fixed personal spending limit.
- Provide generic local fallback labels if API quota is unavailable.

## Testing

Use pytest. All tests run with `MOCK_LLM=true` and never require network access.

Synthetic fixtures include three to four deliberately different bank exports:

- Different languages
- Different date formats
- Different sign conventions
- European decimal separators
- Separate debit and credit columns
- One multi-sheet Excel file

Required tests:

- Missing or malformed date and amount values.
- Empty file.
- Fewer than 10 valid expense transactions.
- No valid K-means range.
- Weak clustering fallback (best silhouette below 0.15).
- Invalid GPT JSON.
- Plausible but incorrect GPT column mapping.
- Manual mapping fallback.
- European number parsing.
- Merchant normalization variants.
- Per-month feature normalization: same merchant behaviour over 2 vs 12 months yields comparable features.
- Export shorter than 2 full months: recurrence caveat shown.
- PDF generation end to end.
- API-unavailable fallback behaviour.

## Deployment

- Develop locally first.
- Deploy the working app to Streamlit Community Cloud.
- Keep the OpenAI API key in Streamlit Community Cloud Secrets.
- Use a public GitHub repository if possible; private repositories are also allowed if shared with the required judges.
- Deploy a basic working upload-to-chart version early, then improve it through normal Git pushes.
- Add one free UptimeRobot monitor for uptime alerts only.
- Do not use artificial requests to prevent Streamlit app hibernation; note in the README that the first load may take up to a minute if the app was asleep.
- Include a prominent “Load demo data” button using bundled synthetic transactions.

## Build Week compliance

- Use Codex throughout core development.
- Preserve the relevant Codex session ID for submission.
- Use GPT-5.6 meaningfully for optional mapping, cluster naming, and trend analysis.
- Maintain Git history from day one.
- Add a README section documenting how Codex and GPT-5.6 contributed to the product.
- Provide installation and testing instructions.
- Record a public YouTube demo with audio under three minutes.
- Show synthetic data only in the demo.
- Make the deployed app available for judging.
- Reserve the final day for testing, video recording, README completion, and submission.

## Out of scope

Only consider these after the core experience is stable:

- Chat with your spending data
- Accounts and saved dashboards
- Multi-currency conversion
- Budget goals
- In-browser Stlite/WebAssembly version
- True client-only privacy mode
