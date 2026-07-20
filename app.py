"""AI Personal Finance Assistant walking skeleton: upload a bank export or explore demo data."""
from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.elements.lib.streamlit_plotly_theme import (
    BG_COLOR,
    CATEGORY_0,
    CATEGORY_1,
    CATEGORY_2,
    CATEGORY_3,
    CATEGORY_4,
    GRAY_90,
)

from core import clustering, features, llm, merchants, parser

logger = logging.getLogger(__name__)

DEMO_PATH = Path(__file__).resolve().parent / "data" / "spending_demo.csv"
LIMIT_ANALYSES = 5
MAX_ROWS = 5000
CURRENCY_MARKERS = {
    "€": ("EUR", "€"),
    "$": ("USD", "$"),
    "£": ("GBP", "£"),
    "¥": ("JPY", "¥"),
}


@st.cache_data(show_spinner="Asking GPT-5.6 for analysis…", ttl="24h")
def cached_analysis(stats_json: str, period_json: str, prompt_fingerprint: str) -> dict:
    """One GPT-5.6 call per unique dataset: identical inputs (any rerun,
    any visitor) reuse the cached result instead of paying for a new call.
    The prompt fingerprint busts stale cache entries whenever the analysis
    prompt changes, so a redeployed prompt takes effect immediately."""
    return llm.analyze_clusters(json.loads(stats_json), json.loads(period_json))


@st.cache_data(show_spinner="Asking GPT-5.6 for a summary…", ttl="24h")
def cached_summary(
    stats_json: str, period_json: str, recurring_json: str, prompt_fingerprint: str
) -> str:
    """One summary call per (dataset, selected period): switching periods
    regenerates the summary for that window once, then serves it from cache."""
    return llm.summarize_period(
        json.loads(stats_json), json.loads(period_json), json.loads(recurring_json)
    )


@st.cache_data(show_spinner="Analyzing your transactions…")
def build_pipeline(raw_df: pd.DataFrame, mapping: parser.ColumnMapping):
    """Parse, normalize, and cluster the full export. (stats schema v2)

    Independent of the live period-filter widget and the theme, so a rerun
    triggered by either (or by anything else) reuses this instead of
    re-parsing and re-clustering from scratch every time. The schema note
    above busts st.cache_data when cluster_stats gains fields, since the
    cache key only hashes this function's own source.
    """
    parse_result = parser.apply_mapping(raw_df, mapping)
    all_transactions = merchants.add_merchant_column(parse_result.df)
    feats = cluster_result = None
    all_stats: list[dict] = []
    if len(all_transactions) >= clustering.MIN_TRANSACTIONS:
        feats = features.build_features(all_transactions)
        categories = all_transactions.groupby("merchant")["category"].agg(
            lambda values: values.mode().iat[0] if not values.mode().empty else ""
        )
        cluster_result = clustering.cluster_merchants(
            feats, n_transactions=len(all_transactions), categories=categories
        )
        all_stats = clustering.cluster_stats(all_transactions, cluster_result.labels)
    return parse_result, all_transactions, feats, cluster_result, all_stats


def infer_currency(raw_df: pd.DataFrame) -> str:
    """Return the most common explicit currency marker from an export."""
    sample = " ".join(raw_df.head(200).astype(str).to_numpy().ravel()).upper()
    matches = {
        symbol: sum(sample.count(marker) for marker in markers)
        for symbol, markers in CURRENCY_MARKERS.items()
    }
    return max(matches, key=matches.get) if any(matches.values()) else ""


def format_amount(value: float, currency: str) -> str:
    """Format an amount without inventing a currency where none was exported."""
    return f"{currency}{value:,.2f}"


DONUT_COLORS = ["#7C3AED", "#A78BFA", "#C4B5FD", "#E9D5FF", "#D946EF"]
DONUT_LABEL_INK = ["#F8F7FF", "#2E1065", "#2E1065", "#2E1065", "#F8F7FF"]


def _donut_point(cx: float, cy: float, r: float, a: float) -> str:
    return f"{cx + r * math.cos(a):.2f} {cy + r * math.sin(a):.2f}"


def _donut_sector(cx, cy, r0, r1, a0, a1, corner):
    """SVG path for an annular sector with rounded corners (d3-style arcs).

    Plotly's pie trace cannot round slice corners, so the donut is drawn as
    plain SVG. The corner radius is clamped so tiny slices shrink their
    rounding instead of degenerating (verified against 3% slices).
    """
    span = a1 - a0
    corner = min(corner, (r1 - r0) / 2, r1 * span * 0.35)
    phi_out, phi_in = corner / r1, corner / r0
    large_out = 1 if span - 2 * phi_out > math.pi else 0
    large_in = 1 if span - 2 * phi_in > math.pi else 0
    p = lambda r, a: _donut_point(cx, cy, r, a)  # noqa: E731
    return (
        f"M {p(r1, a0 + phi_out)} "
        f"A {r1} {r1} 0 {large_out} 1 {p(r1, a1 - phi_out)} "
        f"A {corner} {corner} 0 0 1 {p(r1 - corner, a1)} "
        f"L {p(r0 + corner, a1)} "
        f"A {corner} {corner} 0 0 1 {p(r0, a1 - phi_in)} "
        f"A {r0} {r0} 0 {large_in} 0 {p(r0, a0 + phi_in)} "
        f"A {corner} {corner} 0 0 1 {p(r0 + corner, a0)} "
        f"L {p(r1 - corner, a0)} "
        f"A {corner} {corner} 0 0 1 {p(r1, a0 + phi_out)} Z"
    )


def rounded_donut_svg(stats: list[dict], names: dict[int, str], currency: str) -> str:
    """Render the spending-by-pattern donut as SVG with rounded slice corners.

    Hovering a slice reveals a styled callout next to that slice. It uses the
    active Streamlit theme's surface, border, and text variables, so it reads
    like the native chart hover cards in either light or dark mode.
    """
    # A compact canvas keeps the donut visually centred in its card instead
    # of shrinking it against a very wide SVG viewport.
    view_width = 420.0
    outer, inner = 180.0, 116.0
    total = sum(item["total_spend"] for item in stats) or 1.0
    gap = math.radians(3.2)
    angle = -math.pi / 2
    cx = view_width / 2
    cy = 190.0
    view_height = cy + outer + 60
    tip_width, tip_height = 154.0, 48.0

    slice_parts: list[str] = []
    tooltip_parts: list[str] = []
    for i, item in enumerate(stats):
        share = item["total_spend"] / total
        span = share * math.tau
        a0, a1 = angle + gap / 2, angle + span - gap / 2
        if a1 <= a0:
            angle += span
            continue
        color = DONUT_COLORS[i % len(DONUT_COLORS)]
        ink = DONUT_LABEL_INK[i % len(DONUT_LABEL_INK)]
        name = names[item["cluster_id"]]
        short_name = name if len(name) <= 22 else f"{name[:21]}…"
        mid = (a0 + a1) / 2
        lx = cx + (outer + inner) / 2 * math.cos(mid)
        ly = cy + (outer + inner) / 2 * math.sin(mid)
        anchor_x = cx + (outer + 9) * math.cos(mid)
        anchor_y = cy + (outer + 9) * math.sin(mid)
        tip_x = anchor_x + 8 if math.cos(mid) >= 0 else anchor_x - tip_width - 8
        tip_x = max(6, min(tip_x, view_width - tip_width - 6))
        tip_y = max(6, min(anchor_y - tip_height / 2, view_height - tip_height - 6))
        slice_parts.append(
            f'<g class="sl sl-{i}">'
            f'<path d="{_donut_sector(cx, cy, inner, outer, a0, a1, 9)}" '
            f'fill="{color}"></path>'
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
            f'dominant-baseline="central" font-size="13" font-weight="600" '
            f'fill="{ink}" aria-label="{html.escape(name)}">{share:.0%}</text>'
            "</g>"
        )
        # Tooltips are appended after all slices, keeping them above the donut
        # instead of allowing a later slice to cover the hovered callout.
        tooltip_parts.append(
            f'<g class="tip tip-{i}"><rect x="{tip_x:.1f}" y="{tip_y:.1f}" '
            f'width="{tip_width:.1f}" height="{tip_height:.1f}" rx="8" '
            'fill="var(--secondary-background-color, #1B1229)" '
            'stroke="var(--border-color, #3C2E5A)"></rect>'
            f'<text x="{tip_x + 11:.1f}" y="{tip_y + 17:.1f}" '
            'font-size="11.5" font-weight="600" '
            'fill="var(--text-color, #F8F7FF)">'
            f"{html.escape(short_name)}</text>"
            f'<text x="{tip_x + 11:.1f}" y="{tip_y + 35:.1f}" '
            'font-size="12" fill="var(--primary-color, #A78BFA)">'
            f"{format_amount(item['total_spend'], currency)} · {share:.0%}</text></g></g>"
        )
        angle += span

    tooltip_rules = "".join(
        f".sl-{i}:hover ~ .tip-{i}{{opacity:1}}" for i in range(len(stats))
    )
    center_total = html.escape(format_amount(total, currency))
    return (
        f'<svg viewBox="0 0 {view_width:.0f} {view_height:.0f}" role="img" '
        'aria-label="Spending by pattern donut chart" '
        'style="display:block;max-width:420px;margin:0.6rem auto 0.2rem;width:100%;height:auto;'
        'font-family:inherit;">'
        "<style>.tip{opacity:0;transition:opacity .15s;pointer-events:none}"
        f".sl path{{cursor:pointer}}{tooltip_rules}</style>"
        + "".join(slice_parts)
        + "".join(tooltip_parts)
        + f'<text x="{cx}" y="{cy - 18}" text-anchor="middle" font-size="13" '
        f'letter-spacing="1.5" fill="#A78BFA">TOTAL SPENDING</text>'
        f'<text x="{cx}" y="{cy + 18}" text-anchor="middle" font-size="30" '
        f'font-weight="700" fill="#F8F7FF">{center_total}</text>'
        "</svg>"
    )


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
        # table-layout:fixed + one shared colgroup per table keeps every
        # group's columns the same width; all columns but Merchant align right.
        ".habits table{width:100%;border-collapse:collapse;font-size:13px;"
        "table-layout:fixed;}"
        ".habits th{text-align:right;font-size:10.5px;text-transform:uppercase;"
        "letter-spacing:.06em;color:#8B7FA8;padding:9px 18px 7px;}"
        ".habits td{padding:8px 18px;border-top:1px solid rgba(60,46,90,.55);"
        "text-align:right;overflow-wrap:break-word;}"
        ".habits th:first-child,.habits td:first-child{text-align:left;}"
        ".habits td.num{font-variant-numeric:tabular-nums;white-space:nowrap;}"
        ".habits td.beh{color:#C9BFE0;}"
        ".habits .twrap{overflow-x:auto;}"
        # Phone layout (<=640px): Monthly spend is hidden (the group header
        # already shows the pattern's monthly total), padding and font shrink,
        # and column widths flow with content instead of the fixed template.
        "@media (max-width:640px){"
        ".habits table{table-layout:auto;font-size:12px;}"
        ".habits th{padding:8px 10px 6px;}"
        ".habits td{padding:7px 10px;}"
        ".habits th:nth-child(4),.habits td:nth-child(4){display:none;}"
        ".habits summary{padding:11px 12px;gap:7px;}"
        "}"
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
            f"<td style='font-weight:600;'>{html.escape(merchant)}</td>"
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
            "<div class='twrap'>"
            "<table>"
            "<colgroup><col style='width:26%'><col style='width:17%'>"
            "<col style='width:17%'><col style='width:17%'>"
            "<col style='width:23%'></colgroup>"
            "<thead><tr><th>Merchant</th><th>Frequency</th>"
            "<th>Avg purchase</th><th>Monthly spend</th>"
            "<th>Behaviour</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            "</div>"
            "</details>"
        )
    return css + "<div class='habits'>" + "".join(groups) + "</div>"


def build_figures(stats: list[dict], names: dict[int, str], currency: str = "") -> dict:
    """Build the dashboard charts (also reused by the PDF export).

    Uses Streamlit's own theme-placeholder colors (CATEGORY_0.., BG_COLOR,
    GRAY_90) instead of hardcoded light/dark hex values. Streamlit's frontend
    swaps these for the active theme's real colors client-side, the instant
    the user toggles light/dark - no rerun, no per-theme branching in Python,
    and it automatically uses our own chartCategoricalColors from
    .streamlit/config.toml rather than a generic default. This requires the
    figure's template to be set to "streamlit", or the swap never engages.
    """
    donut_colors = [CATEGORY_0, CATEGORY_1, CATEGORY_2, CATEGORY_3, CATEGORY_4]
    total_spend = sum(item["total_spend"] for item in stats)
    donut = px.pie(
        values=[item["total_spend"] for item in stats],
        names=[names[item["cluster_id"]] for item in stats],
        hole=0.55,
        title="Spending by pattern",
        color_discrete_sequence=donut_colors,
        template="streamlit",
    )
    donut.update_traces(
        sort=False,
        # The separators already create the intended breathing room. Pulling each
        # slice apart makes small segments look accidentally detached.
        pull=[0] * len(stats),
        texttemplate="%{percent:.0%}",
        textposition="inside",
        textfont={"color": GRAY_90, "size": 14},
        marker={"line": {"color": BG_COLOR, "width": 7}},
        hovertemplate=(
            f"%{{label}}<br>{currency}%{{value:,.2f}} (%{{percent}})<extra></extra>"
        ),
    )
    donut.update_layout(
        showlegend=False,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font={"color": GRAY_90},
        margin={"l": 12, "r": 12, "t": 56, "b": 12},
        title={"x": 0, "xanchor": "left"},
        annotations=[
            {
                "text": "TOTAL SPENDING",
                "showarrow": False,
                "x": 0.5,
                "y": 0.55,
                "font": {"color": CATEGORY_0, "size": 13},
            },
            {
                "text": f"<b>{format_amount(total_spend, currency)}</b>",
                "showarrow": False,
                "x": 0.5,
                "y": 0.47,
                "font": {"color": GRAY_90, "size": 40},
            }
        ],
    )
    monthly_rows = [
        {"month": month, "spend": spend}
        for item in stats
        for month, spend in item["monthly_totals"].items()
    ]
    monthly_spend = (
        pd.DataFrame(monthly_rows).groupby("month", as_index=False)["spend"].sum()
    )
    month_labels = pd.PeriodIndex(monthly_spend["month"], freq="M").strftime(
        "%b %Y"
    )
    monthly_spend["month"] = month_labels
    # In-app titles are rendered as markdown above each chart so all three
    # sit at the same level; the figures stay title-free here.
    monthly = px.bar(
        monthly_spend,
        x="month",
        y="spend",
        color_discrete_sequence=[CATEGORY_0],
        template="streamlit",
    )
    # Bar width is relative to one category slot: with one month the slot is
    # the whole axis, so a fixed 0.42 renders enormous. Scaling by month count
    # keeps the on-screen bar width constant regardless of the period chosen.
    bar_width = min(0.42, 0.14 * max(len(month_labels), 1))
    # Matches the SVG donut's hover callout (panel background, subtle border,
    # bold title line + plain amount line). Literal colors are safe in the
    # dark-only theme, same as the gradient below.
    chart_hover_card = {
        "bgcolor": "#1B1229",
        "bordercolor": "#3C2E5A",
        "font": {"color": "#F8F7FF", "size": 12},
        "align": "left",
    }
    monthly.update_traces(
        width=bar_width,
        marker={"cornerradius": 8},
        texttemplate=f"{currency}%{{y:,.0f}}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            f"<b>%{{x}}</b><br>"
            f"<span style='color:#A78BFA'>{currency}%{{y:,.2f}}</span>"
            "<extra></extra>"
        ),
        hoverlabel=chart_hover_card,
    )
    monthly.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=month_labels.tolist(),
        title=None,
    )
    monthly.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font={"color": GRAY_90},
        margin={"l": 12, "r": 12, "t": 24, "b": 12},
        hoverlabel=chart_hover_card,
    )
    monthly.update_yaxes(title=None, tickprefix=currency, tickformat=",.0f")

    cumulative_spend = monthly_spend.copy()
    cumulative_spend["cumulative_spend"] = cumulative_spend["spend"].cumsum()
    cumulative = px.area(
        cumulative_spend,
        x="month",
        y="cumulative_spend",
        color_discrete_sequence=[CATEGORY_0],
        template="streamlit",
    )
    cumulative.update_traces(
        # Dark-only theme, so a literal violet with real alpha is safe here and
        # gives a visible fade: strong under the line, dissolving to fully
        # transparent at the bottom (placeholder colors cannot carry alpha).
        fillgradient={
            "type": "vertical",
            "colorscale": [
                [0.0, "rgba(124, 58, 237, 0.0)"],
                [1.0, "rgba(124, 58, 237, 0.55)"],
            ],
        },
        line={"width": 0},
        opacity=1,
        hoverinfo="skip",
    )
    point_labels = [
        f"{currency}{value:,.0f}" for value in cumulative_spend["cumulative_spend"]
    ]
    cumulative.add_scatter(
        x=cumulative_spend["month"],
        y=cumulative_spend["cumulative_spend"],
        mode="lines+markers+text",
        line={"color": CATEGORY_0, "width": 3, "shape": "linear"},
        marker={"color": CATEGORY_0, "size": 7},
        text=point_labels,
        textposition="top center",
        textfont={"color": GRAY_90, "size": 12},
        hovertemplate=(
            f"<b>%{{x}}</b><br>"
            f"<span style='color:#A78BFA'>{currency}%{{y:,.2f}}</span>"
            "<extra></extra>"
        ),
        hoverlabel=chart_hover_card,
        showlegend=False,
    )
    cumulative.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=month_labels.tolist(),
        title=None,
    )
    cumulative.update_yaxes(title=None, tickprefix=currency, tickformat=",.0f")
    cumulative.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font={"color": GRAY_90},
        margin={"l": 12, "r": 12, "t": 24, "b": 12},
        showlegend=False,
        hoverlabel=chart_hover_card,
    )
    return {"donut": donut, "monthly": monthly, "cumulative": cumulative}

st.set_page_config(page_title="AI Personal Finance Assistant", page_icon="📊", layout="wide")

try:
    cloud_api_key = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    cloud_api_key = ""
if cloud_api_key:
    os.environ.setdefault("OPENAI_API_KEY", cloud_api_key)

st.markdown(
    "<p style='font-size:0.95rem;font-weight:500;letter-spacing:0.02em;"
    "opacity:0.7;margin:0 0 0.35rem;'>YOUR PERSONAL FINANCE ASSISTANT</p>",
    unsafe_allow_html=True,
)
st.title("See the patterns behind your spending", anchor=False)
st.caption("Don't give up control of your data. Files are processed in memory and never stored.")

# The demo button mirrors the height of the uploader's dropzone so the two
# boxes read as equal-weight choices. Scoped to primary buttons only (the
# demo button is the app's single primary action).
st.markdown(
    "<style>div[data-testid='stButton'] button[kind='primary']"
    "{height: 68px;}</style>",
    unsafe_allow_html=True,
)
with st.container(border=True):
    upload_column, demo_column = st.columns([4, 1])
    with upload_column.container(height="stretch"):
        uploaded = st.file_uploader(
            "Start with a bank export (.csv or .xlsx, max 5 MB)", type=["csv", "xlsx"]
        )
    with demo_column.container(height="stretch", vertical_alignment="bottom"):
        use_demo = st.button("Explore demo data", type="primary", width="stretch")

raw_df: pd.DataFrame | None = None
frames: dict[str, pd.DataFrame] | None = None
# Persist the chosen source across reruns: st.button is only True on the
# click's own rerun, so without this the dashboard vanishes on any
# subsequent interaction (consent checkbox, sheet selector, ...).
if use_demo:
    st.session_state["source"] = (DEMO_PATH.read_bytes(), DEMO_PATH.name)
    st.session_state["analyses"] = st.session_state.get("analyses", 0) + 1
elif uploaded is not None:
    uploaded_source = (uploaded.getvalue(), uploaded.name)
    if st.session_state.get("source") != uploaded_source:
        st.session_state["source"] = uploaded_source
        st.session_state["analyses"] = st.session_state.get("analyses", 0) + 1

st.session_state.setdefault("analyses", 0)
if st.session_state["analyses"] > LIMIT_ANALYSES:
    st.error("Demo limit reached for this session - refresh to start over.")
    st.stop()

if "source" in st.session_state:
    source_bytes, source_name = st.session_state["source"]
    try:
        frames = parser.load_frames(source_bytes, source_name)
    except parser.ParserError as error:
        st.error(str(error))
        st.stop()

if frames is not None:
    sheet = parser.pick_best_sheet(frames)
    if sheet is None:
        sheet = st.selectbox("Which sheet holds your transactions?", list(frames))
    raw_df = frames[sheet]

if raw_df is not None and len(raw_df) > MAX_ROWS:
    st.error(f"File has {len(raw_df)} rows; the demo supports up to {MAX_ROWS}.")
    st.stop()

if raw_df is not None:
    # Everything below can fail in ways we haven't anticipated (a bad file,
    # a network hiccup, an API surprise). The page must never show a raw
    # traceback to a visitor, so the whole analysis runs under one guard
    # that always degrades to a friendly message instead of crashing.
    try:
        mapping = parser.guess_mapping(raw_df)
        mapping_score = parser.validate_mapping(raw_df, mapping) if mapping else 0.0
        if mapping is None or mapping_score < parser.MAPPING_MIN_VALID:
            st.warning("We could not automatically identify the transaction columns.")
            manual_tab, ai_tab = st.tabs(["Map columns manually", "AI-assisted mapping"])

            with manual_tab:
                columns = ["(none)"] + list(raw_df.columns)
                with st.form("manual_mapping"):
                    date_column = st.selectbox("Date column", columns)
                    description_column = st.selectbox("Description column", columns)
                    amount_column = st.selectbox("Amount column (signed)", columns)
                    category_column = st.selectbox("Category column (optional)", columns)
                    decimal_separator = st.radio(
                        "Decimal separator", [".", ","], horizontal=True
                    )
                    expense_convention = st.radio(
                        "Expenses are…", ["negative", "positive"], horizontal=True
                    )
                    apply_manual_mapping = st.form_submit_button("Apply manual mapping")
                if apply_manual_mapping and "(none)" not in (
                    date_column,
                    description_column,
                    amount_column,
                ):
                    mapping = parser.ColumnMapping(
                        date_col=date_column,
                        description_col=description_column,
                        amount_col=amount_column,
                        category_col=None if category_column == "(none)" else category_column,
                        decimal_separator=decimal_separator,
                        expenses_are=expense_convention,
                    )

            with ai_tab:
                st.caption(
                    "With your consent, this sends only the header row and up to five "
                    "sample rows to GPT-5.6. Nothing else is sent for column mapping."
                )
                consent = st.checkbox(
                    "I agree to send headers and five sample rows to OpenAI",
                    key="mapping_consent",
                )
                if consent and st.button("Detect columns with AI", key="ai_mapping"):
                    if not llm.is_configured():
                        st.error(
                            "AI-assisted mapping is not configured. Please map columns manually."
                        )
                    else:
                        try:
                            mapped_values = llm.map_columns(
                                list(raw_df.columns),
                                raw_df.head(5).astype(str).values.tolist(),
                            )
                            mapping = parser.ColumnMapping(
                                **{
                                    key: value
                                    for key, value in mapped_values.items()
                                    if value is not None
                                }
                            )
                        except Exception as error:
                            st.error("AI mapping failed. Please map columns manually.")
                            logger.warning("AI column mapping failed: %s", error)

            mapping_score = parser.validate_mapping(raw_df, mapping) if mapping else 0.0
            if mapping is None:
                st.stop()
            if mapping_score < parser.MAPPING_MIN_VALID:
                st.error(
                    "That mapping does not parse at least 90% of rows. "
                    "Please check the selected columns and format."
                )
                st.stop()

        parse_result, all_transactions, feats, cluster_result, all_stats = build_pipeline(
            raw_df, mapping
        )
        if parse_result.skipped_rows:
            st.info(f"Skipped {parse_result.skipped_rows} rows that could not be parsed.")

        if len(all_transactions) < clustering.MIN_TRANSACTIONS:
            st.warning("Not enough expense transactions to find patterns (need at least 10).")
            st.stop()

        period_choice = st.segmented_control(
            "Analysis period",
            options=["1 month", "3 months", "6 months", "All history"],
            default="3 months",
            selection_mode="single",
            width="content",
        )
        transactions = all_transactions
        if period_choice != "All history":
            selected_months = int(period_choice.split()[0])
            latest_month = all_transactions["date"].max().to_period("M")
            first_month = latest_month - (selected_months - 1)
            transactions = all_transactions[
                all_transactions["date"] >= first_month.start_time
            ].copy()
        if transactions.empty:
            st.warning("There are no transactions in this period.")
            st.stop()

        currency = infer_currency(raw_df)
        # feats/cluster_result/all_stats came from the cached full-export pipeline
        # above; changing the display window must not relabel the same merchant
        # each time, so only this cheap re-aggregation reruns per period choice.
        stats = clustering.cluster_stats(transactions, cluster_result.labels)
        start, end, months = features.analysis_period(transactions)
        period = {
            "start": str(start.date()),
            "end": str(end.date()),
            "months": round(months, 1),
            "currency": currency,
        }
        source_start, source_end, source_months = features.analysis_period(
            all_transactions
        )
        source_period = {
            "start": str(source_start.date()),
            "end": str(source_end.date()),
            "months": round(source_months, 1),
            "currency": currency,
        }

        # Per-merchant features ride inside the same single analysis call so
        # GPT-5.6 can phrase each merchant's habit (still aggregates only).
        merchant_features_by_cluster = {
            int(cluster_id): [
                [
                    merchant,
                    round(float(row["tx_per_month"]), 2),
                    round(float(row["avg_amount"]), 2),
                    round(float(row["interval_regularity"]), 2),
                    round(float(row["amount_stability"]), 2),
                ]
                # Top spenders only: the tail gets rule-based phrases anyway,
                # and a smaller response is markedly faster to generate.
                for merchant, row in feats[cluster_result.labels == cluster_id]
                .sort_values("monthly_spend", ascending=False)
                .head(10)
                .iterrows()
            ]
            for cluster_id in cluster_result.labels.unique()
        }
        stats_for_ai = [
            {**item, "merchant_features": merchant_features_by_cluster.get(item["cluster_id"], [])}
            for item in all_stats
        ]

        ai_error: str | None = None
        used_ai = False
        if llm.is_configured():
            try:
                analysis = cached_analysis(
                    json.dumps(stats_for_ai, sort_keys=True),
                    json.dumps(source_period, sort_keys=True),
                    hashlib.sha256(llm.ANALYSIS_SYSTEM.encode()).hexdigest()[:12],
                )
                used_ai = True
            except Exception as error:
                # Any AI/network failure (bad key, no billing, rate limit, timeout,
                # ...) falls back to local labels. The dashboard must still work.
                analysis = llm.generic_labels(all_stats)
                ai_error = str(error)
                logger.warning("AI analysis failed, using local fallback: %s", error)
        else:
            analysis = llm.generic_labels(all_stats)
        names = {item["cluster_id"]: item["name"] for item in analysis["clusters"]}
        emojis = {item["cluster_id"]: item["emoji"] for item in analysis["clusters"]}

        ai_behaviours: dict[str, str] = {}
        for item in analysis["clusters"]:
            ai_behaviours.update(item.get("merchant_behaviours", {}))

        def merchant_behaviour(merchant: str) -> str:
            """AI phrase when available, rule-based fallback otherwise."""
            if merchant in ai_behaviours:
                return ai_behaviours[merchant]
            row = feats.loc[merchant]
            return features.behaviour_label(
                row["tx_per_month"], row["avg_amount"],
                row["interval_regularity"], row["amount_stability"],
            )

        # The summary follows the selected period, unlike names/behaviours,
        # which stay full-export so switching periods never renames patterns.
        # The assigned names travel with the window stats as fixed facts.
        summary_text = llm.generic_labels(all_stats)["summary"]
        if used_ai:
            stats_for_summary = [
                {**item, "name": names[item["cluster_id"]]} for item in stats
            ]
            try:
                summary_text = cached_summary(
                    json.dumps(stats_for_summary, sort_keys=True),
                    json.dumps(period, sort_keys=True),
                    json.dumps(features.recurring_charges(feats), sort_keys=True),
                    hashlib.sha256(llm.SUMMARY_SYSTEM.encode()).hexdigest()[:12],
                )
            except Exception as error:
                ai_error = ai_error or str(error)
                logger.warning("AI summary failed, using local fallback: %s", error)

        st.caption("YOUR SPENDING OVERVIEW")
        st.subheader("Patterns, not just transactions", anchor=False)
        if used_ai:
            st.caption("AI-generated classifications are grounded in the cluster evidence below.")
        else:
            st.caption("Generic labels are shown because AI analysis is not available right now.")
        if ai_error:
            with st.expander("Why am I seeing generic labels instead of AI-generated ones?"):
                st.write(
                    "AI analysis could not complete, so local pattern labels are shown "
                    "instead. Everything else below is unaffected."
                )
                st.code(ai_error)

        total = transactions["amount"].sum()
        recurring_merchants = feats[
            (feats["interval_regularity"] > 0.7)
            & (feats["amount_stability"] > 0.8)
        ].index
        recurring_spend = transactions[
            transactions["merchant"].isin(recurring_merchants)
        ]["amount"].sum()
        biggest = names[stats[0]["cluster_id"]]
        biggest_share = stats[0]["total_spend"] / total if total else 0
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([1.15, 0.75, 1, 1, 1.25])
            c1.metric(
                "Total spending",
                format_amount(total, currency),
                help="Sum of all expenses in the selected period.",
            )
            c2.metric(
                "Transactions",
                len(transactions),
                help="Number of expense transactions in the selected period.",
            )
            c3.metric(
                "Avg transaction",
                format_amount(transactions["amount"].mean(), currency),
                help="Average amount per expense transaction in the selected period.",
            )
            c4.metric(
                "Top pattern share",
                f"{biggest_share:.0%}",
                help=f"Largest pattern: {biggest}",
            )
            c5.metric(
                "Est. recurring / month",
                format_amount(recurring_spend / months, currency),
                help=(
                    "Merchants that charge you at regular intervals with a "
                    "stable amount (subscriptions), summed per average month."
                ),
            )

        st.caption(
            f"Analysis period: {period['start']} to {period['end']} "
            f"({period['months']} months)"
        )
        if source_months < 2:
            st.warning(
                "The uploaded export covers less than two calendar months, so "
                "recurring subscriptions may not be detected reliably."
            )
        elif months < 2:
            st.markdown(
                """
                <div style="background:#E9D5FF; border:1px solid #C4B5FD;
                border-radius:0.5rem; color:#6D28D9; font-size:0.875rem;
                padding:0.7rem 1rem;">
                Subscription estimates use the full export history, so they stay
                consistent while you compare shorter periods.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.space(8)
        if cluster_result.used_fallback:
            st.info(
                "The data didn't form strong behavioural clusters "
                f"(reason: {cluster_result.reason}). Showing basic grouping instead."
            )

        figs = build_figures(stats, names, currency)
        currency_unit = f" ({currency})" if currency else ""
        cumulative_box = st.container(border=True)
        cumulative_box.markdown(f"**Cumulative spending{currency_unit}**")
        cumulative_box.plotly_chart(figs["cumulative"], width="stretch")
        left, right = st.columns(2)
        donut_box = left.container(border=True, height="stretch")
        donut_box.markdown("**Spending by pattern (%)**")
        # st.html strips SVG via DOMPurify (verified empirically); markdown with
        # unsafe_allow_html renders it. The SVG is generated locally with
        # html.escape()d labels, so nothing user-controlled is injected raw.
        donut_box.markdown(
            rounded_donut_svg(stats, names, currency), unsafe_allow_html=True
        )
        monthly_box = right.container(border=True, height="stretch")
        monthly_box.markdown(f"**Monthly spending{currency_unit}**")
        monthly_box.plotly_chart(figs["monthly"], width="stretch")

        st.subheader("Patterns in detail", anchor=False)
        st.caption(
            "Merchants are grouped by spending behaviour, such as frequency, typical "
            "amount, and consistency, not only by merchant type."
        )
        categories_by_id = {
            c["cluster_id"]: c.get("category", "") for c in analysis["clusters"]
        }
        # All cards render inside one CSS grid so each row's pair stretches to
        # the same height regardless of how many merchant rows a card has
        # (the merchant list absorbs the slack via flex).
        cards_html: list[str] = []
        for idx, item in enumerate(stats):
            color = DONUT_COLORS[idx % len(DONUT_COLORS)]
            share = item["total_spend"] / total if total else 0
            merchant_rows = item.get(
                "top_merchant_items",
                [[m, None] for m in item["top_merchants"]],
            )[:3]
            rows_html = "".join(
                "<div style='display:flex;justify-content:space-between;gap:10px;"
                "padding:7px 2px;border-top:1px solid rgba(60,46,90,.55);"
                "font-size:13px;'>"
                f"<span style='color:#C9BFE0;'>{html.escape(str(m))}</span>"
                f"<span style='font-weight:600;white-space:nowrap;'>"
                f"{format_amount(a, currency) if a is not None else ''}</span></div>"
                for m, a in merchant_rows
            )
            subtitle_bits = [categories_by_id.get(item["cluster_id"], "")]
            if item.get("n_merchants"):
                subtitle_bits.append(f"{item['n_merchants']} merchants")
            subtitle = " · ".join(bit for bit in subtitle_bits if bit)
            cards_html.append(
                "<div style='background:#171027;border:1px solid #3C2E5A;"
                "border-radius:16px;padding:16px 18px;display:flex;"
                "flex-direction:column;'>"
                "<div style='display:flex;justify-content:space-between;"
                "align-items:flex-start;gap:10px;'>"
                "<div style='display:flex;gap:10px;'>"
                f"<div style='width:5px;border-radius:3px;background:{color};"
                "align-self:stretch;'></div>"
                "<div>"
                f"<p style='font-size:18px;font-weight:700;margin:0;line-height:1.3;'>"
                f"{html.escape(names[item['cluster_id']])}</p>"
                f"<p style='font-size:11.5px;color:#8B7FA8;margin:2px 0 0;'>"
                f"{html.escape(subtitle)}</p>"
                "</div></div>"
                "<span style='font-size:11.5px;font-weight:700;color:#C9BFE0;"
                "background:#241A38;border-radius:999px;padding:4px 10px;"
                f"white-space:nowrap;'>{item['n_transactions']} transactions</span>"
                "</div>"
                "<div style='display:flex;justify-content:space-between;"
                "align-items:baseline;margin-top:12px;'>"
                f"<p style='font-size:19px;font-weight:700;margin:0;'>"
                f"{format_amount(item['total_spend'], currency)}</p>"
                f"<span style='font-size:12px;color:#8B7FA8;'>{share:.0%} of "
                "spending</span></div>"
                "<div style='height:8px;background:#241A38;border-radius:999px;"
                "margin-top:8px;overflow:hidden;'>"
                f"<div style='width:{share:.0%};height:100%;border-radius:999px;"
                f"background:{color};'></div></div>"
                "<p style='font-size:10.5px;letter-spacing:.06em;"
                "text-transform:uppercase;color:#8B7FA8;margin:12px 0 0;'>"
                "Top merchants</p>"
                f"<div style='flex:1;'>{rows_html}</div>"
                "</div>"
            )
        st.markdown(
            # margin-bottom mirrors the charts row above "Patterns in detail",
            # so both section headings get the same breathing room. The grid
            # lives in a class so the phone layout (single column under 640px)
            # can override it without touching the desktop rendering.
            "<style>"
            ".pattern-cards{display:grid;margin-bottom:16px;"
            "grid-template-columns:repeat(2, minmax(0, 1fr));"
            "gap:16px;align-items:stretch;}"
            "@media (max-width:640px){"
            ".pattern-cards{grid-template-columns:1fr;}"
            "}"
            "</style>"
            "<div class='pattern-cards'>" + "".join(cards_html) + "</div>",
            unsafe_allow_html=True,
        )

        st.subheader("Merchant habits", anchor=False)
        st.caption("Your spending habits, merchant by merchant.")
        st.markdown(
            merchant_habits_html(
                stats, names, feats, cluster_result.labels,
                merchant_behaviour, currency,
            ),
            unsafe_allow_html=True,
        )

        st.subheader("What GPT-5.6 sees in your spending", anchor=False)
        st.markdown(summary_text)
    except Exception as error:
        st.error(
            "Something unexpected happened while analyzing this file. "
            "Please try again, or use the bundled demo data instead."
        )
        logger.exception("Unhandled error while processing an upload")
        with st.expander("Technical details"):
            st.code(str(error))
