"""
Options Market-Maker Exposure Dashboard
========================================
Launch:  streamlit run app.py
"""

import time
from datetime import datetime, timezone, timedelta
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

WIB = timezone(timedelta(hours=7))

from src.data.data_provider import get_provider
from src.engine.exposure import aggregate_by_strike, compute_exposures, summarize

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Options Exposure",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS — dark card UI ────────────────────────────────────────────────

st.markdown("""
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0a0a0f">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}
.stDeployButton {display: none;}

/* Streamlit rerun spinner — make it look nice */
[data-testid="stStatusWidget"] {
    position: fixed !important;
    top: 50% !important; left: 50% !important;
    transform: translate(-50%, -50%) !important;
    z-index: 9999 !important;
    background: rgba(10,10,15,0.92) !important;
    border: 1px solid #1a1a2e !important;
    border-radius: 14px !important;
    padding: 28px 40px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6) !important;
}
/* Smooth fade-in for page content */
.stApp > div { animation: fadeIn 0.4s ease-in; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0a0a0f; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

.card {
    background: linear-gradient(145deg, #111118 0%, #0d0d14 100%);
    border: 1px solid #1a1a2e;
    border-radius: 14px;
    padding: 24px 28px 18px 28px;
    margin-bottom: 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
.card-title { font-size: 1.15rem; font-weight: 700; color: #e2e8f0; margin-bottom: 2px; }
.card-subtitle { font-size: 0.78rem; color: #6b7280; margin-bottom: 14px; }
.card-legend { display: flex; gap: 28px; margin-bottom: 10px; font-size: 0.8rem; color: #9ca3af; }
.legend-green::before {
    content: ''; display: inline-block; width: 12px; height: 12px;
    background: #26a69a; border-radius: 3px; margin-right: 6px; vertical-align: middle;
}
.legend-red::before {
    content: ''; display: inline-block; width: 12px; height: 12px;
    background: #ef5350; border-radius: 3px; margin-right: 6px; vertical-align: middle;
}

.kpi-row { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 18px; }
.kpi-box {
    background: linear-gradient(145deg, #111118 0%, #0d0d14 100%);
    border: 1px solid #1a1a2e; border-radius: 12px;
    padding: 16px 18px; box-shadow: 0 2px 12px rgba(0,0,0,0.4);
}
.kpi-label { font-size: 0.65rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.kpi-value { font-size: 1.2rem; font-weight: 700; color: #e2e8f0; }
.kpi-value.positive { color: #26a69a; }
.kpi-value.negative { color: #ef5350; }
.kpi-value.neutral  { color: #fbbf24; }

.dash-header { font-size: 1.5rem; font-weight: 700; color: #e2e8f0; margin-bottom: 2px; }
.dash-sub { font-size: 0.85rem; color: #6b7280; margin-bottom: 16px; }

section[data-testid="stSidebar"] { background-color: #0d0d14; border-right: 1px solid #1a1a2e; }
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111118 0%, #0d0d14 100%);
    border: 1px solid #1a1a2e; border-radius: 10px; padding: 12px 16px;
}
div[data-testid="stMetric"] label { color: #6b7280; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #fbbf24; font-size: 1.25rem; font-weight: 700; }

/* Regime badge */
.regime-badge {
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}
.regime-badge.long-gamma {
    background: linear-gradient(135deg, rgba(38,166,154,0.15) 0%, rgba(38,166,154,0.05) 100%);
    border: 1px solid rgba(38,166,154,0.4);
}
.regime-badge.short-gamma {
    background: linear-gradient(135deg, rgba(239,83,80,0.15) 0%, rgba(239,83,80,0.05) 100%);
    border: 1px solid rgba(239,83,80,0.4);
}
.regime-badge.neutral-gamma {
    background: linear-gradient(135deg, rgba(251,191,36,0.15) 0%, rgba(251,191,36,0.05) 100%);
    border: 1px solid rgba(251,191,36,0.4);
}
.regime-title { font-size: 1.1rem; font-weight: 700; }
.regime-sub { font-size: 0.8rem; color: #9ca3af; margin-top: 2px; }

/* Key levels */
.levels-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 18px; }
.level-box {
    background: linear-gradient(145deg, #111118 0%, #0d0d14 100%);
    border: 1px solid #1a1a2e; border-radius: 12px;
    padding: 16px 18px;
}
.level-title { font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }
.level-item { font-size: 0.9rem; font-weight: 600; margin-bottom: 4px; }
.level-item.green { color: #26a69a; }
.level-item.red { color: #ef5350; }
.level-item.yellow { color: #fbbf24; }
.level-sub { font-size: 0.7rem; color: #6b7280; }

/* Signal filter */
.signal-box {
    background: linear-gradient(145deg, #111118 0%, #0d0d14 100%);
    border: 1px solid #1a1a2e; border-radius: 12px;
    padding: 18px 24px; margin-bottom: 18px;
}
.signal-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
.signal-title { font-size: 1.05rem; font-weight: 700; color: #e2e8f0; }
.signal-count { font-size: 0.9rem; font-weight: 600; }
.signal-count .buy { color: #26a69a; }
.signal-count .sell { color: #ef5350; }
.signal-detail { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }
.signal-item {
    font-size: 0.78rem; padding: 6px 10px; border-radius: 6px;
    display: flex; justify-content: space-between;
}
.signal-item.buy-item { background: rgba(38,166,154,0.1); border: 1px solid rgba(38,166,154,0.25); color: #26a69a; }
.signal-item.sell-item { background: rgba(239,83,80,0.1); border: 1px solid rgba(239,83,80,0.25); color: #ef5350; }

/* Trading guide */
.guide-box {
    background: linear-gradient(145deg, #111118 0%, #0d0d14 100%);
    border-radius: 12px; padding: 18px 24px; margin-bottom: 18px;
}
.guide-box.buy-guide { border: 1px solid rgba(38,166,154,0.4); }
.guide-box.sell-guide { border: 1px solid rgba(239,83,80,0.4); }
.guide-box.neutral-guide { border: 1px solid rgba(251,191,36,0.4); }
.guide-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 10px; }
.guide-step { font-size: 0.85rem; color: #d1d5db; margin-bottom: 6px; padding-left: 8px; border-left: 3px solid; }
.guide-step.g-green { border-color: #26a69a; }
.guide-step.g-red { border-color: #ef5350; }
.guide-step.g-yellow { border-color: #fbbf24; }
.guide-levels { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; margin-top: 12px; }
.guide-level-box { background: rgba(255,255,255,0.03); border-radius: 8px; padding: 10px 12px; text-align: center; }
.guide-level-label { font-size: 0.65rem; color: #6b7280; text-transform: uppercase; margin-bottom: 4px; }
.guide-level-val { font-size: 1rem; font-weight: 700; }

/* Interpretation text */
.interpretation {
    font-size: 0.8rem;
    color: #9ca3af;
    font-style: italic;
    padding: 8px 0 4px 0;
    border-top: 1px solid #1a1a2e;
    margin-top: 8px;
}

/* Timestamp */
.timestamp { font-size: 0.75rem; color: #4b5563; }

/* Mobile responsive */
@media (max-width: 768px) {
    .kpi-row { grid-template-columns: repeat(3, 1fr) !important; gap: 8px; }
    .kpi-label { font-size: 0.55rem; }
    .kpi-value { font-size: 0.95rem; }
    .card { padding: 16px 14px 12px 14px; margin-bottom: 12px; border-radius: 10px; }
    .card-title { font-size: 1rem; }
    .card-legend { flex-wrap: wrap; gap: 12px; font-size: 0.7rem; }
    .dash-header { font-size: 1.2rem; }
    .block-container { padding-left: 0.5rem; padding-right: 0.5rem; }
    .levels-row { grid-template-columns: 1fr !important; }
    .signal-detail { grid-template-columns: repeat(2, 1fr) !important; }
    .regime-badge { flex-direction: column; align-items: flex-start; }
    .guide-levels { grid-template-columns: repeat(2, 1fr) !important; }
}
</style>
""", unsafe_allow_html=True)

# ── XAUUSD conversion mapping ───────────────────────────────────────────────

CONVERT_MAP = {
    "GLD": {"underlying": "GC=F", "label": "XAUUSD"},
    "SLV": {"underlying": "SI=F", "label": "XAGUSD"},
    "USO": {"underlying": "CL=F", "label": "WTI"},
}


@st.cache_data(ttl=300)
def _fetch_underlying_price(yf_ticker):
    tk = yf.Ticker(yf_ticker)
    info = tk.fast_info
    price = getattr(info, "last_price", None)
    if not price:
        hist = tk.history(period="1d")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
    return float(price) if price else None


def _get_conversion(ticker, spot):
    cfg = CONVERT_MAP.get(ticker.upper())
    if cfg is None:
        return None, None, 1.0
    ul_price = _fetch_underlying_price(cfg["underlying"])
    if ul_price is None or spot == 0:
        return None, cfg["label"], 1.0
    ratio = ul_price / spot
    return ul_price, cfg["label"], ratio

# ── Auto-refresh ─────────────────────────────────────────────────────────────

REFRESH_INTERVALS = {"Off": 0, "1 min": 60, "5 min": 300, "10 min": 600, "15 min": 900}

# ── Top controls (accessible on mobile) ──────────────────────────────────────

provider = get_provider()

ctrl_1, ctrl_2, ctrl_3, ctrl_4 = st.columns([1, 1, 1, 1])

with ctrl_1:
    ticker = st.text_input(
        "Ticker Symbol", value="SPY", max_chars=10,
        help="e.g. SPY, QQQ, AAPL, TSLA, NVDA, GLD",
    ).upper()

spot = provider.fetch_spot(ticker)
ul_price, ul_label, conv_ratio = _get_conversion(ticker, spot)
has_conversion = ul_price is not None

chain = provider.fetch_chain(ticker)
expiry_dates = sorted(chain["expiry"].dt.date.unique())
expiry_labels = ["All Expiries"] + [d.strftime("%Y-%m-%d") for d in expiry_dates]

with ctrl_2:
    selected_label = st.selectbox("Expiry Date", expiry_labels)
    expiry_filter = None if selected_label == "All Expiries" else selected_label

with ctrl_3:
    spot_text = f"${spot:,.2f}"
    if has_conversion:
        spot_text += f"  ({ul_label} ${ul_price:,.2f})"
    st.markdown(
        f'<div style="margin-top:24px;">'
        f'<span style="font-size:0.7rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.06em;">Spot Price</span><br>'
        f'<span style="font-size:1.25rem;font-weight:700;color:#fbbf24;">{spot_text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

with ctrl_4:
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        auto_interval = st.selectbox("Auto Refresh", list(REFRESH_INTERVALS.keys()), index=2)
    with rc2:
        oanda_price = st.number_input(
            "Harga Forex (XAUUSD)",
            min_value=0.0, value=0.0, step=0.1, format="%.2f",
            help="Isi harga XAUUSD dari broker kamu (Exness, IC Markets, dll)"
        )
    with rc3:
        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
        if st.button("Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

interval_sec = REFRESH_INTERVALS[auto_interval]
if interval_sec > 0:
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    elapsed = time.time() - st.session_state.last_refresh
    if elapsed >= interval_sec:
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
        st.rerun()

    remaining = int(interval_sec - elapsed)
    mins, secs = divmod(remaining, 60)
    st.caption(f"Next refresh in {mins}m {secs}s")

    st.markdown(
        f'<script>setTimeout(function(){{window.location.reload();}}, {remaining * 1000});</script>',
        unsafe_allow_html=True,
    )

# ── Sidebar (model params — advanced) ────────────────────────────────────────

st.sidebar.markdown("## Model Params")
st.sidebar.markdown("---")
r = st.sidebar.slider("Risk-Free Rate", 0.0, 0.10, 0.05, 0.005, format="%.3f")
q = st.sidebar.slider("Dividend Yield", 0.0, 0.05, 0.015, 0.005, format="%.3f")


# ── Compute ──────────────────────────────────────────────────────────────────

exposures = compute_exposures(chain, spot, r=r, q=q)
agg = aggregate_by_strike(exposures, expiry_filter=expiry_filter)

filtered = exposures if expiry_filter is None else exposures[
    exposures["expiry"] == pd.Timestamp(expiry_filter)
]
kpi = summarize(filtered, spot)
if kpi.gamma_flip is None and expiry_filter is not None:
    kpi_all = summarize(exposures, spot)
    if kpi_all.gamma_flip is not None:
        from dataclasses import replace
        kpi = replace(kpi, gamma_flip=kpi_all.gamma_flip)

# ── OANDA offset calculation ─────────────────────────────────────────────────

use_oanda = oanda_price > 0 and has_conversion
if use_oanda:
    oanda_offset = (spot * conv_ratio) - oanda_price
else:
    oanda_offset = 0.0

def _to_oanda(xauusd_price):
    if use_oanda:
        return xauusd_price - oanda_offset
    return xauusd_price

def _oanda_str(xauusd_price):
    if use_oanda:
        return f" → Forex **${_to_oanda(xauusd_price):,.2f}**"
    return ""

# ── Header ───────────────────────────────────────────────────────────────────

exp_label = selected_label if expiry_filter else "All Expiries"
header_extra = f"  &middot;  {ul_label} ${ul_price:,.2f}" if has_conversion else ""
if use_oanda:
    header_extra += f"  &middot;  Forex ~<b>${oanda_price:,.2f}</b>  &middot;  Selisih <b>${oanda_offset:+.2f}</b>"
now_wib = datetime.now(WIB).strftime("%H:%M WIB — %d %b %Y")
st.markdown(
    f'<div style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;">'
    f'<div>'
    f'<div class="dash-header">{ticker} Exposure Dashboard</div>'
    f'<div class="dash-sub">Spot ${spot:,.2f}{header_extra}  &middot;  {exp_label}</div>'
    f'</div>'
    f'<div class="timestamp">Last update: {now_wib}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _fmt_short(v):
    abs_v = abs(v)
    if abs_v >= 1e9:
        return f"${v / 1e9:+,.2f}B"
    if abs_v >= 1e6:
        return f"${v / 1e6:+,.2f}M"
    if abs_v >= 1e3:
        return f"${v / 1e3:+,.1f}K"
    return f"${v:+,.0f}"


# ── Market Regime Badge ──────────────────────────────────────────────────────

if kpi.gamma_flip and spot > kpi.gamma_flip:
    regime_class = "long-gamma"
    regime_emoji = "✅"
    regime_text = "LONG GAMMA — STABLE MARKET"
    regime_color = "#26a69a"
elif kpi.gamma_flip and spot <= kpi.gamma_flip:
    regime_class = "short-gamma"
    regime_emoji = "⚠️"
    regime_text = "SHORT GAMMA — VOLATILE MARKET"
    regime_color = "#ef5350"
else:
    regime_class = "neutral-gamma"
    regime_emoji = "⚫"
    regime_text = "NEUTRAL — NO CLEAR REGIME"
    regime_color = "#fbbf24"

cex_bias = "Bullish" if kpi.total_cex > 0 else "Bearish"
cex_bias_color = "#26a69a" if kpi.total_cex > 0 else "#ef5350"

st.markdown(
    f'<div class="regime-badge {regime_class}">'
    f'<div>'
    f'<div class="regime-title" style="color:{regime_color};">{regime_emoji} {regime_text}</div>'
    f'<div class="regime-sub">Spot ${spot:,.2f} vs Flip ${kpi.gamma_flip:,.2f}</div>'
    f'</div>'
    f'<div style="text-align:right;">'
    f'<div style="font-size:0.8rem;color:#9ca3af;">Daily Bias</div>'
    f'<div style="font-size:1rem;font-weight:700;color:{cex_bias_color};">{cex_bias} (CEX {_fmt_short(kpi.total_cex)})</div>'
    f'</div>'
    f'</div>' if kpi.gamma_flip else
    f'<div class="regime-badge {regime_class}">'
    f'<div class="regime-title" style="color:{regime_color};">{regime_emoji} {regime_text}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Key Levels Summary ──────────────────────────────────────────────────────

_pos_gex = agg[agg["gex"] > 0].nlargest(3, "gex")
_neg_gex = agg[agg["gex"] < 0].nsmallest(3, "gex")


def _level_str(strike):
    if has_conversion:
        xau = strike * conv_ratio
        if use_oanda:
            exn = _to_oanda(xau)
            return f"${strike:,.0f} ({ul_label} ${xau:,.0f} → Forex ${exn:,.0f})"
        return f"${strike:,.0f} ({ul_label} ${xau:,.0f})"
    return f"${strike:,.0f}"


res_items = "".join(
    f'<div class="level-item green">{_level_str(r["strike"])}</div>'
    for _, r in _pos_gex.iterrows()
) or '<div class="level-item" style="color:#4b5563;">No resistance levels</div>'

sup_items = "".join(
    f'<div class="level-item red">{_level_str(r["strike"])}</div>'
    for _, r in _neg_gex.iterrows()
) or '<div class="level-item" style="color:#4b5563;">No support levels</div>'

flip_item = f'<div class="level-item yellow">{_level_str(kpi.gamma_flip)}</div>' if kpi.gamma_flip else '<div class="level-item" style="color:#4b5563;">N/A</div>'

st.markdown(
    f'<div class="levels-row">'
    f'<div class="level-box"><div class="level-title">Resistance (MM Sell)</div>{res_items}</div>'
    f'<div class="level-box"><div class="level-title">Support (MM Buy)</div>{sup_items}</div>'
    f'<div class="level-box"><div class="level-title">Gamma Flip</div>{flip_item}'
    f'<div class="level-sub" style="margin-top:8px;">Di atas = stabil<br>Di bawah = volatile</div></div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Signal Filter ────────────────────────────────────────────────────────────

_signals = [
    ("Net Delta", kpi.net_delta > 0),
    ("Net Gamma", kpi.net_gamma > 0),
    ("Total GEX", kpi.total_gex > 0),
    ("Total VEX", kpi.total_vex > 0),
    ("Total CEX", kpi.total_cex > 0),
    ("Spot vs Flip", (kpi.gamma_flip is not None and spot > kpi.gamma_flip)),
]
_buy_count = sum(1 for _, b in _signals if b)
_sell_count = len(_signals) - _buy_count

signal_items = "".join(
    f'<div class="signal-item {"buy-item" if is_buy else "sell-item"}">'
    f'<span>{name}</span><span>{"BUY" if is_buy else "SELL"}</span></div>'
    for name, is_buy in _signals
)

st.markdown(
    f'<div class="signal-box">'
    f'<div class="signal-header">'
    f'<div class="signal-title">\U0001f3af Signal Filter</div>'
    f'<div class="signal-count">'
    f'<span class="buy">BUY ({_buy_count}/{len(_signals)})</span>'
    f' &middot; '
    f'<span class="sell">SELL ({_sell_count}/{len(_signals)})</span>'
    f'</div></div>'
    f'<div class="signal-detail">{signal_items}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Trading Guide ────────────────────────────────────────────────────────────

_is_short_gamma = kpi.gamma_flip is not None and spot <= kpi.gamma_flip
_is_long_gamma = kpi.gamma_flip is not None and spot > kpi.gamma_flip
_cex_bearish = kpi.total_cex < 0

_top_res = agg[agg["gex"] > 0].nlargest(1, "gex")
_top_sup = agg[agg["gex"] < 0].nsmallest(1, "gex")
_res_price = float(_top_res["strike"].iloc[0]) * conv_ratio if len(_top_res) > 0 else None
_sup_price = float(_top_sup["strike"].iloc[0]) * conv_ratio if len(_top_sup) > 0 else None
_flip_price = kpi.gamma_flip * conv_ratio if kpi.gamma_flip else None

if _is_short_gamma and _cex_bearish:
    _guide_class = "sell-guide"
    _guide_color = "#ef5350"
    _guide_emoji = "🔴"
    _guide_title = "SETUP: SELL — Short Gamma + CEX Bearish"
    _guide_steps = [
        ("g-yellow", "1. Tunggu harga NAIK ke resistance"),
        ("g-red", "2. Harga sampai resistance → SELL"),
        ("g-red", "3. SL: $10-15 di atas resistance"),
        ("g-green", "4. TP: Support terdekat"),
    ]
elif _is_long_gamma and not _cex_bearish:
    _guide_class = "buy-guide"
    _guide_color = "#26a69a"
    _guide_emoji = "🟢"
    _guide_title = "SETUP: BUY — Long Gamma + CEX Bullish"
    _guide_steps = [
        ("g-yellow", "1. Tunggu harga TURUN ke support"),
        ("g-green", "2. Harga sampai support → BUY"),
        ("g-green", "3. SL: $10-15 di bawah support"),
        ("g-red", "4. TP: Resistance terdekat"),
    ]
elif _is_short_gamma and not _cex_bearish:
    _guide_class = "neutral-guide"
    _guide_color = "#fbbf24"
    _guide_emoji = "🟡"
    _guide_title = "MIXED — Short Gamma tapi CEX Bullish"
    _guide_steps = [
        ("g-yellow", "1. Hati-hati, sinyal campur"),
        ("g-yellow", "2. Tunggu konfirmasi arah dari price action"),
        ("g-yellow", "3. Lot kecil, SL ketat"),
        ("g-yellow", "4. Atau skip, tunggu sinyal lebih jelas"),
    ]
else:
    _guide_class = "neutral-guide"
    _guide_color = "#fbbf24"
    _guide_emoji = "🟡"
    _guide_title = "MIXED — Long Gamma tapi CEX Bearish"
    _guide_steps = [
        ("g-yellow", "1. Market stabil tapi ada tekanan jual harian"),
        ("g-green", "2. Bisa scalp BUY di support"),
        ("g-red", "3. SL ketat — CEX tekan harga turun pelan"),
        ("g-yellow", "4. Jangan hold lama, ambil profit cepat"),
    ]

steps_html = "".join(
    f'<div class="guide-step {cls}">{text}</div>' for cls, text in _guide_steps
)

levels_html = '<div class="guide-levels">'
if _res_price:
    levels_html += f'<div class="guide-level-box"><div class="guide-level-label">Entry (Sell) / TP (Buy)</div><div class="guide-level-val" style="color:#ef5350;">${_res_price:,.0f}</div></div>'
if _sup_price:
    levels_html += f'<div class="guide-level-box"><div class="guide-level-label">Entry (Buy) / TP (Sell)</div><div class="guide-level-val" style="color:#26a69a;">${_sup_price:,.0f}</div></div>'
if _flip_price:
    levels_html += f'<div class="guide-level-box"><div class="guide-level-label">Gamma Flip</div><div class="guide-level-val" style="color:#fbbf24;">${_flip_price:,.0f}</div></div>'
levels_html += f'<div class="guide-level-box"><div class="guide-level-label">Spot Now</div><div class="guide-level-val" style="color:#e2e8f0;">${spot * conv_ratio:,.0f}</div></div>'
levels_html += '</div>'

st.markdown(
    f'<div class="guide-box {_guide_class}">'
    f'<div class="guide-title" style="color:{_guide_color};">{_guide_emoji} {_guide_title}</div>'
    f'{steps_html}'
    f'{levels_html}'
    f'</div>',
    unsafe_allow_html=True,
)

# ── KPI row ──────────────────────────────────────────────────────────────────


def _fmt(v, prefix="$"):
    abs_v = abs(v)
    if abs_v >= 1e9:
        return f"{prefix}{v / 1e9:+,.2f}B"
    if abs_v >= 1e6:
        return f"{prefix}{v / 1e6:+,.2f}M"
    if abs_v >= 1e3:
        return f"{prefix}{v / 1e3:+,.1f}K"
    return f"{prefix}{v:+,.0f}"


def _color_class(v):
    if v > 0:
        return "positive"
    if v < 0:
        return "negative"
    return "neutral"


def _kpi_box(label, value_str, value_num=None, special_class=""):
    cls = special_class or (_color_class(value_num) if value_num is not None else "")
    return (
        f'<div class="kpi-box">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {cls}">{value_str}</div>'
        f'</div>'
    )


kpi_html = '<div class="kpi-row">'
kpi_html += _kpi_box("Net Delta", _fmt(kpi.net_delta), kpi.net_delta)
kpi_html += _kpi_box("Net Gamma", f"{kpi.net_gamma:,.2f}", kpi.net_gamma)
kpi_html += _kpi_box("Total GEX", _fmt(kpi.total_gex), kpi.total_gex)
kpi_html += _kpi_box("Total VEX", _fmt(kpi.total_vex), kpi.total_vex)
kpi_html += _kpi_box("Total CEX", _fmt(kpi.total_cex), kpi.total_cex)

if kpi.gamma_flip:
    flip_gld = f"${kpi.gamma_flip:,.2f}"
    if has_conversion:
        flip_ul = kpi.gamma_flip * conv_ratio
        flip_gld += f" ({ul_label} ${flip_ul:,.0f})"
        if use_oanda:
            flip_gld += f" → Forex ${_to_oanda(flip_ul):,.0f}"
    kpi_html += _kpi_box("Gamma Flip", flip_gld, special_class="neutral")
else:
    kpi_html += _kpi_box("Gamma Flip", "N/A", special_class="neutral")

kpi_html += '</div>'
st.markdown(kpi_html, unsafe_allow_html=True)

# ── Chart helpers ────────────────────────────────────────────────────────────

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#9ca3af"),
    margin=dict(l=55, r=20, t=10, b=50),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(size=11, color="#9ca3af"), bgcolor="rgba(0,0,0,0)",
    ),
    height=400,
    xaxis=dict(gridcolor="#1a1a2e", zeroline=False),
    yaxis=dict(gridcolor="#1a1a2e", zeroline=True, zerolinecolor="#2a2a3e"),
)


def _spot_marker(fig, spot, gamma_flip):
    spot_label = f"${spot:,.0f}"
    if has_conversion:
        spot_label += f" ({ul_label} ${spot * conv_ratio:,.0f})"
    fig.add_vline(
        x=spot, line_dash="dash", line_color="rgba(251,191,36,0.5)", line_width=1.5,
        annotation_text=f"Spot {spot_label}",
        annotation_font=dict(color="rgba(251,191,36,0.7)", size=10, family="Inter"),
        annotation_bgcolor="rgba(0,0,0,0)",
        annotation_borderwidth=0, annotation_borderpad=3,
        annotation_position="bottom right",
        annotation_yshift=20,
    )
    if gamma_flip is not None:
        flip_label = f"${gamma_flip:,.0f}"
        if has_conversion:
            flip_label += f" ({ul_label} ${gamma_flip * conv_ratio:,.0f})"
        gap_ul = abs(spot - gamma_flip) * conv_ratio
        fig.add_vline(
            x=gamma_flip, line_dash="dot", line_color="#00e5a0", line_width=1.5,
            annotation_text=f"FLIP {flip_label} | Gap ${gap_ul:,.0f}",
            annotation_font=dict(color="#00e5a0", size=11, family="Inter"),
            annotation_bgcolor="rgba(0,229,160,0.12)",
            annotation_bordercolor="rgba(0,229,160,0.4)",
            annotation_borderwidth=1, annotation_borderpad=5,
            annotation_position="top left",
        )


def _add_zones(fig, spot, gamma_flip):
    if gamma_flip is None:
        return
    x_min = fig.layout.xaxis.range[0] if fig.layout.xaxis.range else spot * 0.7
    x_max = fig.layout.xaxis.range[1] if fig.layout.xaxis.range else spot * 1.3
    fig.add_vrect(
        x0=x_min, x1=gamma_flip,
        fillcolor="rgba(239,83,80,0.06)", line_width=0, layer="below",
    )
    fig.add_vrect(
        x0=gamma_flip, x1=x_max,
        fillcolor="rgba(38,166,154,0.06)", line_width=0, layer="below",
    )
    fig.add_annotation(
        x=gamma_flip, y=1, yref="paper", xanchor="right", xshift=-12,
        text="▼ SELL ZONE — ikut arah",
        font=dict(color="rgba(239,83,80,0.6)", size=11, family="Inter"),
        showarrow=False, bgcolor="rgba(0,0,0,0)",
    )
    fig.add_annotation(
        x=gamma_flip, y=1, yref="paper", xanchor="left", xshift=12,
        text="▲ BUY ZONE — lawan arah",
        font=dict(color="rgba(38,166,154,0.6)", size=11, family="Inter"),
        showarrow=False, bgcolor="rgba(0,0,0,0)",
    )


def _card_start(title, emoji, subtitle, legend_green, legend_red, hint="Hover untuk detail"):
    return (
        f'<div class="card">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        f'<div><div class="card-title">{emoji} {title}</div>'
        f'<div class="card-subtitle">{subtitle}</div></div>'
        f'<div style="font-size:0.75rem;color:#4b5563;">{hint}</div></div>'
        f'<div class="card-legend">'
        f'<span class="legend-green">{legend_green}</span>'
        f'<span class="legend-red">{legend_red}</span></div>'
    )


_CARD_END = '</div>'


def _interp(text):
    st.markdown(f'<div class="interpretation">{text}</div>', unsafe_allow_html=True)


def _build_customdata(values, strikes, spot, label_pos="RESISTANCE (MM Sell)", label_neg="SUPPORT (MM Buy)"):
    labels = np.where(
        np.array(values) >= 0,
        label_pos,
        label_neg,
    )
    max_abs = np.max(np.abs(values)) if len(values) > 0 else 1.0
    scores = np.array(values) / (max_abs if max_abs > 0 else 1.0)
    ul_prices = strikes * conv_ratio
    vs_spot_ul = ul_prices - (spot * conv_ratio)
    vs_spot_str = [
        f"+${v:,.2f}" if v >= 0 else f"-${abs(v):,.2f}" for v in vs_spot_ul
    ]
    exness_prices = [f"${_to_oanda(p):,.0f}" if use_oanda else "" for p in ul_prices]
    return list(zip(
        labels,
        [f"${p:,.0f}" for p in ul_prices],
        vs_spot_str,
        [f"{s:+.4f}" for s in scores],
        exness_prices,
    ))


def _make_hover(metric_name):
    exness_line = "Forex: %{customdata[4]}<br>" if use_oanda else ""
    if has_conversion:
        return (
            f"<b>GLD $%{{x:,.0f}} ({ul_label} %{{customdata[1]}})</b><br>"
            f"<span style='color:#26a69a;font-size:13px'>&#9632;</span>"
            f" <span style='color:#fbbf24'>%{{customdata[0]}}</span><br>"
            f"Score: %{{customdata[3]}}<br>"
            f"vs Spot: %{{customdata[2]}}<br>"
            f"{exness_line}"
            f"{metric_name}: $%{{y:+,.0f}}<br>"
            "<extra></extra>"
        )
    return (
        f"<b>Strike $%{{x:,.0f}} ({ticker})</b><br>"
        f"<span style='color:#fbbf24'>%{{customdata[0]}}</span><br>"
        f"Score: %{{customdata[3]}}<br>"
        f"{metric_name}: $%{{y:+,.0f}}<br>"
        "<extra></extra>"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  1. GAMMA EXPOSURE (GEX)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    _card_start(
        "Dealer Gamma Exposure (GEX)", "\U0001f525",
        "Net gamma pressure per strike — positive = resistance, negative = support",
        "Positif — Resistance (MM Sell)",
        "Negatif — Support (MM Buy)",
    )
    + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 20px;margin-top:8px;font-size:0.73rem;">'
    + '<div style="color:#26a69a;font-weight:700;font-size:0.8rem;">LONG GAMMA</div>'
    + '<div style="color:#ef5350;font-weight:700;font-size:0.8rem;">SHORT GAMMA</div>'
    + '<div style="color:#9ca3af;font-weight:600;">Kiri Flip</div>'
    + '<div style="color:#9ca3af;font-weight:600;">Kiri Flip</div>'
    + '<div><span style="color:#26a69a;">■</span> <span style="color:#9ca3af;">Hijau = Resistance</span></div>'
    + '<div><span style="color:#26a69a;">■</span> <span style="color:#9ca3af;">Hijau = Lantai (nahan jatuh)</span></div>'
    + '<div><span style="color:#ef5350;">■</span> <span style="color:#9ca3af;">Merah = Support</span></div>'
    + '<div><span style="color:#ef5350;">■</span> <span style="color:#9ca3af;">Merah = Lubang (jatuh cepat)</span></div>'
    + '<div style="color:#9ca3af;font-weight:600;margin-top:4px;">Kanan Flip</div>'
    + '<div style="color:#9ca3af;font-weight:600;margin-top:4px;">Kanan Flip</div>'
    + '<div><span style="color:#26a69a;">■</span> <span style="color:#9ca3af;">Hijau = Resistance</span></div>'
    + '<div><span style="color:#26a69a;">■</span> <span style="color:#9ca3af;">Hijau = Tembok (nahan naik)</span></div>'
    + '<div><span style="color:#ef5350;">■</span> <span style="color:#9ca3af;">Merah = Support</span></div>'
    + '<div><span style="color:#ef5350;">■</span> <span style="color:#9ca3af;">Merah = Celah (bisa naik lewat)</span></div>'
    + '</div>',
    unsafe_allow_html=True,
)

colors_gex = ["#26a69a" if v >= 0 else "#ef5350" for v in agg["gex"]]
customdata_gex = _build_customdata(agg["gex"].values, agg["strike"].values, spot)

fig_gex = go.Figure()
fig_gex.add_trace(go.Bar(
    x=agg["strike"], y=agg["gex"],
    marker_color=colors_gex, opacity=0.9,
    customdata=customdata_gex,
    hovertemplate=_make_hover("GEX"),
    showlegend=False,
))
_spot_marker(fig_gex, spot, kpi.gamma_flip)
_gex_cap = np.percentile(np.abs(agg["gex"].values), 95) * 1.3 if len(agg) > 0 else 1
fig_gex.update_layout(xaxis_title="Strike Price", yaxis_title="Gamma Exposure ($)", yaxis_range=[-_gex_cap, _gex_cap], **_LAYOUT)
st.plotly_chart(fig_gex, key="gex")
_pos_count = (agg["gex"] > 0).sum()
_neg_count = (agg["gex"] < 0).sum()
if _neg_count > _pos_count:
    _interp(f"Bearish — dealer selling pressure dominan ({_neg_count} strikes negatif vs {_pos_count} positif)")
elif _pos_count > _neg_count:
    _interp(f"Bullish — dealer resistance dominan ({_pos_count} strikes positif vs {_neg_count} negatif)")
else:
    _interp("Netral — tekanan resistance dan support seimbang")
st.markdown(_CARD_END, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  1a. VANNA EXPOSURE (VEX) & CHARM EXPOSURE (CEX)
# ══════════════════════════════════════════════════════════════════════════════

col_v, col_c = st.columns(2)

with col_v:
    st.markdown(
        _card_start(
            "Vanna Exposure (VEX)", "\U0001f300",
            "Delta sensitivity to implied volatility changes",
            "Positif — Magnet (tertarik)",
            "Negatif — Tolak (diusir)",
        )
        + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 20px;margin-top:8px;font-size:0.73rem;">'
        + '<div style="color:#9ca3af;font-weight:600;">Kiri Spot</div>'
        + '<div style="color:#9ca3af;font-weight:600;">Kanan Spot</div>'
        + '<div><span style="color:#26a69a;">■</span> <span style="color:#9ca3af;">Hijau = Magnet turun</span></div>'
        + '<div><span style="color:#26a69a;">■</span> <span style="color:#9ca3af;">Hijau = Magnet naik</span></div>'
        + '<div><span style="color:#ef5350;">■</span> <span style="color:#9ca3af;">Merah = Tolak (lantai, nggak turun jauh)</span></div>'
        + '<div><span style="color:#ef5350;">■</span> <span style="color:#9ca3af;">Merah = Tolak (plafon, nggak naik jauh)</span></div>'
        + '</div>',
        unsafe_allow_html=True,
    )
    colors_vex = ["#26a69a" if v >= 0 else "#ef5350" for v in agg["vex"]]
    customdata_vex = _build_customdata(agg["vex"].values, agg["strike"].values, spot, label_pos="MAGNET (Tertarik)", label_neg="TOLAK (Diusir)")
    fig_vex = go.Figure(go.Bar(
        x=agg["strike"], y=agg["vex"],
        marker_color=colors_vex, opacity=0.9,
        customdata=customdata_vex,
        hovertemplate=_make_hover("VEX"),
        showlegend=False,
    ))
    _spot_marker(fig_vex, spot, kpi.gamma_flip)
    _vex_cap = np.percentile(np.abs(agg["vex"].values), 95) * 1.3 if len(agg) > 0 else 1
    fig_vex.update_layout(xaxis_title="Strike Price", yaxis_title="Vanna Exposure ($)", yaxis_range=[-_vex_cap, _vex_cap], **_LAYOUT)
    st.plotly_chart(fig_vex, key="vex")
    if abs(kpi.total_vex) < 1e3:
        _interp("Netral — perubahan volatilitas tidak trigger hedging signifikan")
    elif kpi.total_vex > 0:
        _interp(f"VEX positif ({_fmt_short(kpi.total_vex)}) — jika IV turun, dealer harus beli (bullish)")
    else:
        _interp(f"VEX negatif ({_fmt_short(kpi.total_vex)}) — jika IV naik, dealer harus jual (bearish)")
    st.markdown(_CARD_END, unsafe_allow_html=True)

with col_c:
    st.markdown(
        _card_start(
            "Charm Exposure (CEX)", "⏳",
            "Daily delta decay — time-driven hedging pressure",
            "Positif — Daily Buying",
            "Negatif — Daily Selling",
        ),
        unsafe_allow_html=True,
    )
    colors_cex = ["#26a69a" if v >= 0 else "#ef5350" for v in agg["cex"]]
    customdata_cex = _build_customdata(agg["cex"].values, agg["strike"].values, spot)
    fig_cex = go.Figure(go.Bar(
        x=agg["strike"], y=agg["cex"],
        marker_color=colors_cex, opacity=0.9,
        customdata=customdata_cex,
        hovertemplate=_make_hover("CEX"),
        showlegend=False,
    ))
    _spot_marker(fig_cex, spot, kpi.gamma_flip)
    _cex_cap = np.percentile(np.abs(agg["cex"].values), 95) * 1.3 if len(agg) > 0 else 1
    fig_cex.update_layout(xaxis_title="Strike Price", yaxis_title="Charm Exposure ($)", yaxis_range=[-_cex_cap, _cex_cap], **_LAYOUT)
    st.plotly_chart(fig_cex, key="cex")
    if kpi.total_cex < -1e6:
        _interp(f"Selling pressure harian kuat ({_fmt_short(kpi.total_cex)}) — bearish drift, hati-hati hold long")
    elif kpi.total_cex > 1e6:
        _interp(f"Buying pressure harian kuat ({_fmt_short(kpi.total_cex)}) — bullish drift")
    else:
        _interp("Charm netral — tidak ada tekanan harian yang signifikan")
    st.markdown(_CARD_END, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  2. ABSOLUTE GAMMA — OPEN INTEREST
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    _card_start(
        "Absolute Gamma — Open Interest", "\U0001f4ca",
        "Call vs Put open interest concentration per strike",
        "Call OI (Upside)",
        "Put OI (Downside)",
    ),
    unsafe_allow_html=True,
)

fig_abs = go.Figure()

if has_conversion:
    call_hover = (
        f"<b>Strike $%{{x:,.0f}} ({ul_label} %{{customdata[0]}})</b><br>"
        "Call OI: %{y:,.0f} contracts<br>"
        "<extra></extra>"
    )
    put_hover = (
        f"<b>Strike $%{{x:,.0f}} ({ul_label} %{{customdata[1]}})</b><br>"
        "Put OI: %{customdata[0]:,.0f} contracts<br>"
        "<extra></extra>"
    )
    ul_strike_str = [f"${s * conv_ratio:,.0f}" for s in agg["strike"]]
    fig_abs.add_trace(go.Bar(
        x=agg["strike"], y=agg["call_oi"], name="Call OI",
        marker_color="#26a69a", opacity=0.9,
        customdata=list(zip(ul_strike_str)),
        hovertemplate=(
            f"<b>Strike $%{{x:,.0f}} ({ul_label} %{{customdata[0]}})</b><br>"
            "Call OI: %{y:,.0f} contracts<br><extra></extra>"
        ),
    ))
    fig_abs.add_trace(go.Bar(
        x=agg["strike"], y=-agg["put_oi"], name="Put OI",
        marker_color="#ef5350", opacity=0.9,
        customdata=list(zip(agg["put_oi"], ul_strike_str)),
        hovertemplate=(
            f"<b>GLD $%{{x:,.0f}} ({ul_label} %{{customdata[1]}})</b><br>"
            "Put OI: %{customdata[0]:,.0f} contracts<br><extra></extra>"
        ),
    ))
else:
    fig_abs.add_trace(go.Bar(
        x=agg["strike"], y=agg["call_oi"], name="Call OI",
        marker_color="#26a69a", opacity=0.9,
        hovertemplate=f"<b>Strike $%{{x:,.0f}} ({ticker})</b><br>Call OI: %{{y:,.0f}} contracts<br><extra></extra>",
    ))
    fig_abs.add_trace(go.Bar(
        x=agg["strike"], y=-agg["put_oi"], name="Put OI",
        marker_color="#ef5350", opacity=0.9,
        customdata=agg["put_oi"],
        hovertemplate=f"<b>Strike $%{{x:,.0f}} ({ticker})</b><br>Put OI: %{{customdata:,.0f}} contracts<br><extra></extra>",
    ))

_spot_marker(fig_abs, spot, kpi.gamma_flip)
_oi_cap = np.percentile(np.concatenate([agg["call_oi"].values, agg["put_oi"].values]), 95) * 1.3 if len(agg) > 0 else 1
fig_abs.update_layout(
    barmode="relative", xaxis_title="Strike Price",
    yaxis_title="Open Interest (Contracts)", yaxis_range=[-_oi_cap, _oi_cap], **_LAYOUT,
)
st.plotly_chart(fig_abs, key="abs_gex")
total_call = agg["call_oi"].sum()
total_put = agg["put_oi"].sum()
pcr = total_put / total_call if total_call > 0 else 0
if pcr > 1.2:
    _interp(f"Put-heavy (P/C ratio {pcr:.2f}) — protective positioning dominan, pasar defensif")
elif pcr < 0.8:
    _interp(f"Call-heavy (P/C ratio {pcr:.2f}) — bullish positioning dominan")
else:
    _interp(f"Seimbang (P/C ratio {pcr:.2f}) — tidak ada bias kuat dari open interest")
st.markdown(_CARD_END, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  1b. HEDGING PRESSURE HEATMAP (normalized -1 to +1)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    _card_start(
        "Dealer Hedging Pressure Heatmap", "\U0001f525",
        "Normalized hedge pressure per strike (-1 = max support, +1 = max resistance)",
        "Positif — Resistance (MM Sell)",
        "Negatif — Support (MM Buy)",
    ),
    unsafe_allow_html=True,
)

gex_vals = agg["gex"].values
max_abs_gex = np.max(np.abs(gex_vals)) if len(gex_vals) > 0 else 1.0
norm_gex = gex_vals / (max_abs_gex if max_abs_gex > 0 else 1.0)

colors_heatmap = ["#26a69a" if v >= 0 else "#ef5350" for v in norm_gex]
customdata_hm = _build_customdata(gex_vals, agg["strike"].values, spot)

fig_hm = go.Figure()
fig_hm.add_trace(go.Bar(
    x=agg["strike"], y=norm_gex,
    marker_color=colors_heatmap, opacity=0.9,
    customdata=customdata_hm,
    hovertemplate=_make_hover("Score"),
    showlegend=False,
))
_spot_marker(fig_hm, spot, kpi.gamma_flip)
fig_hm.update_layout(
    xaxis_title="Strike Price", yaxis_title="Hedge Pressure Score",
    yaxis_range=[-1.1, 1.1], **_LAYOUT,
)
st.plotly_chart(fig_hm, key="heatmap")
max_strike = agg.loc[agg["gex"].idxmax(), "strike"] if len(agg) > 0 else 0
_interp(f"Tekanan terkuat di strike ${max_strike:,.0f}" + (f" ({ul_label} ${max_strike * conv_ratio:,.0f})" if has_conversion else "") + " — perhatikan level ini")
st.markdown(_CARD_END, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  5. NET DELTA EXPOSURE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    _card_start(
        "Net Delta Exposure", "\U0001f3af",
        "Dealer directional exposure per strike",
        "Positif — Long Delta",
        "Negatif — Short Delta",
    ),
    unsafe_allow_html=True,
)

colors_delta = ["#26a69a" if v >= 0 else "#ef5350" for v in agg["delta_exposure"]]
customdata_delta = _build_customdata(agg["delta_exposure"].values, agg["strike"].values, spot)

fig_delta = go.Figure(go.Bar(
    x=agg["strike"], y=agg["delta_exposure"],
    marker_color=colors_delta, opacity=0.9,
    customdata=customdata_delta,
    hovertemplate=_make_hover("Delta"),
    showlegend=False,
))
_spot_marker(fig_delta, spot, kpi.gamma_flip)
_delta_cap = np.percentile(np.abs(agg["delta_exposure"].values), 95) * 1.3 if len(agg) > 0 else 1
fig_delta.update_layout(xaxis_title="Strike Price", yaxis_title="Delta Exposure ($)", yaxis_range=[-_delta_cap, _delta_cap], **_LAYOUT)
st.plotly_chart(fig_delta, key="delta")
if kpi.net_delta > 0:
    _interp(f"Dealer long delta ({_fmt_short(kpi.net_delta)}) — bullish hedge positioning, ada support dari dealer")
else:
    _interp(f"Dealer short delta ({_fmt_short(kpi.net_delta)}) — bearish positioning, dealer ikut tekan harga turun")
st.markdown(_CARD_END, unsafe_allow_html=True)

# ── Raw data ─────────────────────────────────────────────────────────────────

with st.expander("Raw Aggregated Data"):
    st.dataframe(
        agg.style.format({
            "strike": "{:.1f}",
            "gex": "{:,.0f}",
            "vex": "{:,.0f}",
            "cex": "{:,.0f}",
            "delta_exposure": "{:,.0f}",
            "call_gex": "{:,.0f}",
            "put_gex": "{:,.0f}",
            "call_oi": "{:,.0f}",
            "put_oi": "{:,.0f}",
        }),
        height=400,
    )

