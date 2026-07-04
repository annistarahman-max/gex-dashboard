"""
Options Market-Maker Exposure Dashboard
========================================
Launch:  streamlit run app.py
"""

import contextlib
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

WIB = timezone(timedelta(hours=7))
_ET = ZoneInfo("America/New_York")

# NYSE calendar 2026 — full holidays and early-close days (13:00 ET)
_US_HOLIDAYS_2026 = {
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Presidents Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
}
_US_EARLY_CLOSE_2026 = {
    "2026-07-03": "13:00",  # Jul 4 falls Sat → Fri early close
    "2026-11-27": "13:00",  # Day after Thanksgiving
    "2026-12-24": "13:00",  # Christmas Eve
}

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
    background: rgba(14,17,23,0.95) !important;
    border: 1px solid #232936 !important;
    border-radius: 14px !important;
    padding: 28px 40px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6) !important;
}
/* Smooth fade-in for page content */
.stApp > div { animation: fadeIn 0.4s ease-in; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0e1117; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

.card {
    background: #161b26;
    border: 1px solid #232936;
    border-radius: 12px;
    padding: 22px 26px 16px 26px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.35);
}
.card-title { font-size: 1.05rem; font-weight: 700; color: #e6e9ef; margin-bottom: 2px; letter-spacing: -0.01em; }
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

.kpi-row { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-bottom: 20px; }
.kpi-box {
    background: #161b26;
    border: 1px solid #232936; border-radius: 10px;
    padding: 14px 16px; box-shadow: none;
}
.kpi-label { font-size: 0.65rem; color: #9aa4b2; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.kpi-value { font-size: 1.2rem; font-weight: 700; color: #e6e9ef; font-variant-numeric: tabular-nums; }
.kpi-value.positive { color: #26a69a; }
.kpi-value.negative { color: #ef5350; }
.kpi-value.neutral  { color: #fbbf24; }

.dash-header { font-size: 1.5rem; font-weight: 700; color: #e2e8f0; margin-bottom: 2px; }
.dash-sub { font-size: 0.85rem; color: #6b7280; margin-bottom: 16px; }

section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #232936; }
div[data-testid="stMetric"] {
    background: #161b26;
    border: 1px solid #232936; border-radius: 10px; padding: 12px 16px;
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
    background: #161b26;
    border: 1px solid #232936; border-radius: 10px;
    padding: 14px 16px;
}
.level-title { font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }
.level-item { font-size: 0.9rem; font-weight: 600; margin-bottom: 4px; }
.level-item.green { color: #26a69a; }
.level-item.red { color: #ef5350; }
.level-item.yellow { color: #fbbf24; }
.level-sub { font-size: 0.7rem; color: #6b7280; }

/* Signal filter */
.signal-box {
    background: #161b26;
    border: 1px solid #232936; border-radius: 10px;
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
    background: #161b26;
    border-radius: 10px; padding: 18px 24px; margin-bottom: 18px;
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
    color: #9aa4b2;
    font-style: italic;
    padding: 8px 0 4px 0;
    border-top: 1px solid #232936;
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


@st.cache_data(ttl=60, show_spinner=False)
def _cached_spot(tkr: str) -> float:
    return provider.fetch_spot(tkr)


@st.cache_data(ttl=180, show_spinner="Mengambil options chain...")
def _cached_chain(tkr: str):
    return provider.fetch_chain(tkr)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_candles_1m(tkr: str) -> "pd.DataFrame":
    try:
        df = yf.download(tkr, period="1d", interval="1m", progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    except Exception:
        return pd.DataFrame()


ctrl_1, ctrl_2, ctrl_3, ctrl_4, ctrl_5, ctrl_6 = st.columns([0.75, 1.1, 1.5, 0.75, 1.0, 0.65])

with ctrl_1:
    ticker = st.text_input(
        "Ticker Symbol", value="GLD", max_chars=10,
        help="e.g. SPY, QQQ, AAPL, TSLA, NVDA, GLD",
    ).upper()

spot = _cached_spot(ticker)
ul_price, ul_label, conv_ratio = _get_conversion(ticker, spot)
has_conversion = ul_price is not None

chain = _cached_chain(ticker)

if chain.empty or spot <= 0:
    st.error(
        "⚠️ Gagal mengambil data dari Yahoo Finance (options chain kosong atau "
        "spot = 0). Kemungkinan rate-limit, koneksi bermasalah, atau ticker tanpa "
        "options. Tunggu 1–2 menit lalu klik Refresh Now — JANGAN ambil keputusan "
        "trading dari tampilan ini."
    )
    if st.button("Refresh Now", key="refresh_empty"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

expiry_dates = sorted(chain["expiry"].dt.date.unique())
expiry_labels = ["All Expiries"] + [d.strftime("%Y-%m-%d") for d in expiry_dates]

with ctrl_2:
    # Default = nearest expiry that is still in the future (skip today's 0DTE:
    # its greeks are tiny/degenerate late in the session and bars look empty).
    _today_utc = pd.Timestamp.utcnow().date()
    _future_idx = [i for i, d in enumerate(expiry_dates) if d > _today_utc]
    if _future_idx:
        _default_exp_idx = _future_idx[0] + 1  # +1 karena index 0 = "All Expiries"
    else:
        _default_exp_idx = 1 if len(expiry_labels) > 1 else 0
    selected_label = st.selectbox("Expiry Date", expiry_labels, index=_default_exp_idx)
    expiry_filter = None if selected_label == "All Expiries" else selected_label

with ctrl_3:
    spot_text = f"${spot:,.2f}"
    if has_conversion:
        spot_text += f"  ({ul_label} ${ul_price:,.2f})"
    st.markdown(
        f'<div style="margin-top:24px;">'
        f'<span style="font-size:0.7rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.06em;">Spot Price</span><br>'
        f'<span style="font-size:1.25rem;font-weight:700;color:#fbbf24;font-variant-numeric:tabular-nums;">{spot_text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

with ctrl_4:
    auto_interval = st.selectbox("Auto Refresh", list(REFRESH_INTERVALS.keys()), index=2)

with ctrl_5:
    oanda_price = st.number_input(
        "Forex (XAUUSD)",
        min_value=0.0, value=0.0, step=0.1, format="%.2f",
        help="Isi harga XAUUSD dari broker kamu (Exness, IC Markets, dll)"
    )

with ctrl_6:
    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    if st.button("Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

interval_sec = REFRESH_INTERVALS[auto_interval]
if interval_sec > 0:
    try:
        # Proper self-triggering refresh (works while page is idle).
        # Cache freshness is handled by the ttl on _cached_spot/_cached_chain,
        # so we do NOT clear the whole cache here.
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=interval_sec * 1000, key="auto_refresh_timer")
        st.caption(f"Auto refresh aktif: tiap {auto_interval}")
    except ImportError:
        # Fallback (old behavior): only re-checks when something else causes
        # a rerun. Install for true idle refresh:  pip install streamlit-autorefresh
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = time.time()
        elapsed = time.time() - st.session_state.last_refresh
        if elapsed >= interval_sec:
            st.session_state.last_refresh = time.time()
            st.cache_data.clear()
            st.rerun()
        remaining = int(interval_sec - elapsed)
        mins, secs = divmod(remaining, 60)
        st.caption(
            f"Next refresh in {mins}m {secs}s — ⚠️ pasang `pip install "
            f"streamlit-autorefresh` supaya refresh jalan otomatis saat idle"
        )
        st.markdown(
            f'<script>setTimeout(function(){{window.location.reload();}}, {remaining * 1000});</script>',
            unsafe_allow_html=True,
        )

# ── Sidebar (model params — advanced) ────────────────────────────────────────

st.sidebar.markdown("---")
mode_fokus = st.sidebar.checkbox(
    "Mode Fokus",
    value=True,
    help="ON: hanya tampilkan panel utama. OFF: buka semua panel.",
)
st.sidebar.markdown("---")

st.sidebar.markdown("## Model Params")
st.sidebar.markdown("---")
r = st.sidebar.slider("Risk-Free Rate", 0.0, 0.10, 0.05, 0.005, format="%.3f")
q = st.sidebar.slider(
    "Dividend Yield", 0.0, 0.05, 0.0, 0.005, format="%.3f",
    help="GLD tidak membayar dividen — biarkan 0. Naikkan hanya untuk ticker saham/ETF berdividen (mis. SPY ~0.013).",
)
st.sidebar.markdown("---")
st.sidebar.markdown("### IV Alarm")
iv_spike_threshold = st.sidebar.slider("IV Spike Threshold (poin)", 0.5, 5.0, 1.5, 0.5)
iv_spike_window = st.sidebar.slider("Lookback Window (menit)", 15, 120, 30, 15)


# ── Compute ──────────────────────────────────────────────────────────────────

exposures = compute_exposures(chain, spot, r=r, q=q)
agg = aggregate_by_strike(exposures, expiry_filter=expiry_filter)

filtered = exposures if expiry_filter is None else exposures[
    exposures["expiry"] == pd.Timestamp(expiry_filter)
]
kpi = summarize(filtered, spot)

# ── ATM IV computation — unified interpolated + smoothed ─────────────────────

def _compute_atm_iv(chain_df, ref_spot):
    """Interpolate OI-weighted IV at exact spot from two bounding strikes."""
    if len(chain_df) == 0:
        return None, ref_spot
    grp = chain_df.groupby("strike").apply(
        lambda g: np.average(g["implied_volatility"].values,
                             weights=g["open_interest"].clip(lower=1).values)
    ).reset_index()
    grp.columns = ["strike", "iv"]
    grp = grp.sort_values("strike").reset_index(drop=True)
    below = grp[grp["strike"] <= ref_spot]
    above = grp[grp["strike"] >= ref_spot]
    if len(below) == 0:
        return float(grp["iv"].iloc[0] * 100), float(grp["strike"].iloc[0])
    if len(above) == 0:
        return float(grp["iv"].iloc[-1] * 100), float(grp["strike"].iloc[-1])
    s0, iv0 = float(below.iloc[-1]["strike"]), float(below.iloc[-1]["iv"])
    s1, iv1 = float(above.iloc[0]["strike"]), float(above.iloc[0]["iv"])
    if s0 == s1:
        return float(iv0 * 100), s0
    t = (ref_spot - s0) / (s1 - s0)
    return float((iv0 + t * (iv1 - iv0)) * 100), (s0 + s1) / 2

_iv_src = filtered.copy() if len(filtered) > 0 else chain.copy()
atm_iv_raw, _atm_strike_iv = _compute_atm_iv(_iv_src, spot)

# IV history CSV — save raw, display smoothed
_IV_FILE = Path("iv_history.csv")
_today_str = datetime.now(WIB).strftime("%Y-%m-%d")

if _IV_FILE.exists():
    try:
        _iv_today = pd.read_csv(_IV_FILE)
        _iv_today["timestamp"] = pd.to_datetime(_iv_today["timestamp"])
        if "date" in _iv_today.columns:
            _iv_today = _iv_today[_iv_today["date"] == _today_str]
        # Keep only readings recorded under the SAME expiry selection — mixing
        # expiries makes the trend jump and triggers false IV-spike alarms.
        if "expiry" in _iv_today.columns:
            _iv_today = _iv_today[_iv_today["expiry"] == selected_label]
        else:
            _iv_today = _iv_today.iloc[0:0]  # legacy file without expiry column
        _iv_today = _iv_today.sort_values("timestamp").reset_index(drop=True)
    except Exception:
        _iv_today = pd.DataFrame(columns=["timestamp", "atm_iv", "date"])
else:
    _iv_today = pd.DataFrame(columns=["timestamp", "atm_iv", "date"])

_should_append_iv = True

# Holiday guard — block IV recording on weekends / US market holidays
_now_et = datetime.now(_ET)
_today_et_str = _now_et.strftime("%Y-%m-%d")
_iv_holiday_block = (
    _now_et.weekday() >= 5  # Saturday (5) or Sunday (6)
    or _today_et_str in _US_HOLIDAYS_2026
    or (
        _today_et_str in _US_EARLY_CLOSE_2026
        and _now_et.strftime("%H:%M") >= _US_EARLY_CLOSE_2026[_today_et_str]
    )
)
if _iv_holiday_block:
    _should_append_iv = False

if len(_iv_today) > 0 and atm_iv_raw is not None:
    _last_iv_ts = pd.to_datetime(_iv_today["timestamp"].iloc[-1])
    _elapsed_iv = (datetime.now(WIB).replace(tzinfo=None) - _last_iv_ts.replace(tzinfo=None)).total_seconds() / 60
    if not _iv_holiday_block:
        _should_append_iv = _elapsed_iv >= 1.0

# Reject outlier: if ≥5 prior readings exist and new value deviates >5pt from their median.
# Exception: if 3 consecutive rejections all cluster within 2pt of each other, treat it as a
# genuine IV level shift (news event) and accept the reading.
if "_iv_reject_buf" not in st.session_state:
    st.session_state._iv_reject_buf = []

if _should_append_iv and atm_iv_raw is not None and len(_iv_today) >= 5:
    _iv_ref_median = float(_iv_today["atm_iv"].iloc[-5:].median())
    if abs(atm_iv_raw - _iv_ref_median) > 5.0:
        st.session_state._iv_reject_buf.append(atm_iv_raw)
        _rbuf = st.session_state._iv_reject_buf
        if len(_rbuf) >= 3 and (max(_rbuf[-3:]) - min(_rbuf[-3:])) < 2.0:
            _should_append_iv = True   # genuine level shift — override
        else:
            _should_append_iv = False
    else:
        st.session_state._iv_reject_buf = []   # clean reading resets counter

if atm_iv_raw is not None and _should_append_iv:
    _new_iv = pd.DataFrame([{
        "timestamp": datetime.now(WIB).strftime("%Y-%m-%d %H:%M:%S"),
        "atm_iv": round(atm_iv_raw, 4),
        "date": _today_str,
        "expiry": selected_label,
    }])
    _iv_today = pd.concat([_iv_today, _new_iv], ignore_index=True)
    try:
        _iv_today.to_csv(_IV_FILE, index=False)
    except Exception:
        pass
    st.session_state._iv_reject_buf = []   # reset after successful write

# Rolling median smoothing window=3 — same series used for headline AND chart
if len(_iv_today) > 0:
    _iv_today["atm_iv_smooth"] = _iv_today["atm_iv"].rolling(3, min_periods=1).median()
    atm_iv = float(_iv_today["atm_iv_smooth"].iloc[-1])
    _iv_prev_s = float(_iv_today["atm_iv_smooth"].iloc[-2]) if len(_iv_today) >= 2 else None
    _iv_change = round(atm_iv - _iv_prev_s, 2) if _iv_prev_s is not None else None
else:
    atm_iv = atm_iv_raw
    _iv_change = None

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


# ── Panel KEPUTUSAN ──────────────────────────────────────────────────────────

# Layer 1 — REZIM
_k_long  = kpi.gamma_flip is not None and spot > kpi.gamma_flip
_k_short = kpi.gamma_flip is not None and spot <= kpi.gamma_flip
if _k_long:
    _l1_ok, _l1_label = True,  "✅ LONG GAMMA"
    _l1_desc = "Boleh limit order di level GEX hijau — dealer stabilkan range"
    _l1_clr  = "#26a69a"
elif _k_short:
    _l1_ok, _l1_label = True,  "⚠ SHORT GAMMA"
    _l1_desc = "Hanya momentum — dilarang counter-trend entry"
    _l1_clr  = "#ef5350"
else:
    _l1_ok, _l1_label = False, "⚫ NEUTRAL"
    _l1_desc = "Flip tidak terdeteksi — data tidak cukup"
    _l1_clr  = "#fbbf24"

# Layer 2 — LOKASI
_agg_above_k = agg[agg["strike"] > spot]
_agg_below_k = agg[agg["strike"] < spot]
_wall_df_k   = _agg_above_k[_agg_above_k["gex"] > 0].nlargest(1, "gex")
_hole_df_k   = _agg_below_k[_agg_below_k["gex"] < 0].nsmallest(1, "gex")

def _kprice(strike):
    xau = strike * conv_ratio if has_conversion else strike
    return _to_oanda(xau)

_wall_xau_k = _kprice(float(_wall_df_k["strike"].iloc[0])) if len(_wall_df_k) > 0 else None
_hole_xau_k = _kprice(float(_hole_df_k["strike"].iloc[0])) if len(_hole_df_k) > 0 else None
_flip_xau_k = _kprice(kpi.gamma_flip) if kpi.gamma_flip else None
_spot_ref_k  = _kprice(spot)

# Gap spot→flip for Layer 1 display (appended after kprice helpers are ready)
if kpi.gamma_flip is not None and _flip_xau_k is not None:
    _gap_pct = abs(spot - kpi.gamma_flip) / spot * 100
    _gap_abs = abs(_spot_ref_k - _flip_xau_k)
    _gap_clr = "#26a69a" if _gap_pct > 1.0 else "#fbbf24" if _gap_pct >= 0.5 else "#ef5350"
    _l1_desc += (
        f' <span style="font-size:0.72rem;color:{_gap_clr};font-weight:600;">'
        f'(Gap ${_gap_abs:,.0f} · {_gap_pct:.1f}%)</span>'
    )

def _kfmt(v):
    return f"${v:,.0f}" if v is not None else "N/A"

# Raw GLD-denominated values for snapshot (before broker conversion)
_wall_strike_raw = float(_wall_df_k["strike"].iloc[0]) if len(_wall_df_k) > 0 else None
_wall_gex_raw    = float(_wall_df_k["gex"].iloc[0])    if len(_wall_df_k) > 0 else None
_hole_strike_raw = float(_hole_df_k["strike"].iloc[0]) if len(_hole_df_k) > 0 else None
_hole_gex_raw    = float(_hole_df_k["gex"].iloc[0])    if len(_hole_df_k) > 0 else None

# OI at the level strikes from filtered chain
def _oi_at_k(strike, opt_type):
    if strike is None or len(filtered) == 0:
        return 0
    mask = (filtered["strike"] == strike) & (filtered["option_type"].str.lower() == opt_type)
    return int(filtered[mask]["open_interest"].sum())

_wall_call_oi_k   = _oi_at_k(_wall_strike_raw, "call")
_support_put_oi_k = _oi_at_k(_hole_strike_raw, "put")

# ── GEX snapshot: load → compare → save (1 row per day, overwrite) ──────────
_GEX_HIST = Path("gex_history.csv")
_yesterday_str = (datetime.now(WIB) - timedelta(days=1)).strftime("%Y-%m-%d")

try:
    _gex_hist_df = pd.read_csv(_GEX_HIST) if _GEX_HIST.exists() else pd.DataFrame()
except Exception:
    _gex_hist_df = pd.DataFrame()

# Yesterday's row for badge comparison
_yest_row = None
if len(_gex_hist_df) > 0 and "date" in _gex_hist_df.columns:
    _yest_rows = _gex_hist_df[_gex_hist_df["date"] == _yesterday_str]
    if len(_yest_rows) > 0:
        _yest_row = _yest_rows.iloc[-1]

# Overwrite today's snapshot
_gex_snap_new = pd.DataFrame([{
    "timestamp":     datetime.now(WIB).strftime("%Y-%m-%d %H:%M:%S"),
    "date":          _today_str,
    "wall_strike":   _wall_strike_raw,
    "wall_gex":      round(_wall_gex_raw, 2)   if _wall_gex_raw   is not None else None,
    "wall_call_oi":  _wall_call_oi_k,
    "support_strike":_hole_strike_raw,
    "support_gex":   round(_hole_gex_raw, 2)   if _hole_gex_raw   is not None else None,
    "support_put_oi":_support_put_oi_k,
    "gamma_flip":    round(float(kpi.gamma_flip), 4) if kpi.gamma_flip else None,
}])
if len(_gex_hist_df) > 0 and "date" in _gex_hist_df.columns:
    _gex_hist_df = _gex_hist_df[_gex_hist_df["date"] != _today_str]
_gex_hist_df = pd.concat([_gex_hist_df, _gex_snap_new], ignore_index=True)
try:
    _gex_hist_df.to_csv(_GEX_HIST, index=False)
except Exception:
    pass

# ── Change badge per level ────────────────────────────────────────────────────
def _level_badge(curr_strike, curr_gex, yest_strike_col, yest_gex_col):
    """(badge_html, pct_float_or_None) — compares abs(GEX) vs yesterday."""
    _na = '<span style="color:#4b5563;font-size:0.67rem;">(belum ada pembanding)</span>'
    if _yest_row is None:
        return _na, None
    try:
        y_gex = float(_yest_row[yest_gex_col])
        assert not pd.isna(y_gex)
    except Exception:
        return _na, None
    if curr_strike is None or curr_gex is None:
        return '<span style="color:#4b5563;font-size:0.67rem;">(level hilang)</span>', None
    try:
        y_strike = float(_yest_row[yest_strike_col])
    except Exception:
        y_strike = None
    # Strike shifted to a different level
    if y_strike is not None and not pd.isna(y_strike) and abs(y_strike - curr_strike) > 0.01:
        return (
            f'<span style="color:#9ca3af;font-size:0.67rem;">'
            f'(bergeser dari {_kfmt(_kprice(y_strike))})</span>'
        ), None
    if abs(y_gex) < 1e-6:
        return '<span style="color:#6b7280;font-size:0.67rem;">(baru)</span>', None
    pct = (abs(curr_gex) - abs(y_gex)) / abs(y_gex) * 100
    clr = "#26a69a" if pct >= 0 else "#ef5350"
    sym = "▲" if pct >= 0 else "▼"
    sgn = "+" if pct >= 0 else ""
    return (
        f'<span style="color:{clr};font-size:0.67rem;">({sym} {sgn}{pct:.0f}%)</span>'
    ), pct

_wall_badge, _wall_pct = _level_badge(_wall_strike_raw, _wall_gex_raw, "wall_strike",    "wall_gex")
_hole_badge, _hole_pct = _level_badge(_hole_strike_raw, _hole_gex_raw, "support_strike", "support_gex")

# Flip movement badge
_flip_badge = ""
if _yest_row is not None and kpi.gamma_flip is not None:
    try:
        _y_flip = float(_yest_row["gamma_flip"])
        if not pd.isna(_y_flip) and abs(_y_flip - float(kpi.gamma_flip)) > 0.01:
            _flip_badge = (
                f' <span style="color:#9ca3af;font-size:0.67rem;">'
                f'(geser dari {_kfmt(_kprice(_y_flip))})</span>'
            )
    except Exception:
        pass

# Conditional thinning warning — informational only, does NOT affect banner
_l2_warning_html = ""
for _lv_xau, _lv_pct in [(_wall_xau_k, _wall_pct), (_hole_xau_k, _hole_pct)]:
    if _lv_xau is None or _lv_pct is None:
        continue
    if _lv_pct <= -20 and abs(_spot_ref_k - _lv_xau) / _spot_ref_k * 100 < 0.5:
        _l2_warning_html = (
            '<div style="margin-top:7px;padding:6px 10px;'
            'background:rgba(251,191,36,0.08);border-radius:6px;'
            'border-left:2px solid #fbbf24;">'
            '<span style="font-size:0.71rem;color:#fbbf24;">⚠ Level terdekat menipis — '
            'keandalan berkurang, verifikasi footprint lebih ketat</span></div>'
        )
        break

# Level string with inline badges
_l2_levels_html = (
    f'Tembok: {_kfmt(_wall_xau_k)}&nbsp;{_wall_badge}&emsp;'
    f'Support: {_kfmt(_hole_xau_k)}&nbsp;{_hole_badge}&emsp;'
    f'Flip: {_kfmt(_flip_xau_k)}{_flip_badge}'
)

_key_levels_k = [l for l in [_wall_xau_k, _hole_xau_k, _flip_xau_k] if l is not None]
_at_level_k   = any(abs(_spot_ref_k - l) <= _spot_ref_k * 0.003 for l in _key_levels_k)
if _at_level_k:
    _l2_ok, _l2_label = True,  "⚡ HARGA DI LEVEL"
    _l2_desc = f"{_l2_levels_html} — Siap konfirmasi footprint"
    _l2_clr  = "#fbbf24"
else:
    _l2_ok, _l2_label = False, "NO TRADE ZONE"
    _l2_desc = f"{_l2_levels_html} — Harga di tengah, tunggu mendekati level"
    _l2_clr  = "#6b7280"

# Layer 3 — DRIFT (IV 30 menit + VEX + CEX)
_iv_dir_k, _iv_pts_k = "flat", 0.0
if len(_iv_today) >= 2 and "atm_iv_smooth" in _iv_today.columns:
    _ts_k    = pd.to_datetime(_iv_today["timestamp"])
    _now_k   = datetime.now(WIB).replace(tzinfo=None)
    _old_k   = _iv_today[(_ts_k <= (_now_k - timedelta(minutes=30))).values]
    if len(_old_k) > 0:
        _iv_pts_k = float(_iv_today["atm_iv_smooth"].iloc[-1]) - float(_old_k["atm_iv_smooth"].iloc[-1])
        if abs(_iv_pts_k) > 0.3:
            _iv_dir_k = "naik" if _iv_pts_k > 0 else "turun"

_vex_bull_k = kpi.total_vex > 0
_cex_bull_k = kpi.total_cex > 0
_iv_bull_k  = (_iv_dir_k == "turun")
_iv_bear_k  = (_iv_dir_k == "naik")

if _iv_dir_k == "flat":
    _n_b2 = sum([_vex_bull_k, _cex_bull_k])
    if _n_b2 == 2:
        _l3_ok, _l3_label = True,  "Bias BUY lemah"
        _l3_desc = f"VEX & CEX bullish — IV flat ({_iv_pts_k:+.2f}pt, <0.3pt)"
        _l3_clr  = "#26a69a"
    elif _n_b2 == 0:
        _l3_ok, _l3_label = True,  "Bias SELL lemah"
        _l3_desc = f"VEX & CEX bearish — IV flat ({_iv_pts_k:+.2f}pt, <0.3pt)"
        _l3_clr  = "#ef5350"
    else:
        _l3_ok, _l3_label = False, "Netral"
        _l3_desc = "Sinyal campur — IV flat, VEX/CEX berlawanan"
        _l3_clr  = "#6b7280"
else:
    _nb3 = sum([_iv_bull_k, _vex_bull_k, _cex_bull_k])
    _ns3 = sum([_iv_bear_k, not _vex_bull_k, not _cex_bull_k])
    if _nb3 == 3:
        _l3_ok, _l3_label = True,  "Bias BUY kuat"
        _l3_desc = f"IV turun {abs(_iv_pts_k):.1f}pt + VEX bull + CEX bull — 3/3 searah"
        _l3_clr  = "#26a69a"
    elif _nb3 == 2:
        _l3_ok, _l3_label = True,  "Bias BUY lemah"
        _l3_desc = f"2/3 sinyal bullish — IV {_iv_dir_k} {abs(_iv_pts_k):.1f}pt"
        _l3_clr  = "#26a69a"
    elif _ns3 == 3:
        _l3_ok, _l3_label = True,  "Bias SELL kuat"
        _l3_desc = f"IV naik {abs(_iv_pts_k):.1f}pt + VEX bear + CEX bear — 3/3 searah"
        _l3_clr  = "#ef5350"
    elif _ns3 == 2:
        _l3_ok, _l3_label = True,  "Bias SELL lemah"
        _l3_desc = f"2/3 sinyal bearish — IV {_iv_dir_k} {abs(_iv_pts_k):.1f}pt"
        _l3_clr  = "#ef5350"
    else:
        _l3_ok, _l3_label = False, "Netral"
        _l3_desc = "Sinyal campur — tunggu konfirmasi lebih lanjut"
        _l3_clr  = "#6b7280"

# Layer 4 — TRIGGER (dynamic text + proxy candle indicators)

# Dynamic instruction based on regime
if _k_long:
    _l4_label = "Cari absorption/pembalikan DI level"
elif _k_short:
    _l4_label = "Cari delta kelanjutan DI penembusan"
else:
    _l4_label = "Tunggu footprint di level kunci"
_l4_desc = "Engulfing/delta spike/absorpsi volume di level GEX sebelum entry"

# Proxy indicators from 1-min candles
_c1m = _cached_candles_1m(ticker)
_engulf_found = None
_vol_ratio_k  = None
_vol_spike_k  = False

if len(_c1m) >= 3:
    for _ci, _pi in [(-1, -2), (-2, -3)]:
        _co = float(_c1m["Open"].iloc[_ci]);  _cc = float(_c1m["Close"].iloc[_ci])
        _po = float(_c1m["Open"].iloc[_pi]);  _pc = float(_c1m["Close"].iloc[_pi])
        _c_bull = _cc > _co;  _p_bull = _pc > _po
        if _c_bull and not _p_bull and _co <= _pc and _cc >= _po:
            _engulf_found = "Bullish"; break
        if not _c_bull and _p_bull and _co >= _pc and _cc <= _po:
            _engulf_found = "Bearish"; break

if len(_c1m) >= 20:
    _vol_last_k = float(_c1m["Volume"].iloc[-1])
    _vol_avg_k  = float(_c1m["Volume"].iloc[-20:].mean())
    if _vol_avg_k > 0:
        _vol_ratio_k = _vol_last_k / _vol_avg_k
        _vol_spike_k = _vol_ratio_k >= 2.0

# Build proxy HTML block
_engulf_clr = "#26a69a" if _engulf_found == "Bullish" else "#ef5350" if _engulf_found == "Bearish" else "#6b7280"
_engulf_txt = f"{_engulf_found} engulfing ✓" if _engulf_found else "Tidak ada engulfing (3 candle)"
_vol_clr  = "#fbbf24" if _vol_spike_k else "#9ca3af"
_vol_txt  = (
    f"{_vol_ratio_k:.1f}x ← SPIKE!" if _vol_spike_k
    else (f"{_vol_ratio_k:.1f}x rata-rata 20" if _vol_ratio_k is not None else "data tidak cukup")
)
_proxy_extra = (
    '<div style="margin-top:7px;padding:6px 9px;background:rgba(255,255,255,0.03);'
    'border-radius:6px;border-left:2px solid #374151;">'
    '<div style="font-size:0.6rem;font-weight:700;color:#4b5563;text-transform:uppercase;'
    'letter-spacing:0.08em;margin-bottom:5px;">'
    'PROXY (bukan footprint) — verifikasi tetap di chart footprint</div>'
    f'<div style="display:flex;gap:18px;flex-wrap:wrap;">'
    f'<span style="font-size:0.72rem;color:{_engulf_clr};">📊 Engulfing: {_engulf_txt}</span>'
    f'<span style="font-size:0.72rem;color:{_vol_clr};">📦 Volume: {_vol_txt}</span>'
    f'</div></div>'
)

# Layer 5 — static
_l5_label = "Kelola risiko"
_l5_desc  = "SL di luar level GEX terdekat  ·  Maks 1–2% modal  ·  Jangan averaging rugi"

# Banner (layer 4 & 5 tidak ikut menentukan)
_setup_ok_k = _l1_ok and _l2_ok and _l3_ok
if _setup_ok_k:
    _banner_k = (
        '<div style="background:rgba(38,166,154,0.07);border:1px solid #26a69a;border-radius:10px;'
        'padding:11px 18px;margin-bottom:14px;text-align:center;">'
        '<span style="font-size:0.95rem;font-weight:700;color:#26a69a;letter-spacing:0.04em;">'
        '✅ SETUP LENGKAP — Lanjut ke konfirmasi entry</span></div>'
    )
else:
    _banner_k = (
        '<div style="background:rgba(239,83,80,0.06);border:1px solid #ef5350;border-radius:10px;'
        'padding:11px 18px;margin-bottom:14px;text-align:center;">'
        '<span style="font-size:0.95rem;font-weight:700;color:#ef5350;letter-spacing:0.04em;">'
        '⛔ SETUP BELUM LENGKAP — JANGAN ENTRY</span></div>'
    )


def _krow(n, name, status, desc, color, ok, icon=None, extra=""):
    dot_bg = f"{color}22"
    if icon is not None:
        chk_sym, chk_clr = icon, "#9ca3af"
    else:
        chk_sym = "✓" if ok else "—"
        chk_clr = color if ok else "#4b5563"
    return (
        f'<div style="display:flex;align-items:flex-start;gap:12px;padding:10px 0;'
        f'border-bottom:1px solid rgba(255,255,255,0.05);">'
        f'<div style="flex:0 0 22px;height:22px;border-radius:50%;background:{dot_bg};'
        f'display:flex;align-items:center;justify-content:center;font-size:0.7rem;'
        f'font-weight:700;color:{color};margin-top:1px;flex-shrink:0;">{n}</div>'
        f'<div style="flex:0 0 92px;flex-shrink:0;">'
        f'<div style="font-size:0.62rem;font-weight:700;color:#6b7280;text-transform:uppercase;'
        f'letter-spacing:0.1em;margin-bottom:3px;">{name}</div>'
        f'<div style="font-size:0.82rem;font-weight:700;color:{color};">{status}</div>'
        f'</div>'
        f'<div style="flex:1;font-size:0.74rem;color:#9ca3af;line-height:1.5;padding-top:3px;">'
        f'{desc}{extra}</div>'
        f'<div style="flex:0 0 18px;font-size:1rem;font-weight:700;color:{chk_clr};'
        f'text-align:right;padding-top:3px;flex-shrink:0;">{chk_sym}</div>'
        f'</div>'
    )


st.markdown(
    '<div class="card" style="margin-bottom:14px;">'
    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">'
    '<div class="card-title" style="margin:0;">🎯 Keputusan Trading</div>'
    '<div style="font-size:0.72rem;color:#6b7280;">5-lapis · auto dari data realtime</div>'
    '</div>'
    + _banner_k
    + _krow(1, "REZIM",   _l1_label, _l1_desc, _l1_clr, _l1_ok)
    + _krow(2, "LOKASI",  _l2_label, _l2_desc, _l2_clr, _l2_ok, extra=_l2_warning_html)
    + _krow(3, "DRIFT",   _l3_label, _l3_desc, _l3_clr, _l3_ok)
    + _krow(4, "TRIGGER", _l4_label, _l4_desc, "#6b7280", True, icon="👁", extra=_proxy_extra)
    + _krow(5, "RISIKO",  _l5_label, _l5_desc, "#6b7280", True)
    + '</div>',
    unsafe_allow_html=True,
)


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
# regime badge rendered inside Legacy expander below

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

with st.expander("🎯 Level Kunci Detail", expanded=not mode_fokus):
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

with st.expander("🧪 Sinyal Legacy (referensi)", expanded=not mode_fokus):
    st.caption("Panel lama — keputusan resmi ada di panel Keputusan 5 lapis")
    st.markdown(
        (
            f'<div class="regime-badge {regime_class}">'
            f'<div>'
            f'<div class="regime-title" style="color:{regime_color};">{regime_emoji} {regime_text}</div>'
            f'<div class="regime-sub">Spot ${spot:,.2f} vs Flip ${kpi.gamma_flip:,.2f}</div>'
            f'</div>'
            f'<div style="text-align:right;">'
            f'<div style="font-size:0.8rem;color:#9ca3af;">Daily Bias</div>'
            f'<div style="font-size:1rem;font-weight:700;color:{cex_bias_color};">{cex_bias} (CEX {_fmt_short(kpi.total_cex)})</div>'
            f'</div>'
            f'</div>'
        ) if kpi.gamma_flip else (
            f'<div class="regime-badge {regime_class}">'
            f'<div class="regime-title" style="color:{regime_color};">{regime_emoji} {regime_text}</div>'
            f'</div>'
        ),
        unsafe_allow_html=True,
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
    font=dict(family="Inter, sans-serif", size=11, color="#9aa4b2"),
    margin=dict(l=48, r=12, t=8, b=44),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(size=10, color="#9aa4b2"), bgcolor="rgba(0,0,0,0)",
    ),
    height=400,
    dragmode="pan",
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(gridcolor="#1c2230", zeroline=True, zerolinecolor="#2a3040"),
)


def _spot_marker(fig, spot, gamma_flip):
    spot_label = f"${spot:,.0f}"
    if has_conversion:
        spot_label += f" ({ul_label} ${spot * conv_ratio:,.0f})"
    fig.add_vline(
        x=spot, line_dash="dash", line_color="rgba(251,191,36,0.65)", line_width=1,
        annotation_text=f"Spot {spot_label}",
        annotation_font=dict(color="rgba(251,191,36,0.8)", size=10, family="Inter"),
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
            x=gamma_flip, line_dash="dot", line_color="rgba(38,166,154,0.8)", line_width=1,
            annotation_text=f"FLIP {flip_label} | Gap ${gap_ul:,.0f}",
            annotation_font=dict(color="rgba(38,166,154,0.9)", size=10, family="Inter"),
            annotation_bgcolor="rgba(38,166,154,0.1)",
            annotation_bordercolor="rgba(38,166,154,0.35)",
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


def _card_start(title, emoji, subtitle, legend_green=None, legend_red=None, hint="Hover untuk detail"):
    _leg = (
        f'<div class="card-legend">'
        f'<span class="legend-green">{legend_green}</span>'
        f'<span class="legend-red">{legend_red}</span></div>'
    ) if (legend_green or legend_red) else ""
    return (
        f'<div class="card">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        f'<div><div class="card-title">{title}</div>'
        f'<div class="card-subtitle">{subtitle}</div></div>'
        f'<div style="font-size:0.75rem;color:#4b5563;">{hint}</div></div>'
        + _leg
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

if _k_long:
    _gex_cl = (
        '<span style="color:#26a69a;font-weight:700;">LONG GAMMA</span> — '
        '<span style="color:#26a69a;">■</span> Hijau: resistance/tembok (dealer lawan arah)&emsp;'
        '<span style="color:#ef5350;">■</span> Merah: support (acuan)')
elif _k_short:
    _gex_cl = (
        '<span style="color:#ef5350;font-weight:700;">SHORT GAMMA</span> — '
        '<span style="color:#26a69a;">■</span> Hijau: lantai/tembok yang dibela&emsp;'
        '<span style="color:#ef5350;">■</span> Merah: lubang/celah (percepatan, BUKAN level)')
else:
    _gex_cl = (
        '<span style="color:#26a69a;">■</span> Hijau: GEX positif&emsp;'
        '<span style="color:#ef5350;">■</span> Merah: GEX negatif')

st.markdown(
    _card_start(
        "Dealer Gamma Exposure (GEX)", "\U0001f525",
        "Net gamma pressure per strike — positive = resistance, negative = support",
    )
    + f'<div style="font-size:0.75rem;color:#9ca3af;margin-bottom:8px;">{_gex_cl}</div>',
    unsafe_allow_html=True,
)
with st.expander("ℹ️ Lihat semua kombinasi"):
    st.markdown(
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 20px;font-size:0.73rem;">'
        + '<div style="color:#26a69a;font-weight:700;">LONG GAMMA</div>'
        + '<div style="color:#ef5350;font-weight:700;">SHORT GAMMA</div>'
        + '<div style="color:#9ca3af;font-weight:600;">Di bawah SPOT</div>'
        + '<div style="color:#9ca3af;font-weight:600;">Di bawah SPOT</div>'
        + '<div><span style="color:#26a69a;">■</span> <span style="color:#9ca3af;">Hijau = Resistance</span></div>'
        + '<div><span style="color:#26a69a;">■</span> <span style="color:#9ca3af;">Hijau = Lantai (nahan jatuh)</span></div>'
        + '<div><span style="color:#ef5350;">■</span> <span style="color:#9ca3af;">Merah = Support</span></div>'
        + '<div><span style="color:#ef5350;">■</span> <span style="color:#9ca3af;">Merah = Lubang (jatuh cepat)</span></div>'
        + '<div style="color:#9ca3af;font-weight:600;margin-top:4px;">Di atas SPOT</div>'
        + '<div style="color:#9ca3af;font-weight:600;margin-top:4px;">Di atas SPOT</div>'
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
fig_gex.update_layout(xaxis_title="Strike Price", yaxis_title="Gamma Exposure ($)", yaxis_range=[-_gex_cap, _gex_cap], yaxis_fixedrange=True, **_LAYOUT)
st.plotly_chart(fig_gex, key="gex", config={"displayModeBar": "hover", "displaylogo": False, "scrollZoom": False, "editable": False, "edits": {"axisTitleText": False, "titleText": False}, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
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

with st.expander("📊 VEX & CEX", expanded=not mode_fokus):
    col_v, col_c = st.columns(2)

    with col_v:
        st.markdown(
            _card_start(
                "Vanna Exposure (VEX)", "\U0001f300",
                "Delta sensitivity to implied volatility changes",
            )
            + '<div style="font-size:0.75rem;color:#9ca3af;margin-bottom:8px;line-height:1.7;">'
            + '<span style="color:#26a69a;">■</span> Hijau = magnet (harga tertarik ke strike itu)&emsp;'
            + '<span style="color:#ef5350;">■</span> Merah = penolak (harga sulit lewat)<br>'
            + '<span style="color:#6b7280;font-size:0.7rem;">Arah efeknya mengikuti sisi spot &amp; IV — VEX bekerja saat IV berubah; IV flat = efek vanna tidur</span>'
            + '</div>',
            unsafe_allow_html=True,
        )
        with st.expander("ℹ️ Detail"):
            st.markdown(
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 20px;font-size:0.73rem;">'
                + '<div style="color:#9ca3af;font-weight:600;">Kiri Spot</div>'
                + '<div style="color:#9ca3af;font-weight:600;">Kanan Spot</div>'
                + '<div><span style="color:#26a69a;">■</span> <span style="color:#9ca3af;">Hijau = Magnet turun</span></div>'
                + '<div><span style="color:#26a69a;">■</span> <span style="color:#9ca3af;">Hijau = Magnet naik</span></div>'
                + '<div><span style="color:#ef5350;">■</span> <span style="color:#9ca3af;">Merah = Tolak (lantai)</span></div>'
                + '<div><span style="color:#ef5350;">■</span> <span style="color:#9ca3af;">Merah = Tolak (plafon)</span></div>'
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
        fig_vex.update_layout(xaxis_title="Strike Price", yaxis_title="Vanna Exposure ($)", yaxis_range=[-_vex_cap, _vex_cap], yaxis_fixedrange=True, **_LAYOUT)
        st.plotly_chart(fig_vex, key="vex", config={"displayModeBar": "hover", "displaylogo": False, "scrollZoom": False, "editable": False, "edits": {"axisTitleText": False, "titleText": False}, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
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
            )
            + '<div style="font-size:0.75rem;color:#9ca3af;margin-bottom:8px;line-height:1.7;">'
            + '<span style="color:#26a69a;">■</span> Hijau = tekanan beli harian dealer&emsp;'
            + '<span style="color:#ef5350;">■</span> Merah = tekanan jual harian<br>'
            + '<span style="color:#6b7280;font-size:0.7rem;">Efek menguat mendekati expiry — CEX bekerja tiap hari tanpa perlu pergerakan harga</span>'
            + '</div>',
            unsafe_allow_html=True,
        )
        with st.expander("ℹ️ Detail"):
            st.markdown(
                '<div style="font-size:0.73rem;color:#9ca3af;line-height:1.8;">'
                + '<span style="color:#26a69a;font-weight:600;">Positif (Hijau):</span> '
                + 'Delta dealer naik seiring waktu → dealer beli underlying → bullish drift harian<br>'
                + '<span style="color:#ef5350;font-weight:600;">Negatif (Merah):</span> '
                + 'Delta dealer turun → dealer jual → bearish drift harian<br>'
                + '<span style="color:#6b7280;">Besar CEX bergantung jarak ke expiry — OTM dekat expiry paling kuat</span>'
                + '</div>',
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
        fig_cex.update_layout(xaxis_title="Strike Price", yaxis_title="Charm Exposure ($)", yaxis_range=[-_cex_cap, _cex_cap], yaxis_fixedrange=True, **_LAYOUT)
        st.plotly_chart(fig_cex, key="cex", config={"displayModeBar": "hover", "displaylogo": False, "scrollZoom": False, "editable": False, "edits": {"axisTitleText": False, "titleText": False}, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
        if kpi.total_cex < -1e6:
            _interp(f"Selling pressure harian kuat ({_fmt_short(kpi.total_cex)}) — bearish drift, hati-hati hold long")
        elif kpi.total_cex > 1e6:
            _interp(f"Buying pressure harian kuat ({_fmt_short(kpi.total_cex)}) — bullish drift")
        else:
            _interp("Charm netral — tidak ada tekanan harian yang signifikan")
        st.markdown(_CARD_END, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  1b. IMPLIED VOLATILITY (IV)
# ══════════════════════════════════════════════════════════════════════════════

# IV spike alarm banner
if _iv_change is not None and len(_iv_today) >= 2:
    _now_naive = datetime.now(WIB).replace(tzinfo=None)
    _cutoff = _now_naive - timedelta(minutes=iv_spike_window)
    _iv_today_ts = _iv_today.copy()
    _iv_today_ts["timestamp"] = pd.to_datetime(_iv_today_ts["timestamp"]).dt.tz_localize(None)
    _iv_window = _iv_today_ts[_iv_today_ts["timestamp"] >= pd.Timestamp(_cutoff)]
    if len(_iv_window) >= 2:
        _iv_spike_val = float(_iv_today_ts["atm_iv"].iloc[-1]) - float(_iv_window["atm_iv"].iloc[0])
        if _iv_spike_val >= iv_spike_threshold:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,rgba(239,83,80,0.2),rgba(239,83,80,0.08));'
                f'border:2px solid #ef5350;border-radius:12px;padding:14px 20px;margin-bottom:14px;">'
                f'<span style="font-size:1rem;font-weight:700;color:#ef5350;">⚠️ IV Spike Terdeteksi</span>'
                f'<span style="font-size:0.85rem;color:#d1d5db;margin-left:12px;">'
                f'IV naik <b>+{_iv_spike_val:.2f} poin</b> dalam {iv_spike_window} menit — kemungkinan menjelang news. '
                f'Pertimbangkan kecilkan posisi atau flat.</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

st.markdown(
    _card_start(
        "Implied Volatility (IV)", "📊",
        "ATM IV intraday trend + IV skew per strike",
        "IV Turun — Pasar tenang (hijau)",
        "IV Naik — Tekanan meningkat (merah)",
    ),
    unsafe_allow_html=True,
)

if _iv_holiday_block:
    st.caption("Pasar AS tutup — history IV dijeda")

# ATM IV big number + badge
if atm_iv is not None:
    _iv_badge_color = "#ef5350" if (_iv_change and _iv_change > 0) else "#26a69a" if (_iv_change and _iv_change < 0) else "#6b7280"
    _iv_arrow = "↑" if (_iv_change and _iv_change > 0) else "↓" if (_iv_change and _iv_change < 0) else "→"
    _iv_badge_txt = f"{_iv_arrow} {abs(_iv_change):.2f} pts" if _iv_change is not None else "Data pertama"
    _atm_lbl = f"${_atm_strike_iv * conv_ratio:,.0f} ({ul_label})" if has_conversion else f"${_atm_strike_iv:,.0f}"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:16px;margin-bottom:14px;">'
        f'<div><div style="font-size:0.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;">ATM IV — Strike {_atm_lbl}</div>'
        f'<div style="font-size:2rem;font-weight:700;color:#fbbf24;line-height:1.2;font-variant-numeric:tabular-nums;">{atm_iv:.1f}%</div></div>'
        f'<div style="background:{_iv_badge_color};color:#fff;border-radius:8px;padding:6px 14px;font-size:0.9rem;font-weight:700;">'
        f'{_iv_badge_txt}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

col_iv1, col_iv2 = st.columns(2)

with col_iv1:
    if len(_iv_today) >= 2 and "atm_iv_smooth" in _iv_today.columns:
        _iv_plot = _iv_today.copy()
        _iv_plot["timestamp"] = pd.to_datetime(_iv_plot["timestamp"])
        fig_iv_line = go.Figure(go.Scatter(
            x=_iv_plot["timestamp"], y=_iv_plot["atm_iv_smooth"],
            mode="lines+markers",
            line=dict(color="#fbbf24", width=2),
            marker=dict(size=5, color="#fbbf24"),
            hovertemplate="<b>%{x|%H:%M}</b><br>IV: %{y:.2f}%<extra></extra>",
        ))
        fig_iv_line.update_layout(
            xaxis_title="Waktu (WIB)", yaxis_title="ATM IV (%)",
            showlegend=False, **_LAYOUT,
        )
        st.plotly_chart(fig_iv_line, key="iv_line", config={"displayModeBar": "hover", "displaylogo": False, "scrollZoom": False, "editable": False, "edits": {"axisTitleText": False, "titleText": False}, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    else:
        st.markdown(
            '<div style="color:#6b7280;font-size:0.85rem;padding:30px 0;text-align:center;">'
            '📡 Mengumpulkan data intraday...<br>Refresh beberapa kali untuk melihat trend IV.</div>',
            unsafe_allow_html=True,
        )

with col_iv2:
    _skew_src = filtered.copy() if len(filtered) > 0 else chain.copy()
    if len(_skew_src) > 0:
        # Filter to ±12% from spot, minimum OI > 0 to remove illiquid wings
        _skew_lo, _skew_hi = spot * 0.88, spot * 1.12
        _skew_src = _skew_src[
            (_skew_src["strike"] >= _skew_lo) &
            (_skew_src["strike"] <= _skew_hi) &
            (_skew_src["open_interest"] > 0)
        ]
        _skew = _skew_src.groupby("strike", as_index=False)["implied_volatility"].mean()
        _skew["iv_pct"] = _skew["implied_volatility"] * 100
        # Remove isolated spikes: exclude if > 5 poin dari rolling median tetangga
        _skew_excluded = 0
        if len(_skew) >= 3:
            _skew["_lmed"] = _skew["iv_pct"].rolling(3, center=True, min_periods=1).median()
            _mask = abs(_skew["iv_pct"] - _skew["_lmed"]) <= 5.0
            _skew_excluded = int((~_mask).sum())
            _skew = _skew[_mask].reset_index(drop=True)
    if len(_skew_src) > 0 and len(_skew) > 1:
        fig_skew = go.Figure(go.Scatter(
            x=_skew["strike"], y=_skew["iv_pct"],
            mode="lines+markers",
            line=dict(color="#a78bfa", width=2),
            marker=dict(size=4, color="#a78bfa"),
            hovertemplate="<b>Strike $%{x:,.0f}</b><br>IV: %{y:.1f}%<extra></extra>",
        ))
        _spot_marker(fig_skew, spot, kpi.gamma_flip)
        _skew_med = float(_skew["iv_pct"].median())
        fig_skew.update_layout(
            xaxis_title="Strike (±12% dari spot)", yaxis_title="Implied Volatility (%)",
            yaxis_range=[max(0, _skew_med * 0.7), _skew_med * 1.5],
            showlegend=False, **_LAYOUT,
        )
        st.plotly_chart(fig_skew, key="iv_skew", config={"displayModeBar": "hover", "displaylogo": False, "scrollZoom": False, "editable": False, "edits": {"axisTitleText": False, "titleText": False}, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
        if _skew_excluded > 0:
            st.markdown(
                f'<div style="font-size:0.72rem;color:#4b5563;margin-top:-8px;">'
                f'⚠ {_skew_excluded} strike di-exclude (spike IV >5 poin dari median tetangga)</div>',
                unsafe_allow_html=True,
            )

# Auto interpretation
_iv_dir = "naik" if (_iv_change and _iv_change > 0) else "turun" if (_iv_change and _iv_change < 0) else None
_vex_bull = kpi.total_vex > 0

if _iv_dir == "turun" and _vex_bull and _is_long_gamma:
    _iv_txt = "IV melandai + VEX positif + Long Gamma — dealer terdorong beli saat volatilitas turun. Drift naik tenang (bullish)"
elif _iv_dir == "naik" and _is_short_gamma:
    _iv_txt = "IV naik + Short Gamma — tekanan turun dipercepat dealer. Risiko whipsaw tinggi, kecilkan ukuran posisi"
elif _iv_dir == "naik" and _is_long_gamma:
    _iv_txt = "IV naik + Long Gamma — pasar antisipasi gerakan besar. Level masih dibela dealer tapi waspada perubahan rezim"
elif _iv_dir == "turun" and not _vex_bull:
    _iv_txt = "IV turun + VEX negatif — tekanan jual mereda tapi VEX masih bearish. Hati-hati false recovery"
elif _iv_dir is None:
    _iv_txt = "Mengumpulkan data intraday untuk analisis trend IV..."
else:
    _iv_txt = f"IV {_iv_dir} — {'Long Gamma, pasar terkendali' if _is_long_gamma else 'Short Gamma, waspadai momentum kuat'}"
_interp(_iv_txt)
st.markdown(_CARD_END, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  2. ABSOLUTE GAMMA — OPEN INTEREST
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("🔥 Heatmap & Absolute Gamma", expanded=not mode_fokus):
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
        yaxis_title="Open Interest (Contracts)", yaxis_range=[-_oi_cap, _oi_cap], yaxis_fixedrange=True, **_LAYOUT,
    )
    st.plotly_chart(fig_abs, key="abs_gex", config={"displayModeBar": "hover", "displaylogo": False, "scrollZoom": False, "editable": False, "edits": {"axisTitleText": False, "titleText": False}, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
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

    # ══ HEDGING PRESSURE HEATMAP (normalized -1 to +1) ══════════════════════

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
        yaxis_range=[-1.1, 1.1], yaxis_fixedrange=True, **_LAYOUT,
    )
    st.plotly_chart(fig_hm, key="heatmap", config={"displayModeBar": "hover", "displaylogo": False, "scrollZoom": False, "editable": False, "edits": {"axisTitleText": False, "titleText": False}, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    max_strike = agg.loc[agg["gex"].idxmax(), "strike"] if len(agg) > 0 else 0
    _interp(f"Tekanan terkuat di strike ${max_strike:,.0f}" + (f" ({ul_label} ${max_strike * conv_ratio:,.0f})" if has_conversion else "") + " — perhatikan level ini")
    st.markdown(_CARD_END, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  5. NET DELTA EXPOSURE
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("📈 Net Delta & OI", expanded=not mode_fokus):
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
    fig_delta.update_layout(xaxis_title="Strike Price", yaxis_title="Delta Exposure ($)", yaxis_range=[-_delta_cap, _delta_cap], yaxis_fixedrange=True, **_LAYOUT)
    st.plotly_chart(fig_delta, key="delta", config={"displayModeBar": "hover", "displaylogo": False, "scrollZoom": False, "editable": False, "edits": {"axisTitleText": False, "titleText": False}, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    if kpi.net_delta > 0:
        _interp(f"Dealer long delta ({_fmt_short(kpi.net_delta)}) — bullish hedge positioning, ada support dari dealer")
    else:
        _interp(f"Dealer short delta ({_fmt_short(kpi.net_delta)}) — bearish positioning, dealer ikut tekan harga turun")
    st.markdown(_CARD_END, unsafe_allow_html=True)

# ── Raw data ─────────────────────────────────────────────────────────────────

with st.expander("📓 Jurnal — Data Mentah"):
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

