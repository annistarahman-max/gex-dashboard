# CLAUDE.md — Options Market Maker Exposure Dashboard

Project context for future Claude sessions. Read this before touching any code.

## What this is

Streamlit dashboard (`app.py`) for analysing options market-maker (MM) exposure on
equity/ETF tickers. Primary use-case: **GLD** (SPDR Gold ETF) with live XAUUSD
conversion displayed alongside. Also works for SPY, QQQ, AAPL, etc.

**Data source**: Yahoo Finance via `yfinance`. No paid feeds.  
**Deploy target**: Streamlit Cloud (also runs locally via `run.bat`).  
**Language**: Python 3.11 (pinned in `runtime.txt`).

---

## File map

```
app.py                  — entire dashboard (one large script, ~1200 lines)
src/data/data_provider.py — YFinanceProvider / MockDataProvider
src/engine/exposure.py  — GEX / VEX / CEX / net-delta maths
src/engine/greeks.py    — Black-Scholes greeks
.streamlit/config.toml  — theme (dark, #0a0a0f bg) + headless=true
requirements.txt        — pinned versions; tzdata required for ZoneInfo on Cloud
run.bat                 — local launcher: `start /b streamlit run app.py` → 3s delay → browser
```

`iv_history.csv` and `gex_history.csv` are **gitignored** — they are ephemeral
local files written at runtime, not source-controlled.

---

## Architecture: data flow

```
yfinance (Yahoo Finance)
    ↓
_cached_spot(tkr)   ttl=60s   → None on any exception (cached, prevents hammering)
_cached_chain(tkr)  ttl=300s  → pd.DataFrame() on any exception (cached)
_fetch_underlying_price()  ttl=300s  → None on exception
    ↓
Rate-limit check (lines ~357–378 in app.py)
    — if spot is None OR chain empty: show purple banner ⏳, offer Coba Lagi, st.stop()
    — stale spot saved in session_state["_stale_spot"] + "_stale_spot_ts"
    — if stale spot exists while chain fails: banner shown but execution continues
      (spot shown from stale; if chain empty st.stop() anyway)
    ↓
Chain sanity guard — _chain_sanity(chain, spot) (lines ~380–398)
    — (a) ATM strike within ±5% of spot
    — (b) ≥10 strikes in ±12% band
    — (c) ATM IV < 100%
    — FAIL → yellow banner ⚠️ + st.stop(); iv_history NOT written (never reached)
    ↓
src/engine/* computes GEX / VEX / CEX / net-delta / ATM IV
    ↓
Charts (8 Plotly figures) + Keputusan 5 lapis panel
```

**Why failures are cached**: if `_cached_spot` raises, Streamlit won't cache the
exception, so every rerun would immediately retry Yahoo. By catching inside the
`@st.cache_data` function and returning None/empty, the failure result IS cached for
the TTL, preventing hammering while rate-limited. "Coba Lagi" button calls
`st.cache_data.clear()` then `st.rerun()` to force a fresh attempt.

---

## Key constants (top of app.py)

```python
WIB = timezone(timedelta(hours=7))          # display timezone
_ET = ZoneInfo("America/New_York")          # for IV history holiday guard
_US_HOLIDAYS_2026 = { ... }                 # NYSE full-holiday dates
_US_EARLY_CLOSE_2026 = { ... }              # NYSE 13:00 ET early-close dates

CONVERT_MAP = {
    "GLD": {"underlying": "GC=F", "label": "XAUUSD"},
    "SLV": {"underlying": "SI=F", "label": "XAGUSD"},
    "USO": {"underlying": "CL=F", "label": "WTI"},
}
```

`_to_oanda()` applies a broker-specific price offset for display; `_kprice()` applies
both `conv_ratio` and `_to_oanda()` to convert GLD strikes → XAUUSD display prices.

---

## IV history guard

`iv_history.csv` must not receive readings on weekends or NYSE holidays (Yahoo
returns stale/zero IV on closed days, contaminating the intraday trend chart).

Guard runs before every append:
- Weekend (`.weekday() >= 5`) → `_should_append_iv = False`
- Full holiday in `_US_HOLIDAYS_2026` → blocked
- Past early-close time in `_US_EARLY_CLOSE_2026` → blocked
- Minimum 1-minute gap between readings enforced

Caption "Pasar AS tutup — history IV dijeda" shown in IV panel when blocked.

**Note**: `iv_history.csv` was deleted once (mid-2026) after contaminated data was
written. If the chart shows wild spikes, delete the file — it regenerates fresh.

---

## Plotly chart config (ALL 8 charts)

```python
config = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "scrollZoom": False,
    "editable": False,
    "edits": {"axisTitleText": False, "titleText": False},
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}
```

All axes are **fully free** (no `fixedrange`). `dragmode="pan"` is set in `_LAYOUT`
so default touch interaction is pan, not zoom. `scrollZoom: False` prevents pinch-
zoom accidents on mobile.

`_LAYOUT` dict is shared across all 8 figures via `**_LAYOUT`.

---

## Control bar layout

6 top-level columns: `[0.75, 1.1, 1.5, 0.75, 1.0, 0.65]`

| col | widget |
|-----|--------|
| ctrl_1 | Ticker text input |
| ctrl_2 | Expiry date selectbox |
| ctrl_3 | Spot price display |
| ctrl_4 | Auto Refresh selectbox (default index=0 = "Off") |
| ctrl_5 | Forex (XAUUSD) number_input for Oanda broker offset |
| ctrl_6 | "Refresh Now" button (margin-top:28px to align) |

Auto Refresh default is **Off** — prevents unintended Yahoo hammering on Cloud.

---

## Panel structure (top → bottom)

1. Control bar
2. Spot price header + timestamp
3. **Keputusan 5 lapis** (main decision panel) — the ONE element allowed to be bold
4. KPI cards row
5. GEX chart
6. Absolute Gamma OI chart
7. VEX | CEX side-by-side
8. Net Delta chart
9. ATM IV intraday trend + IV Skew (in IV card)
10. `🧪 Sinyal Legacy` expander (contains old regime badge LONG/SHORT GAMMA +
    signal-box summary; NOT shown on main screen)
11. Raw data expander

**Mode Fokus** sidebar checkbox (default ON) collapses all expanders by default:
`expanded=not mode_fokus`.

---

## GEX legend labels

Inside the `ℹ️ Lihat semua kombinasi` expander:
- Below spot → "Di bawah SPOT" (green)
- Above spot → "Di atas SPOT" (gray)

(Previously was "Kiri Flip / Kanan Flip" — changed permanently.)

---

## Layer 1 gap info (Keputusan panel)

After "REZIM" lapis 1 label, a small colored span shows distance from spot to
gamma flip:
- Green `#26a69a` if gap > 1%
- Yellow `#fbbf24` if gap 0.5–1%
- Red `#ef5350` if gap < 0.5%

Format: `(Gap $54 · 1.3%)`

---

## Local launch

```
run.bat
```

- Activates `venv\Scripts\activate.bat`
- `start /b streamlit run app.py` (non-blocking)
- `timeout /t 3` (wait for server)
- `start http://localhost:8501` (single tab)
- `headless = true` in config.toml stops Streamlit's own auto-open

**Single tab**: `headless=true` + manual `start` after delay = exactly one browser
tab. Without `headless=true`, Streamlit opens its own tab AND `start` opens another.

---

## Decisions NOT to revisit without good reason

- **No fixedrange anywhere** — was tried (full lock, selective lock, IV-only lock)
  and all caused worse UX than fully free axes with `dragmode=pan`.
- **`_cached_*` functions catch inside, not at call site** — so failure IS cached;
  prevents hammering Yahoo while rate-limited.
- **Chain sanity guard at app level, not inside `_cached_chain`** — sanity runs on
  every rerun; a "passes sanity" result should not be cached because spot changes.
- **Regime badge (LONG/SHORT GAMMA) inside Legacy expander only** — was on main
  screen, removed to reduce noise. Keputusan 5 lapis is the single authoritative signal.
- **TTL chain = 300s** (was 180s) — reduced Yahoo request frequency on Cloud.
- **`iv_history.csv` gitignored** — it's a local ephemeral log; Cloud gets a fresh
  file each deploy.
- **`tzdata` in requirements.txt** — required for `ZoneInfo` to work on Linux-based
  Streamlit Cloud (Windows has it built-in; Linux does not).

---

## Pushing to GitHub

Remote: `https://github.com/annistarahman-max/gex-dashboard.git`

No credential helper is configured in the dev environment's git config.
**Push via GitHub Desktop** — it has stored credentials. Claude cannot `git push`
directly from the terminal in this environment.
