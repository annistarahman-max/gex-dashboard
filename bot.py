"""
GEX Dashboard Telegram Bot
===========================
Single command /info — disguised as manual analysis.
Launch: python bot.py
"""

from datetime import datetime, timezone, timedelta
from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, ContextTypes

from src.data.data_provider import get_provider
from src.engine.exposure import compute_exposures, aggregate_by_strike, summarize

import yfinance as yf
import numpy as np
import pandas as pd

BOT_TOKEN = "8813543907:AAH5wQSkboiKzqAGR9lccRuOBE0lkCPgCOg"
CHANNEL_ID = -1002791495721
WIB = timezone(timedelta(hours=7))


def _fetch_ul_price(yf_ticker):
    tk = yf.Ticker(yf_ticker)
    info = tk.fast_info
    price = getattr(info, "last_price", None)
    if not price:
        hist = tk.history(period="1d")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
    return float(price) if price else None


def _get_data():
    provider = get_provider()
    spot = provider.fetch_spot("GLD")
    chain = provider.fetch_chain("GLD")

    expiry_dates = sorted(chain["expiry"].dt.date.unique())
    today = datetime.now().date()
    nearest = None
    for d in expiry_dates:
        if d >= today:
            nearest = d
            break
    if nearest is None and expiry_dates:
        nearest = expiry_dates[-1]

    nearest_str = str(nearest)
    exposures = compute_exposures(chain, spot)
    agg = aggregate_by_strike(exposures, expiry_filter=nearest_str)
    kpi = summarize(
        exposures[exposures["expiry"] == pd.Timestamp(nearest_str)], spot
    )

    if len(agg) < 3 or ((agg["gex"] > 0).sum() == 0 and (agg["gex"] < 0).sum() == 0):
        agg = aggregate_by_strike(exposures)
        kpi = summarize(exposures, spot)
        nearest_str = "All Expiries"

    ul_price = _fetch_ul_price("GC=F")
    conv_ratio = ul_price / spot if ul_price and spot > 0 else 1.0

    return spot, agg, kpi, ul_price, conv_ratio, nearest_str


def _fmt(v):
    abs_v = abs(v)
    if abs_v >= 1e9:
        return f"${v / 1e9:+,.2f}B"
    if abs_v >= 1e6:
        return f"${v / 1e6:+,.2f}M"
    if abs_v >= 1e3:
        return f"${v / 1e3:+,.1f}K"
    return f"${v:+,.0f}"


def _find_key_levels(agg, kpi, conv_ratio, is_long):
    gex_res = set()
    gex_sup = set()

    if is_long:
        for _, r in agg[agg["gex"] > 0].nlargest(5, "gex").iterrows():
            gex_res.add(round(r["strike"] * conv_ratio))
        for _, r in agg[agg["gex"] < 0].nsmallest(5, "gex").iterrows():
            gex_sup.add(round(r["strike"] * conv_ratio))
    else:
        for _, r in agg[agg["gex"] > 0].nlargest(5, "gex").iterrows():
            gex_res.add(round(r["strike"] * conv_ratio))
            gex_sup.add(round(r["strike"] * conv_ratio))

    vex_magnet = set()
    for _, r in agg[agg["vex"] > 0].nlargest(5, "vex").iterrows():
        vex_magnet.add(round(r["strike"] * conv_ratio))

    res_overlap = gex_res & vex_magnet
    sup_overlap = gex_sup & vex_magnet

    if not res_overlap:
        res_overlap = gex_res
    if not sup_overlap:
        sup_overlap = gex_sup

    res_list = sorted(res_overlap, reverse=True)[:3]
    sup_list = sorted(sup_overlap)[:3]

    return res_list, sup_list


async def cmd_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Analyzing...")

    spot, agg, kpi, ul_price, conv_ratio, expiry = _get_data()
    now = datetime.now(WIB).strftime("%H:%M WIB — %d %b %Y")

    spot_xau = spot * conv_ratio

    # Session
    hour_wib = datetime.now(WIB).hour
    if 7 <= hour_wib < 14:
        session = "🌏 Session: Asia"
    elif 14 <= hour_wib < 20:
        session = "🌍 Session: Europe"
    else:
        session = "🌎 Session: New York"

    # Regime
    if kpi.gamma_flip and spot > kpi.gamma_flip:
        regime = "🟢 Market: Stable — Range Bound"
        bias = "Buy & Sell (scalp)"
        is_long = True
    elif kpi.gamma_flip:
        regime = "🔴 Market: Volatile — Trending"
        is_long = False
    else:
        regime = "⚫ Market: Neutral"
        bias = "Wait"
        is_long = False

    # Flip
    flip_xau = kpi.gamma_flip * conv_ratio if kpi.gamma_flip else 0
    diff = abs(spot_xau - flip_xau) if kpi.gamma_flip else 0

    # Risk level
    if diff > 150:
        risk = "⚡⚡⚡ High"
    elif diff > 50:
        risk = "⚡⚡ Medium"
    else:
        risk = "⚡ Low"

    # Mood
    if kpi.total_cex < -1e9 and not is_long:
        mood = "😱 Mood: Fear"
    elif kpi.total_cex > 1e9 and is_long:
        mood = "🤑 Mood: Greed"
    elif is_long:
        mood = "😊 Mood: Calm"
    else:
        mood = "😰 Mood: Cautious"

    # Signal count
    signals = [
        kpi.net_delta > 0,
        kpi.net_gamma > 0,
        kpi.total_gex > 0,
        kpi.total_vex > 0,
        kpi.total_cex > 0,
        kpi.gamma_flip is not None and spot > kpi.gamma_flip,
    ]
    buy_count = sum(1 for b in signals if b)
    sell_count = 6 - buy_count

    # CEX bias
    if kpi.total_cex > 1e6:
        cex_text = f"📈 Daily Flow: Buying ({_fmt(kpi.total_cex)})"
        if not is_long:
            bias = "Buy (with caution)"
    elif kpi.total_cex < -1e6:
        cex_text = f"📉 Daily Flow: Selling ({_fmt(kpi.total_cex)})"
        if not is_long:
            bias = "Sell"
    else:
        cex_text = "➖ Daily Flow: Neutral"
        if not is_long:
            bias = "Wait"

    if is_long:
        pass

    # Net Delta
    if kpi.net_delta > 0:
        delta_text = f"💪 Net Delta: {_fmt(kpi.net_delta)} (Bullish)"
    else:
        delta_text = f"👎 Net Delta: {_fmt(kpi.net_delta)} (Bearish)"

    # Put/Call ratio
    total_call = agg["call_oi"].sum() if "call_oi" in agg.columns else 0
    total_put = agg["put_oi"].sum() if "put_oi" in agg.columns else 0
    pcr = total_put / total_call if total_call > 0 else 0
    if pcr > 1.2:
        pcr_text = f"⚖️ P/C Ratio: {pcr:.2f} (Put Heavy)"
    elif pcr < 0.8:
        pcr_text = f"⚖️ P/C Ratio: {pcr:.2f} (Call Heavy)"
    else:
        pcr_text = f"⚖️ P/C Ratio: {pcr:.2f} (Balanced)"

    # Key levels (GEX + VEX overlap)
    res_list, sup_list = _find_key_levels(agg, kpi, conv_ratio, is_long)

    # VEX magnet/tolak
    vex_pos = agg[agg["vex"] > 0].nlargest(1, "vex")
    vex_neg = agg[agg["vex"] < 0].nsmallest(1, "vex")
    magnet = f"${float(vex_pos['strike'].iloc[0]) * conv_ratio:,.0f}" if len(vex_pos) > 0 else "-"
    tolak = f"${float(vex_neg['strike'].iloc[0]) * conv_ratio:,.0f}" if len(vex_neg) > 0 else "-"

    res_lines = ""
    for p in res_list:
        res_lines += f"  🔺 ${p:,.0f}\n"
    if not res_lines:
        res_lines = "  -\n"

    sup_lines = ""
    for p in sup_list:
        sup_lines += f"  🔻 ${p:,.0f}\n"
    if not sup_lines:
        sup_lines = "  -\n"

    # Setup
    if "Sell" in bias:
        setup = (
            f"⚡ *Setup: SELL*\n"
            f"  → Entry di 🔺 Resistance\n"
            f"  → SL: $10-15 di atas\n"
            f"  → TP: Support terdekat\n"
            f"  → Confidence: {sell_count}/6\n"
        )
    elif "Buy" in bias and "caution" in bias:
        setup = (
            f"⚡ *Setup: CAUTION*\n"
            f"  → Sinyal campur\n"
            f"  → Lot kecil, SL ketat\n"
            f"  → Atau tunggu konfirmasi\n"
            f"  → Confidence: {buy_count}/6\n"
        )
    elif "Buy" in bias:
        setup = (
            f"⚡ *Setup: BUY*\n"
            f"  → Entry di 🔻 Support\n"
            f"  → SL: $10-15 di bawah\n"
            f"  → TP: Resistance terdekat\n"
            f"  → Confidence: {buy_count}/6\n"
        )
    elif "scalp" in bias:
        setup = (
            f"⚡ *Setup: SCALP*\n"
            f"  → BUY di 🔻 Support\n"
            f"  → SELL di 🔺 Resistance\n"
            f"  → Bolak-balik, SL ketat\n"
        )
    else:
        setup = (
            f"⚡ *Setup: WAIT*\n"
            f"  → No clear direction\n"
        )

    msg = (
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *XAUUSD ANALYSIS*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"

        f"{session}\n"
        f"{regime}\n"
        f"{mood}\n\n"

        f"💰 Spot: ${spot_xau:,.0f}\n"
        f"🎯 Pivot: ${flip_xau:,.0f}\n"
        f"📏 Gap: ${diff:,.0f}\n"
        f"🔥 Risk: {risk}\n"
        f"{delta_text}\n"
        f"{pcr_text}\n\n"

        f"📍 *Resistance:*\n{res_lines}\n"
        f"📍 *Support:*\n{sup_lines}\n"

        f"🧲 Magnet: {magnet}\n"
        f"💨 Tolak: {tolak}\n\n"

        f"{cex_text}\n\n"

        f"{setup}\n"

        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ DYOR — Not financial advice\n"
        f"⏰ {now}"
    )
    await update.message.reply_text("✅ Analisa terkirim ke channel!", parse_mode="Markdown")
    await context.bot.send_message(chat_id=CHANNEL_ID, text=msg, parse_mode="Markdown")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"📊 *XAUUSD Analysis*\n\n"
        f"Ketik /info untuk analisa terbaru."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def post_init(app: Application):
    await app.bot.set_my_commands([
        BotCommand("info", "XAUUSD Analysis"),
    ])


def main():
    print("Bot starting...")
    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("info", cmd_info))
    print("Bot is running! Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
