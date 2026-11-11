#!/usr/bin/env python3
"""
Binance top-100 SMA20/SMA200 crossover scanner (15m) - spot + futures.
Built for GitHub Actions: each run scans once, emails new crosses
(up AND down) and appends a short market snapshot at the bottom.

Environment variables (GitHub Secrets):
    ALERT_EMAIL_USER   Gmail address that sends the mail
    ALERT_EMAIL_PASS   Gmail App Password
    ALERT_EMAIL_TO     (optional) where to receive alerts, defaults to USER
    FUTURES_BASE       (optional) alternative futures API base URL

Local use:
    python scanner.py --dry     # scan and print, no email, no state saved
    python scanner.py --test    # send a test email
"""
import os
import sys
import json
import time
import smtplib
import ssl
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.message import EmailMessage

import requests

# ---------------- Settings (edit these) ----------------
MARKETS = ["spot", "futures"]
TOP_N = 100                                   # top N USDT pairs by 24h volume, per market
EXTRA_SYMBOLS = {"spot": [], "futures": ["ARIAUSDT"]}   # always include these
INTERVAL = "15m"
INTERVAL_MS = 15 * 60 * 1000
FAST, SLOW = 20, 200
LOOKBACK = 2          # also check the last N closed candles, so a delayed run misses nothing
WORKERS = 8
STATE_FILE = "state.json"

EMAIL_USER = os.environ.get("ALERT_EMAIL_USER", "")
EMAIL_PASS = os.environ.get("ALERT_EMAIL_PASS", "")
EMAIL_TO = os.environ.get("ALERT_EMAIL_TO") or EMAIL_USER
SMTP_HOST = os.environ.get("ALERT_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("ALERT_SMTP_PORT", "465"))

FUT = os.environ.get("FUTURES_BASE", "https://fapi.binance.com").rstrip("/")
ENDPOINTS = {
    "spot": {
        "klines": ["https://data-api.binance.vision/api/v3/klines", "https://api.binance.com/api/v3/klines"],
        "ticker": ["https://data-api.binance.vision/api/v3/ticker/24hr", "https://api.binance.com/api/v3/ticker/24hr"],
    },
    "futures": {
        "klines": [f"{FUT}/fapi/v1/klines"],
        "ticker": [f"{FUT}/fapi/v1/ticker/24hr"],
    },
}
STABLES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "USD1", "PYUSD", "XUSD", "AEUR", "EUR"}


# ---------------- Networking ----------------
def get_json(urls, params=None, retries=2):
    last = None
    for url in urls:
        for attempt in range(retries + 1):
            try:
                r = requests.get(url, params=params, timeout=20)
                if r.status_code in (451, 403):
                    last = RuntimeError(f"HTTP {r.status_code} from {url.split('/')[2]} (region blocked)")
                    break  # try next URL
                if r.status_code in (418, 429):
                    last = RuntimeError("rate limited by Binance")
                    time.sleep(2 * (attempt + 1))
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                last = e
                time.sleep(1)
    raise last


def top_symbols(market):
    data = get_json(ENDPOINTS[market]["ticker"])
    rows = []
    for t in data:
        s = t["symbol"]
        if not s.endswith("USDT") or "_" in s or s[:-4] in STABLES:
            continue
        rows.append((float(t["quoteVolume"]), s, float(t.get("priceChangePercent") or 0)))
    rows.sort(reverse=True)
    top = rows[:TOP_N]
    symbols = [s for _, s, _ in top]
    changes = {s: c for _, s, c in top}
    for extra in EXTRA_SYMBOLS.get(market, []):
        if extra not in symbols:
            symbols.append(extra)
    return symbols, changes


def fetch(market, symbol):
    """Return (symbol, data|None, error|None); only closed candles are kept."""
    try:
        rows = get_json(
            ENDPOINTS[market]["klines"],
            {"symbol": symbol, "interval": INTERVAL, "limit": SLOW + LOOKBACK + 10},
        )
        now_ms = int(time.time() * 1000)
        rows = [k for k in rows if k[6] < now_ms]
        return symbol, {
            "times": [int(k[0]) for k in rows],
            "closes": [float(k[4]) for k in rows],
            "vols": [float(k[5]) for k in rows],
        }, None
    except Exception as e:
        return symbol, None, str(e)


# ---------------- Indicators ----------------
def sma(closes, n, end):
    return sum(closes[end - n:end]) / n


def rsi(closes, period=14):
    closes = closes[-150:]
    if len(closes) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0)
        losses += max(-d, 0)
    ag, al = gains / period, losses / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (period - 1) + max(d, 0)) / period
        al = (al * (period - 1) + max(-d, 0)) / period
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def find_crosses(d):
    """Return [(direction, candle_index)] for the last LOOKBACK closed candles."""
    c = d["closes"]
    out = []
    if len(c) < SLOW + LOOKBACK:
        return out
    for back in range(LOOKBACK):
        end = len(c) - back
        f1, s1 = sma(c, FAST, end), sma(c, SLOW, end)
        f0, s0 = sma(c, FAST, end - 1), sma(c, SLOW, end - 1)
        if f0 <= s0 and f1 > s1:
            out.append(("up", end - 1))
        elif f0 >= s0 and f1 < s1:
            out.append(("down", end - 1))
    return out


# ---------------- Scanning ----------------
def scan_market(market, state, notes):
    try:
        symbols, changes = top_symbols(market)
    except Exception as e:
        notes.append(f"{market.upper()} data unavailable: {e}")
        return [], None

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(lambda s: fetch(market, s), symbols))

    data = {s: d for s, d, _ in results if d and len(d["closes"]) >= SLOW + LOOKBACK}
    errors = [e for _, d, e in results if e]
    print(f"{market}: {len(data)}/{len(symbols)} pairs loaded, {len(errors)} errors")
    if not data:
        notes.append(f"{market.upper()} data unavailable: {errors[0] if errors else 'no data'}")
        return [], None

    hits, bullish = [], 0
    for sym, d in data.items():
        c = d["closes"]
        if sma(c, FAST, len(c)) > sma(c, SLOW, len(c)):
            bullish += 1
        for direction, idx in find_crosses(d):
            key = f"{market}:{sym}:{d['times'][idx]}:{direction}"
            if key in state["sent"]:
                continue
            vols = d["vols"]
            base = vols[max(0, idx - 20):idx]
            vol_ratio = vols[idx] / (sum(base) / len(base)) if base and sum(base) > 0 else None
            fast, slow = sma(c, FAST, idx + 1), sma(c, SLOW, idx + 1)
            hits.append({
                "key": key, "market": market, "symbol": sym, "dir": direction,
                "close": c[idx], "rsi": rsi(c[:idx + 1]), "vol": vol_ratio,
                "gap": (fast - slow) / slow * 100,
                "time": datetime.fromtimestamp((d["times"][idx] + INTERVAL_MS) / 1000, tz=timezone.utc),
            })

    snap = {"n": len(data), "breadth": bullish / len(data) * 100,
            "chg": sum(changes.get(s, 0) for s in data) / len(data), "btc": None}
    if "BTCUSDT" in data:
        c = data["BTCUSDT"]["closes"]
        f, s = sma(c, FAST, len(c)), sma(c, SLOW, len(c))
        snap["btc"] = {"above": c[-1] > s, "up": f > s, "rsi": rsi(c)}
    return hits, snap


# ---------------- Email text ----------------
def fmt_hit(h):
    rsi_txt = f"RSI {h['rsi']:.0f}" if h["rsi"] is not None else "RSI n/a"
    vol_txt = f"vol x{h['vol']:.1f}" if h["vol"] is not None else "vol n/a"
    return (f"  {h['symbol']} ({h['market']})  close {h['close']:g} | {rsi_txt} | {vol_txt} "
            f"| {h['time']:%H:%M} UTC")


def names(hs, limit=6):
    out = ", ".join(h["symbol"].replace("USDT", "") for h in hs[:limit])
    return out + ("..." if len(hs) > limit else "")


def market_read(snaps, hits):
    live = [s for s in snaps.values() if s]
    if not live:
        return []
    avg = sum(s["breadth"] for s in live) / len(live)
    if avg >= 60:
        lines = ["Bullish-leaning: most top pairs are in short-term uptrends (SMA20 above SMA200)."]
    elif avg <= 40:
        lines = ["Bearish-leaning: most top pairs are in short-term downtrends (SMA20 below SMA200)."]
    else:
        lines = ["Mixed / range-bound: no clear trend across the market."]

    ups = [h for h in hits if h["dir"] == "up"]
    downs = [h for h in hits if h["dir"] == "down"]
    if len(ups) >= 3 and len(ups) > 2 * len(downs):
        lines.append("Bullish crosses dominate this scan, so momentum is turning up broadly.")
    elif len(downs) >= 3 and len(downs) > 2 * len(ups):
        lines.append("Bearish crosses dominate this scan, so momentum is turning down broadly.")

    btc = next((s["btc"] for s in live if s["btc"]), None)
    if btc:
        btc_bull = btc["above"] and btc["up"]
        btc_bear = (not btc["above"]) and (not btc["up"])
        if ups and btc_bull:
            lines.append("Bullish crosses align with a bullish BTC trend, which is a supportive backdrop.")
        elif ups and btc_bear:
            lines.append("Bullish crosses are going against a bearish BTC trend, so be extra cautious.")
        if downs and btc_bear:
            lines.append("Bearish crosses align with a bearish BTC trend.")
        elif downs and btc_bull:
            lines.append("Bearish crosses are going against a bullish BTC trend, so they may be pullbacks.")

    def ok(h):
        if h["vol"] is None or h["rsi"] is None or h["vol"] < 1.5:
            return False
        return (h["dir"] == "up" and 45 <= h["rsi"] <= 70) or (h["dir"] == "down" and 30 <= h["rsi"] <= 55)

    strong = [h for h in hits if ok(h)]
    stretched = [h for h in hits if h["rsi"] is not None and
                 ((h["dir"] == "up" and h["rsi"] > 75) or (h["dir"] == "down" and h["rsi"] < 25))]
    weak = [h for h in hits if h["vol"] is not None and h["vol"] < 0.8]
    if strong:
        lines.append(f"Higher-quality looking (volume 1.5x+ and RSI not stretched): {names(strong)}.")
    if stretched:
        lines.append(f"Already stretched (RSI extreme), so the move may be late: {names(stretched)}.")
    if weak:
        lines.append(f"Low-volume crosses, so they are more prone to fake-outs: {names(weak)}.")
    return lines


def snapshot_text(market, snap):
    if not snap:
        return f"{market.upper()}: data unavailable"
    line = (f"{market.upper()}: {snap['breadth']:.0f}% of {snap['n']} pairs have SMA20 above SMA200; "
            f"average 24h change {snap['chg']:+.1f}%.")
    b = snap["btc"]
    if b:
        rsi_txt = f"{b['rsi']:.0f}" if b["rsi"] is not None else "n/a"
        line += (f"\n  BTC: price {'above' if b['above'] else 'below'} its SMA200, "
                 f"SMA20 {'above' if b['up'] else 'below'} SMA200, RSI14 {rsi_txt}.")
    return line


def build_email(hits, snaps, notes):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ups = sorted([h for h in hits if h["dir"] == "up"], key=lambda h: (h["market"], h["symbol"]))
    downs = sorted([h for h in hits if h["dir"] == "down"], key=lambda h: (h["market"], h["symbol"]))
    parts = [f"{INTERVAL} SMA{FAST}/{SLOW} crossovers - {stamp}\n"]
    for n in notes:
        parts.append(f"WARNING: {n}\n")
    if ups:
        parts.append("BULLISH (SMA20 crossed ABOVE SMA200):\n" + "\n".join(fmt_hit(h) for h in ups) + "\n")
    if downs:
        parts.append("BEARISH (SMA20 crossed BELOW SMA200):\n" + "\n".join(fmt_hit(h) for h in downs) + "\n")
    parts.append("-" * 40 + "\nMARKET SNAPSHOT")
    parts.extend(snapshot_text(m, s) for m, s in snaps.items())
    read = market_read(snaps, hits)
    if read:
        parts.append("\nQUICK READ")
        parts.extend(f"- {line}" for line in read)
    parts.append("\nNote: 15m crossovers lag price and often whipsaw in sideways markets. "
                 "Consider confirming on 1h/4h and using a stop-loss. This is not financial advice.")
    subject = f"SMA{FAST}/{SLOW} cross: {len(ups)} up, {len(downs)} down"
    if not hits:
        subject = "Scanner warning: data source problem"
    return subject, "\n".join(parts)


# ---------------- State / email ----------------
def load_state():
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except Exception:
        state = {}
    state.setdefault("sent", {})
    state.setdefault("warn_day", "")
    cutoff = time.time() - 2 * 24 * 3600
    state["sent"] = {k: v for k, v in state["sent"].items() if v > cutoff}
    return state


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=1)


def send_email(subject, body):
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, EMAIL_USER, EMAIL_TO
    msg.set_content(body)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context()) as s:
        s.login(EMAIL_USER, EMAIL_PASS)
        s.send_message(msg)


def main():
    dry = "--dry" in sys.argv
    if "--test" in sys.argv:
        send_email("SMA scanner test", "Your scanner can send email.")
        print("Test email sent.")
        return
    if not dry and not (EMAIL_USER and EMAIL_PASS):
        sys.exit("Set ALERT_EMAIL_USER and ALERT_EMAIL_PASS first.")

    state = load_state()
    hits, snaps, notes = [], {}, []
    for market in MARKETS:
        h, snap = scan_market(market, state, notes)
        hits += h
        snaps[market] = snap

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    warn_due = bool(notes) and state["warn_day"] != today
    if not hits and not warn_due:
        print("No new crosses.")
        save_state(state) if not dry else None
        return

    subject, body = build_email(hits, snaps, notes)
    print(body)
    if dry:
        return
    send_email(subject, body)
    now = time.time()
    for h in hits:
        state["sent"][h["key"]] = now
    if notes:
        state["warn_day"] = today
    save_state(state)
    print("Email sent.")


if __name__ == "__main__":
    main()
