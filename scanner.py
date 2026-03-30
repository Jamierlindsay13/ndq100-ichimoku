#!/usr/bin/env python3
"""
Nasdaq 100 MTF Ichimoku Scanner
Calculates Ichimoku signals across 1m, 10m, 1H, 1D for all NDX 100 stocks.
Output: ichimoku_scan.html  (open in any browser — sortable, filterable)

Usage:
    python scanner.py
"""

import warnings
import json
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# NASDAQ 100 TICKERS
# ─────────────────────────────────────────────────────────────────────────────
_RAW_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","ASML","AMD","AZN","TMUS","LIN","ISRG","CSCO","INTU","PEP",
    "QCOM","TXN","CMCSA","AMGN","HON","AMAT","MU","BKNG","PANW","VRTX",
    "ADI","SBUX","REGN","MELI","LRCX","KLAC","GILD","MDLZ","SNPS","CDNS",
    "ADP","CSX","MRVL","PYPL","MAR","CTAS","ORLY","NXPI","MNST","PCAR",
    "ADSK","WDAY","CHTR","ROST","MCHP","LULU","DXCM","CPRT","ODFL","CRWD",
    "FTNT","BIIB","KDP","FAST","EA","DLTR","BKR","GEHC","EXC","VRSK",
    "XEL","CSGP","CTSH","ON","TEAM","TTD","FSLR","DDOG","ZS","IDXX",
    "ILMN","ALGN","CDW","ENPH","CCEP","CEG","SMCI","ARM","APP","PLTR",
    "INTC","HOOD","PAYX","OKTA","ZM","RIVN","MDB","FANG","LCID","SIRI",
]
_seen = set()
TICKERS = [t for t in _RAW_TICKERS if not (t in _seen or _seen.add(t))]

# ─────────────────────────────────────────────────────────────────────────────
# ICHIMOKU ENGINE
# ─────────────────────────────────────────────────────────────────────────────
TENKAN_P   = 9
KIJUN_P    = 26
SENKOU_B_P = 52
DISPLACE   = 26
MIN_BARS   = SENKOU_B_P + DISPLACE + 5   # 83 bars minimum


def calc_ichimoku(df: pd.DataFrame) -> dict | None:
    """Compute Ichimoku signal for an OHLCV DataFrame. Returns None if insufficient data."""
    if df is None or len(df) < MIN_BARS:
        return None

    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    tenkan = (h.rolling(TENKAN_P).max() + l.rolling(TENKAN_P).min()) / 2
    kijun  = (h.rolling(KIJUN_P).max()  + l.rolling(KIJUN_P).min())  / 2

    span_a_raw = (tenkan + kijun) / 2
    span_b_raw = (h.rolling(SENKOU_B_P).max() + l.rolling(SENKOU_B_P).min()) / 2

    # Current cloud: spans shifted forward DISPLACE periods
    # .shift(n).iloc[-1] == raw value from n bars ago
    span_a = span_a_raw.shift(DISPLACE)
    span_b = span_b_raw.shift(DISPLACE)

    cur_c  = float(c.iloc[-1])
    cur_t  = _safe(tenkan.iloc[-1])
    cur_k  = _safe(kijun.iloc[-1])
    cur_sa = _safe(span_a.iloc[-1])
    cur_sb = _safe(span_b.iloc[-1])

    if any(v is None for v in [cur_t, cur_k, cur_sa, cur_sb]):
        return None

    # Future cloud color (26 bars ahead)
    fut_sa = _safe(span_a_raw.iloc[-1])
    fut_sb = _safe(span_b_raw.iloc[-1])

    # Chikou: close vs price 27 bars ago
    chikou_ref = float(c.iloc[-(DISPLACE + 1)]) if len(c) > DISPLACE + 1 else None

    cloud_top    = max(cur_sa, cur_sb)
    cloud_bottom = min(cur_sa, cur_sb)

    # ── Score components ────────────────────────────────────────────────────
    # Price vs cloud  (±2)
    if cur_c > cloud_top:
        price_pos, pp_s = "above",  2
    elif cur_c < cloud_bottom:
        price_pos, pp_s = "below", -2
    else:
        price_pos, pp_s = "inside", 0

    # TK cross  (±1)
    td = cur_t - cur_k
    if td > 0:   tk, tk_s = "bull",    1
    elif td < 0: tk, tk_s = "bear",   -1
    else:        tk, tk_s = "neutral", 0

    # Future cloud color  (±1)
    if fut_sa is not None and fut_sb is not None:
        if fut_sa > fut_sb:   cloud_col, cc_s = "bull",    1
        elif fut_sa < fut_sb: cloud_col, cc_s = "bear",   -1
        else:                 cloud_col, cc_s = "neutral", 0
    else:
        cloud_col, cc_s = "neutral", 0

    # Chikou  (±1)
    if chikou_ref is not None:
        if cur_c > chikou_ref:   chi, chi_s = "bull",    1
        elif cur_c < chikou_ref: chi, chi_s = "bear",   -1
        else:                    chi, chi_s = "neutral", 0
    else:
        chi, chi_s = "n/a", 0

    total = pp_s + tk_s + cc_s + chi_s   # range: -5 … +5

    if total >= 3:    signal = "BULL"
    elif total <= -3: signal = "BEAR"
    else:             signal = "NEUTRAL"

    return {
        "signal":       signal,
        "score":        total,
        "price_pos":    price_pos,
        "tk":           tk,
        "cloud_color":  cloud_col,
        "chikou":       chi,
        "cloud_top":    round(cloud_top, 3),
        "cloud_bottom": round(cloud_bottom, 3),
        "tenkan":       round(cur_t, 3),
        "kijun":        round(cur_k, 3),
    }


def _safe(v):
    return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)


# ─────────────────────────────────────────────────────────────────────────────
# SQN(100) ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def calc_sqn(df: pd.DataFrame, period: int = 100) -> dict | None:
    """
    SQN(100) = (mean(daily_returns[-100:]) / std(daily_returns[-100:])) * sqrt(100)
    Thresholds: Bull Volatile >1.7 | Bull Quiet >0.6 | Neutral ±0.6 | Bear Quiet <-0.6 | Bear Volatile <-1.7
    """
    if df is None or len(df) < period + 2:
        return None
    closes = df["Close"].astype(float)
    rets   = closes.pct_change().dropna()
    last   = rets.iloc[-period:]
    if len(last) < period:
        return None
    mean_r = float(last.mean())
    std_r  = float(last.std(ddof=1))
    if std_r == 0:
        return None
    sqn = round((mean_r / std_r) * np.sqrt(period), 2)

    if sqn > 1.7:
        label, css = "Bull Volatile", "sqn-blue"
    elif sqn > 0.6:
        label, css = "Bull Quiet",    "sqn-green"
    elif sqn >= -0.6:
        label, css = "Neutral",       "sqn-yellow"
    elif sqn >= -1.7:
        label, css = "Bear Quiet",    "sqn-red"
    else:
        label, css = "Bear Volatile", "sqn-crimson"

    return {"value": sqn, "label": label, "css": css}


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
def _parse_batch(raw: pd.DataFrame, tickers: list) -> dict:
    """Unpack a yf.download MultiIndex result into {ticker: DataFrame}."""
    out = {}
    if raw is None or raw.empty:
        return out
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                lvl0 = raw.columns.get_level_values(0).unique().tolist()
                lvl1 = raw.columns.get_level_values(1).unique().tolist()
                if t in lvl0:
                    df = raw[t].copy()
                elif t in lvl1:
                    df = raw.xs(t, axis=1, level=1).copy()
                else:
                    continue
            else:
                df = raw.copy()
            df.index = pd.to_datetime(df.index)
            df = df.dropna(how="all")
            if len(df) >= 20:
                out[t] = df
        except (KeyError, TypeError):
            pass
    return out


def fetch(tickers: list, interval: str, period: str, label: str) -> dict:
    print(f"  [{label:>4}] downloading {interval:>3} data … ", end="", flush=True)
    try:
        raw = yf.download(
            tickers, interval=interval, period=period,
            group_by="ticker", auto_adjust=True, progress=False,
        )
        result = _parse_batch(raw, tickers)
        print(f"{len(result)}/{len(tickers)} ok", flush=True)
        return result
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        return {}


def resample_10m(data_1m: dict) -> dict:
    out = {}
    for t, df in data_1m.items():
        try:
            df10 = df.resample("10min").agg(
                {"Open": "first", "High": "max", "Low": "min",
                 "Close": "last", "Volume": "sum"}
            # Volume.sum() returns 0 (not NaN) for empty bins, so dropna(how='all')
            # lets overnight/weekend ghost rows survive and corrupt rolling windows.
            # Drop on Close instead to keep only real traded bars.
            ).dropna(subset=["Close"])
            if len(df10) >= MIN_BARS:
                out[t] = df10
        except Exception:
            pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SCANNER
# ─────────────────────────────────────────────────────────────────────────────
def scan(tickers: list) -> list:
    print("\nStep 1/3  Fetching market data")
    data_1m = fetch(tickers, "1m",  "5d",  "1m")
    data_1h = fetch(tickers, "1h",  "60d", "1H")
    data_1d = fetch(tickers, "1d",  "2y",  "1D")

    print("\nStep 2/3  Resampling 1m -> 10m")
    data_10m = resample_10m(data_1m)
    print(f"  [10m]  {len(data_10m)}/{len(data_1m)} tickers resampled ok")

    print("\nStep 3/3  Calculating Ichimoku signals")
    results = []

    for t in tickers:
        # Price and daily change from 1D data
        price = change_pct = None
        df_d = data_1d.get(t)
        if df_d is not None and len(df_d) >= 2:
            price      = round(float(df_d["Close"].iloc[-1]), 2)
            prev_c     = float(df_d["Close"].iloc[-2])
            change_pct = round((price - prev_c) / prev_c * 100, 2) if prev_c else None

        i1m  = calc_ichimoku(data_1m.get(t))
        i10m = calc_ichimoku(data_10m.get(t))
        i1h  = calc_ichimoku(data_1h.get(t))
        i1d  = calc_ichimoku(df_d)
        sqn  = calc_sqn(df_d)

        tfs    = [i1m, i10m, i1h, i1d]
        valid  = [tf for tf in tfs if tf is not None]
        bulls  = sum(1 for tf in valid if tf["signal"] == "BULL")
        bears  = sum(1 for tf in valid if tf["signal"] == "BEAR")
        n_valid = len(valid)

        if n_valid == 0:
            alignment = "N/A"
        elif bulls == n_valid:
            alignment = "Full Bull"
        elif bears == n_valid:
            alignment = "Full Bear"
        elif bulls >= 3:
            alignment = "Mostly Bull"
        elif bears >= 3:
            alignment = "Mostly Bear"
        else:
            alignment = "Mixed"

        results.append({
            "ticker":     t,
            "price":      price,
            "change_pct": change_pct,
            "1m":         i1m,
            "10m":        i10m,
            "1h":         i1h,
            "1d":         i1d,
            "bulls":      bulls,
            "bears":      bears,
            "n_valid":    n_valid,
            "alignment":  alignment,
            "sqn":        sqn,
        })

        bar = "".join(
            ("^" if tf and tf["signal"]=="BULL" else ("v" if tf and tf["signal"]=="BEAR" else "-"))
            for tf in tfs
        )
        print(f"  {t:6s}  {bar}  {alignment}", flush=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# HTML GENERATION
# ─────────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NDX 100 — MTF Ichimoku Scanner</title>
<style>
:root{
  --bg:#0d1117;--surface:#161b22;--surface2:#1f2937;--border:#30363d;
  --text:#e6edf3;--muted:#8b949e;--accent:#58a6ff;
  --green:#3fb950;--green-dim:rgba(63,185,80,.12);
  --red:#f85149;--red-dim:rgba(248,81,73,.12);
  --yellow:#d29922;--yellow-dim:rgba(210,153,34,.12);
  --purple:#bc8cff;--teal:#39d353;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;font-size:14px;min-height:100vh;}

/* ── HEADER ── */
header{background:linear-gradient(135deg,#0d1117,#1a2332,#0d1117);border-bottom:1px solid var(--border);padding:28px 24px 20px;}
.header-row{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;max-width:1600px;margin:0 auto;}
.header-left h1{font-size:22px;font-weight:800;background:linear-gradient(135deg,#58a6ff,#bc8cff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.header-left p{color:var(--muted);font-size:12px;margin-top:3px;}
.scan-time{color:var(--muted);font-size:12px;text-align:right;}
.scan-time strong{color:var(--accent);}

/* ── STATS BAR ── */
.stats-bar{background:var(--surface);border-bottom:1px solid var(--border);padding:12px 24px;}
.stats-inner{display:flex;flex-wrap:wrap;gap:10px;max-width:1600px;margin:0 auto;align-items:center;}
.stat-pill{display:flex;align-items:center;gap:6px;background:var(--surface2);border:1px solid var(--border);border-radius:20px;padding:4px 12px;font-size:12px;font-weight:600;}
.stat-pill .num{font-size:16px;font-weight:800;}
.stat-pill.total .num{color:var(--accent);}
.stat-pill.fbull .num{color:var(--green);}
.stat-pill.mbull .num{color:#6ee7b7;}
.stat-pill.mixed .num{color:var(--yellow);}
.stat-pill.mbear .num{color:#fca5a5;}
.stat-pill.fbear .num{color:var(--red);}
.stat-pill.na    .num{color:var(--muted);}

/* ── CONTROLS ── */
.controls{background:var(--surface);border-bottom:1px solid var(--border);padding:10px 24px;position:sticky;top:0;z-index:50;}
.controls-inner{display:flex;flex-wrap:wrap;gap:8px;align-items:center;max-width:1600px;margin:0 auto;}
.filter-btn{border:1px solid var(--border);background:transparent;color:var(--muted);border-radius:6px;padding:5px 12px;font-size:12px;font-weight:600;cursor:pointer;transition:all .15s;}
.filter-btn:hover,.filter-btn.active{color:var(--text);border-color:var(--accent);background:rgba(88,166,255,.1);}
.filter-btn.fbull.active{border-color:var(--green);background:var(--green-dim);color:var(--green);}
.filter-btn.mbull.active{border-color:#6ee7b7;background:rgba(110,231,183,.1);color:#6ee7b7;}
.filter-btn.mixed.active{border-color:var(--yellow);background:var(--yellow-dim);color:var(--yellow);}
.filter-btn.mbear.active{border-color:#fca5a5;background:rgba(252,165,165,.1);color:#fca5a5;}
.filter-btn.fbear.active{border-color:var(--red);background:var(--red-dim);color:var(--red);}
.sep{color:var(--border);font-size:18px;user-select:none;}
#search{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:5px 10px;color:var(--text);font-size:12px;width:160px;outline:none;}
#search:focus{border-color:var(--accent);}
.count-display{margin-left:auto;color:var(--muted);font-size:12px;}
.count-display strong{color:var(--text);}

/* ── TABLE WRAPPER ── */
.table-wrap{overflow-x:auto;padding:0 24px 40px;max-width:1600px;margin:0 auto;}
table{width:100%;border-collapse:collapse;margin-top:16px;}
thead th{
  background:var(--surface);border:1px solid var(--border);
  padding:8px 10px;text-align:left;font-size:11px;font-weight:700;
  letter-spacing:.8px;text-transform:uppercase;color:var(--muted);
  cursor:pointer;user-select:none;white-space:nowrap;position:sticky;top:45px;z-index:10;
}
thead th:hover{color:var(--text);background:var(--surface2);}
thead th .sort-icon{margin-left:4px;opacity:.4;font-size:10px;}
thead th.sorted .sort-icon{opacity:1;color:var(--accent);}
tbody tr{border-bottom:1px solid var(--border);transition:background .1s;}
tbody tr:hover{background:var(--surface);}
tbody td{padding:8px 10px;vertical-align:middle;white-space:nowrap;}
tbody tr.hidden{display:none;}

/* ── CELLS ── */
.ticker-cell a{color:var(--accent);font-weight:700;font-size:13px;text-decoration:none;}
.ticker-cell a:hover{text-decoration:underline;}
.price-cell{font-weight:600;font-size:13px;}
.chg-cell{font-size:12px;font-weight:600;}
.chg-cell.up{color:var(--green);}
.chg-cell.dn{color:var(--red);}
.chg-cell.fl{color:var(--muted);}

/* ── SIGNAL BADGES ── */
.badge{display:inline-flex;align-items:center;gap:4px;border-radius:5px;padding:3px 8px;font-size:11px;font-weight:700;letter-spacing:.5px;cursor:default;position:relative;}
.badge.bull{background:var(--green-dim);border:1px solid rgba(63,185,80,.35);color:var(--green);}
.badge.bear{background:var(--red-dim);border:1px solid rgba(248,81,73,.35);color:var(--red);}
.badge.neutral{background:var(--yellow-dim);border:1px solid rgba(210,153,34,.3);color:var(--yellow);}
.badge.na{background:transparent;border:1px solid var(--border);color:var(--muted);}

/* ── TOOLTIP ── */
.badge[data-tip]:hover::after{
  content:attr(data-tip);
  position:absolute;top:calc(100% + 6px);left:50%;transform:translateX(-50%);
  background:#1e2a3a;border:1px solid var(--border);border-radius:6px;
  padding:8px 10px;font-size:11px;font-weight:400;line-height:1.6;
  color:var(--text);white-space:pre-line;min-width:200px;z-index:999;
  box-shadow:0 4px 20px rgba(0,0,0,.5);letter-spacing:0;
}
.badge[data-tip]:hover::before{
  content:'';position:absolute;top:calc(100% + 1px);left:50%;transform:translateX(-50%);
  border:5px solid transparent;border-bottom-color:#1e2a3a;z-index:1000;
}

/* ── MTF DOTS ── */
.dots{display:flex;gap:4px;align-items:center;}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block;flex-shrink:0;}
.dot.bull{background:var(--green);box-shadow:0 0 4px rgba(63,185,80,.6);}
.dot.bear{background:var(--red);box-shadow:0 0 4px rgba(248,81,73,.6);}
.dot.neutral{background:var(--yellow);box-shadow:0 0 4px rgba(210,153,34,.5);}
.dot.na{background:var(--border);}
.mtf-score{margin-left:6px;font-size:11px;font-weight:700;color:var(--muted);}
.mtf-score.all-bull{color:var(--green);}
.mtf-score.all-bear{color:var(--red);}

/* ── SQN BADGE ── */
.sqn-badge{display:inline-flex;align-items:center;gap:5px;border-radius:5px;padding:3px 8px;font-size:11px;font-weight:700;letter-spacing:.4px;cursor:default;}
.sqn-badge .sqn-val{font-size:10px;opacity:.75;font-weight:600;}
.sqn-blue   {background:rgba(96,165,250,.13);border:1px solid rgba(96,165,250,.35);color:#60a5fa;}
.sqn-green  {background:var(--green-dim);border:1px solid rgba(63,185,80,.35);color:var(--green);}
.sqn-yellow {background:var(--yellow-dim);border:1px solid rgba(210,153,34,.3);color:var(--yellow);}
.sqn-red    {background:var(--red-dim);border:1px solid rgba(248,81,73,.35);color:var(--red);}
.sqn-crimson{background:rgba(220,38,38,.15);border:1px solid rgba(220,38,38,.4);color:#f87171;}
.sqn-na     {background:transparent;border:1px solid var(--border);color:var(--muted);}

/* ── ALIGNMENT BADGE ── */
.align-badge{display:inline-block;border-radius:12px;padding:3px 10px;font-size:11px;font-weight:700;letter-spacing:.5px;}
.align-badge.Full.Bull{background:var(--green-dim);color:var(--green);border:1px solid rgba(63,185,80,.3);}
.align-badge.Mostly.Bull{background:rgba(110,231,183,.1);color:#6ee7b7;border:1px solid rgba(110,231,183,.3);}
.align-badge.Mixed{background:var(--yellow-dim);color:var(--yellow);border:1px solid rgba(210,153,34,.3);}
.align-badge.Mostly.Bear{background:rgba(252,165,165,.1);color:#fca5a5;border:1px solid rgba(252,165,165,.3);}
.align-badge.Full.Bear{background:var(--red-dim);color:var(--red);border:1px solid rgba(248,81,73,.3);}
.align-badge.NA{background:transparent;color:var(--muted);border:1px solid var(--border);}

/* ── EMPTY STATE ── */
.no-results{text-align:center;padding:60px 20px;color:var(--muted);}
.no-results p{font-size:16px;margin-bottom:8px;}

/* ── SCORE BAR ── */
.score-bar{display:flex;align-items:center;gap:4px;}
.score-num{font-size:12px;font-weight:700;min-width:14px;text-align:right;}
.score-num.pos{color:var(--green);}
.score-num.neg{color:var(--red);}
.score-num.zer{color:var(--muted);}
.score-track{width:36px;height:5px;background:var(--border);border-radius:3px;overflow:hidden;}
.score-fill{height:100%;border-radius:3px;}
.score-fill.pos{background:var(--green);}
.score-fill.neg{background:var(--red);}
</style>
</head>
<body>

<header>
  <div class="header-row">
    <div class="header-left">
      <h1>NDX 100 — MTF Ichimoku Scanner</h1>
      <p>1-Minute &nbsp;·&nbsp; 10-Minute &nbsp;·&nbsp; 1-Hour &nbsp;·&nbsp; 1-Day &nbsp;·&nbsp; Nasdaq 100 Universe</p>
    </div>
    <div class="scan-time">
      Scanned: <strong id="ts">—</strong><br>
      <span style="color:var(--muted);font-size:11px;">Re-run scanner.py to refresh data</span>
    </div>
  </div>
</header>

<div class="stats-bar">
  <div class="stats-inner" id="stats-bar"><!-- filled by JS --></div>
</div>

<div class="controls">
  <div class="controls-inner">
    <button class="filter-btn active" data-filter="all">All</button>
    <button class="filter-btn fbull" data-filter="Full Bull">Full Bull ▲▲▲▲</button>
    <button class="filter-btn mbull" data-filter="Mostly Bull">Mostly Bull</button>
    <button class="filter-btn mixed" data-filter="Mixed">Mixed</button>
    <button class="filter-btn mbear" data-filter="Mostly Bear">Mostly Bear</button>
    <button class="filter-btn fbear" data-filter="Full Bear">Full Bear ▼▼▼▼</button>
    <span class="sep">|</span>
    <input id="search" type="text" placeholder="Search ticker…" autocomplete="off">
    <span class="count-display"><strong id="visible-count">—</strong> stocks shown</span>
  </div>
</div>

<div class="table-wrap">
  <table id="main-table">
    <thead>
      <tr>
        <th data-col="ticker"  data-type="str">Ticker <span class="sort-icon">↕</span></th>
        <th data-col="price"   data-type="num">Price <span class="sort-icon">↕</span></th>
        <th data-col="chg"     data-type="num">Chg% <span class="sort-icon">↕</span></th>
        <th data-col="1m"      data-type="sig">1m <span class="sort-icon">↕</span></th>
        <th data-col="10m"     data-type="sig">10m <span class="sort-icon">↕</span></th>
        <th data-col="1h"      data-type="sig">1H <span class="sort-icon">↕</span></th>
        <th data-col="1d"      data-type="sig">1D <span class="sort-icon">↕</span></th>
        <th data-col="bulls"   data-type="num">MTF <span class="sort-icon">↕</span></th>
        <th data-col="align"   data-type="str">Alignment <span class="sort-icon">↕</span></th>
        <th data-col="sqn"     data-type="num">SQN(100) <span class="sort-icon">↕</span></th>
      </tr>
    </thead>
    <tbody id="tbody">
      <tr><td colspan="10" style="text-align:center;padding:40px;color:var(--muted);">Loading…</td></tr>
    </tbody>
  </table>
  <div class="no-results" id="no-results" style="display:none;">
    <p>No stocks match your filter.</p>
    <span style="font-size:12px;">Try a different filter or clear the search.</span>
  </div>
</div>

<script>
// ── DATA INJECTED BY PYTHON ───────────────────────────────────────────────
const DATA      = __DATA__;
const TIMESTAMP = "__TIMESTAMP__";
const STATS     = __STATS__;

// ── HELPERS ───────────────────────────────────────────────────────────────
const SIG_ORDER = {BULL:2, NEUTRAL:1, BEAR:0, null:-1};
const ALIGN_ORDER = {"Full Bull":5,"Mostly Bull":4,"Mixed":3,"Mostly Bear":2,"Full Bear":1,"N/A":0};

function sigClass(s){
  if(!s) return "na";
  return {BULL:"bull",BEAR:"bear",NEUTRAL:"neutral"}[s]||"na";
}

function sigIcon(s){
  return {BULL:"▲",BEAR:"▼",NEUTRAL:"–"}[s]||"?";
}

function tip(tf){
  if(!tf) return "No data";
  return [
    "Price:  "+tf.price_pos+" cloud",
    "TK:     "+tf.tk,
    "Cloud:  "+tf.cloud_color,
    "Chikou: "+tf.chikou,
    "Score:  "+tf.score+"/5",
    "Cloud:  "+tf.cloud_bottom+" – "+tf.cloud_top,
    "Tenkan: "+tf.tenkan,
    "Kijun:  "+tf.kijun,
  ].join("\n");
}

function badge(tf, label){
  if(!tf) return '<span class="badge na">— N/A</span>';
  const cls = sigClass(tf.signal);
  const ico = sigIcon(tf.signal);
  return '<span class="badge '+cls+'" data-tip="'+tip(tf).replace(/"/g,"&quot;")+'">'+ico+' '+tf.signal+'</span>';
}

function mtfCell(row){
  const tfs = [row["1m"], row["10m"], row["1h"], row["1d"]];
  const labels = ["1m","10m","1H","1D"];
  const dots = tfs.map((tf,i)=>{
    const cls = tf ? sigClass(tf.signal) : "na";
    const t = tf ? (labels[i]+": "+tf.signal+" ("+tf.score+"/5)") : labels[i]+": No data";
    return '<span class="dot '+cls+'" title="'+t+'"></span>';
  }).join("");
  const allBull = row.bulls === row.n_valid && row.n_valid > 0;
  const allBear = row.bears === row.n_valid && row.n_valid > 0;
  const scoreClass = allBull?"all-bull":allBear?"all-bear":"";
  const scoreStr = row.n_valid > 0 ? row.bulls+"/"+row.n_valid : "—";
  return '<div class="dots">'+dots+'<span class="mtf-score '+scoreClass+'">'+scoreStr+'</span></div>';
}

function alignBadge(a){
  const cls = a.replace(/ /g,".");
  return '<span class="align-badge '+cls+'">'+a+'</span>';
}

function chgCell(v){
  if(v==null) return '<span class="chg-cell fl">—</span>';
  const cls = v>0?"up":v<0?"dn":"fl";
  const sign = v>0?"+":"";
  return '<span class="chg-cell '+cls+'">'+sign+v.toFixed(2)+'%</span>';
}

function sqnBadge(s){
  if(!s) return '<span class="sqn-badge sqn-na">— N/A</span>';
  const sign = s.value > 0 ? "+" : "";
  return '<span class="sqn-badge '+s.css+'" title="SQN(100) = '+s.value+'">'+
    s.label+'<span class="sqn-val">('+sign+s.value+')</span></span>';
}

// ── RENDER ────────────────────────────────────────────────────────────────
let currentData   = [...DATA];
let sortCol       = "bulls";
let sortDir       = -1;   // -1 = desc, 1 = asc
let activeFilter  = "all";
let searchText    = "";

function getSortVal(row, col){
  switch(col){
    case "ticker":  return row.ticker;
    case "price":   return row.price ?? -Infinity;
    case "chg":     return row.change_pct ?? -Infinity;
    case "1m":      return SIG_ORDER[row["1m"]?.signal ?? null];
    case "10m":     return SIG_ORDER[row["10m"]?.signal ?? null];
    case "1h":      return SIG_ORDER[row["1h"]?.signal ?? null];
    case "1d":      return SIG_ORDER[row["1d"]?.signal ?? null];
    case "bulls":   return row.bulls ?? -1;
    case "align":   return ALIGN_ORDER[row.alignment ?? "N/A"];
    case "sqn":     return row.sqn?.value ?? -Infinity;
    default:        return 0;
  }
}

function sortData(){
  currentData.sort((a,b)=>{
    const va = getSortVal(a, sortCol);
    const vb = getSortVal(b, sortCol);
    if(typeof va === "string") return sortDir * va.localeCompare(vb);
    return sortDir * (vb - va);
  });
}

function matchFilter(row){
  if(activeFilter === "all") return true;
  return row.alignment === activeFilter;
}

function matchSearch(row){
  if(!searchText) return true;
  return row.ticker.toLowerCase().includes(searchText);
}

function renderTable(){
  sortData();
  const tbody = document.getElementById("tbody");
  const frag  = document.createDocumentFragment();
  let visible = 0;

  for(const row of currentData){
    const show = matchFilter(row) && matchSearch(row);
    const tr = document.createElement("tr");
    if(!show) tr.classList.add("hidden");
    else visible++;

    tr.innerHTML =
      '<td class="ticker-cell"><a href="https://finance.yahoo.com/quote/'+row.ticker+'" target="_blank">'+row.ticker+'</a></td>'+
      '<td class="price-cell">'+(row.price != null ? "$"+row.price.toFixed(2) : "—")+'</td>'+
      '<td>'+chgCell(row.change_pct)+'</td>'+
      '<td>'+badge(row["1m"])+'</td>'+
      '<td>'+badge(row["10m"])+'</td>'+
      '<td>'+badge(row["1h"])+'</td>'+
      '<td>'+badge(row["1d"])+'</td>'+
      '<td>'+mtfCell(row)+'</td>'+
      '<td>'+alignBadge(row.alignment||"N/A")+'</td>'+
      '<td>'+sqnBadge(row.sqn)+'</td>';

    frag.appendChild(tr);
  }

  tbody.innerHTML = "";
  tbody.appendChild(frag);

  document.getElementById("visible-count").textContent = visible;
  document.getElementById("no-results").style.display = visible === 0 ? "block" : "none";
  document.getElementById("main-table").style.display  = visible === 0 ? "none"  : "";
}

// ── STATS BAR ─────────────────────────────────────────────────────────────
function renderStats(){
  const bar = document.getElementById("stats-bar");
  const pills = [
    {cls:"total", num:STATS.total,      label:"Scanned"},
    {cls:"fbull", num:STATS.full_bull,  label:"Full Bull"},
    {cls:"mbull", num:STATS.mostly_bull,label:"Mostly Bull"},
    {cls:"mixed", num:STATS.mixed,      label:"Mixed"},
    {cls:"mbear", num:STATS.mostly_bear,label:"Mostly Bear"},
    {cls:"fbear", num:STATS.full_bear,  label:"Full Bear"},
    {cls:"na",    num:STATS.na,         label:"No Data"},
  ];
  bar.innerHTML = pills.map(p=>
    '<div class="stat-pill '+p.cls+'"><span class="num">'+p.num+'</span><span>'+p.label+'</span></div>'
  ).join("");
}

// ── EVENTS ────────────────────────────────────────────────────────────────
document.querySelectorAll("thead th").forEach(th=>{
  th.addEventListener("click",()=>{
    const col = th.dataset.col;
    if(sortCol === col){ sortDir *= -1; }
    else { sortCol = col; sortDir = -1; }
    document.querySelectorAll("thead th").forEach(h=>{
      h.classList.remove("sorted");
      h.querySelector(".sort-icon").textContent = "↕";
    });
    th.classList.add("sorted");
    th.querySelector(".sort-icon").textContent = sortDir === -1 ? "▼" : "▲";
    renderTable();
  });
});

document.querySelectorAll(".filter-btn").forEach(btn=>{
  btn.addEventListener("click",()=>{
    document.querySelectorAll(".filter-btn").forEach(b=>b.classList.remove("active"));
    btn.classList.add("active");
    activeFilter = btn.dataset.filter;
    renderTable();
  });
});

document.getElementById("search").addEventListener("input", e=>{
  searchText = e.target.value.trim().toLowerCase();
  renderTable();
});

// ── INIT ──────────────────────────────────────────────────────────────────
document.getElementById("ts").textContent = TIMESTAMP;
renderStats();

// Default: sort by bulls desc
sortCol = "bulls"; sortDir = -1;
const bullHeader = document.querySelector('th[data-col="bulls"]');
if(bullHeader){
  bullHeader.classList.add("sorted");
  bullHeader.querySelector(".sort-icon").textContent = "▼";
}
renderTable();
</script>
</body>
</html>
"""


def generate_html(results: list, timestamp: str) -> str:
    stats = {
        "total":      len(results),
        "full_bull":  sum(1 for r in results if r["alignment"] == "Full Bull"),
        "mostly_bull":sum(1 for r in results if r["alignment"] == "Mostly Bull"),
        "mixed":      sum(1 for r in results if r["alignment"] == "Mixed"),
        "mostly_bear":sum(1 for r in results if r["alignment"] == "Mostly Bear"),
        "full_bear":  sum(1 for r in results if r["alignment"] == "Full Bear"),
        "na":         sum(1 for r in results if r["alignment"] == "N/A"),
    }

    data_json  = json.dumps(results, default=str)
    stats_json = json.dumps(stats)

    return (HTML_TEMPLATE
            .replace("__DATA__",      data_json)
            .replace("__TIMESTAMP__", timestamp)
            .replace("__STATS__",     stats_json))


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  NDX 100 MTF Ichimoku Scanner")
    print("=" * 60)

    results   = scan(TICKERS)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html      = generate_html(results, timestamp)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ichimoku_scan.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("\n" + "=" * 60)
    print(f"  Done!  {len(results)} stocks scanned")
    print(f"  Output: {out_path}")
    print("=" * 60)

    # Auto-open in default browser
    try:
        os.startfile(out_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
