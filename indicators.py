import math
import numpy as np
import pandas as pd

def first_numeric(d):
    if isinstance(d, dict):
        for k in ("value","valueMACD","valueMACDSignal","valueMACDHist"):
            if k in d and isinstance(d[k], (int,float)): return float(d[k])
        for v in d.values():
            if isinstance(v,(int,float)): return float(v)
            if isinstance(v,dict):
                for vv in v.values():
                    if isinstance(vv,(int,float)): return float(vv)
    return float("nan")

def get_value_by_id(results: dict, id_contains: str, default=float("nan")):
    for id_str, result in results.items():
        if id_contains in id_str:
            num = first_numeric(result)
            if isinstance(num,(int,float)) and not math.isnan(num):
                return float(num)
    return default

# indicators.py
import numpy as np
import pandas as pd

def compute_vwap_from_df(
    df: pd.DataFrame,
    *,
    tz: str = "America/New_York",
    reset: str = "daily",          # "daily" or "none"
    rth_only: bool = False,        # leave False if you don't want RTH-only
) -> pd.Series:
    """Session-reset VWAP with zero-volume ignored; index converted to ET."""
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    # local import so the module doesn't crash if ZoneInfo wasn't imported at top
    from zoneinfo import ZoneInfo

    d = df.copy()

    # normalize column names
    want = {"Open","High","Low","Close","Volume"}
    if not want.issubset(set(map(str.title, d.columns)) | set(d.columns)):
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] for c in d.columns]
    d.rename(columns={c: c.title() for c in d.columns}, inplace=True)

    # ensure ET, sorted
    idx = pd.to_datetime(d.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(ZoneInfo(tz))
    d = d.set_index(idx).sort_index()

    # optional RTH mask
    if rth_only:
        mask = d.index.indexer_between_time("09:30", "16:00")
        keep = pd.Series(False, index=d.index)
        keep.iloc[mask] = True
        d = d.where(keep)

    tp  = (d["High"] + d["Low"] + d["Close"]) / 3.0
    vol = d["Volume"].replace(0, np.nan)

    if reset == "none":
        return (tp * vol).cumsum() / vol.cumsum()

    # reset each ET calendar day
    day = d.index.date
    cum_tpvol = (tp * vol).groupby(day).cumsum()
    cum_vol   = vol.groupby(day).cumsum()
    return cum_tpvol / cum_vol

def first5_momentum(df):
    if df is None or df.empty: return None
    df5 = df.resample("5min").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
    if len(df5)==0: return None
    c = df5.iloc[0]
    rng = max(c["High"]-c["Low"], 1e-6)
    if c["Close"] > c["Open"] and c["Close"] >= (c["High"] - 0.2*rng): return True
    if c["Close"] < c["Open"] and c["Close"] <= (c["Low"] + 0.2*rng): return False
    return None

def round_magnets(price, step=5.0):
    if price is None or (isinstance(price, float) and math.isnan(price)): return []
    base = round(price / step) * step
    return [round(base - 2*step, 2), round(base - step, 2), round(base, 2), round(base + step, 2), round(base + 2*step, 2)]

def choose_strike(last, decision, step=1.0):
    mags = round_magnets(last, 5.0)
    if not mags or math.isnan(last): return None
    if "CALL" in decision:
        above = sorted([m for m in mags if m >= last])
        target = above[0] if above else mags[-1]
    elif "PUT" in decision:
        below = sorted([m for m in mags if m <= last], reverse=True)
        target = below[0] if below else mags[0]
    else:
        return None
    return round(target / step) * step

def compute_bias(spy_last, qqq_last, spy_vwap, qqq_vwap, breadth_pct, f5_spy, f5_qqq, es_green, nq_green):
    score = 0.0; parts = {}
    for tag, g in [("ES", es_green), ("NQ", nq_green)]:
        if g is not None: s = 1.0 if g else -1.0; score += s; parts[tag]=s
    for tag, g in [("SPY 1st5", f5_spy), ("QQQ 1st5", f5_qqq)]:
        if g is not None: s = 1.0 if g else -1.0; score += s; parts[tag]=s
    for tag, g in [("SPY>VWAP", (spy_last>spy_vwap) if not (math.isnan(spy_last) or math.isnan(spy_vwap)) else None),
                   ("QQQ>VWAP", (qqq_last>qqq_vwap) if not (math.isnan(qqq_last) or math.isnan(qqq_vwap)) else None)]:
        if g is not None: s = 1.0 if g else -1.0; score += s; parts[tag]=s
    if not math.isnan(breadth_pct):
        s = 1.0 if breadth_pct>=60 else (-1.0 if breadth_pct<=40 else 0.0)
        score += s; parts["Breadth"] = s
    decision = "CALL bias" if score>=2 else ("PUT bias" if score<=-2 else "NEUTRAL / WAIT")
    return score, decision, parts

def compute_opening_range(df, minutes=15):
    if df is None or df.empty: return None, None, float("nan"), float("nan")
    day = df.index[-1].date()
    todays = df[df.index.date == day]
    if todays.empty: return None, None, float("nan"), float("nan")
    start_ts = todays.index[0]
    end_ts = start_ts + pd.Timedelta(minutes=minutes)
    window = todays[(todays.index >= start_ts) & (todays.index <= end_ts)]
    if window.empty: return start_ts, end_ts, float("nan"), float("nan")
    or_high = float(window["High"].max()); or_low  = float(window["Low"].min())
    return start_ts, end_ts, or_high, or_low

# ====== Confidence helpers ======
def taapi_confidence_bits(core: dict, prefix: str):
    rsi = get_value_by_id(core, prefix + "rsi_14_")
    macd_hist = float("nan")
    for k, v in core.items():
        if prefix + "macd_" in k and isinstance(v, dict):
            macd_hist = v.get("valueMACDHist", float("nan")); break
    ema9  = get_value_by_id(core, prefix + "ema_9_")
    ema20 = get_value_by_id(core, prefix + "ema_20_")
    bits = {"rsi_up": (rsi>55) if not math.isnan(rsi) else None,
            "macd_green": (macd_hist>0) if not math.isnan(macd_hist) else None,
            "ema_stack": (ema9>ema20) if not (math.isnan(ema9) or math.isnan(ema20)) else None}
    return bits

def confidence_label(bits, decision):
    score = 0; used = []
    for key, name in [("rsi_up","RSI>55"),("macd_green","MACD hist>0"),("ema_stack","EMA9>20")]:
        val = bits.get(key, None)
        if val is True: score += 1; used.append(name)
        elif val is False: score -= 1; used.append("-"+name)
    if "WAIT" in decision: return "Low", score, used
    if score >= 2: return "🔥 High", score, used
    if score == 1: return "Medium", score, used
    if score <= -1: return "Low", score, used
    return "Medium", score, used

def last_n_dirs(df, n=3):
    if df is None or df.empty: return []
    closes = df["Close"].tail(n).tolist(); opens = df["Open"].tail(n).tolist()
    out = []
    for o, c in zip(opens, closes):
        if c > o: out.append(True)
        elif c < o: out.append(False)
        else: out.append(None)
    return out

def candle_emojis(df, n=3):
    if df is None or df.empty: return "—"
    closes = df["Close"].tail(n).tolist(); opens = df["Open"].tail(n).tolist()
    out = []
    for o, c in zip(opens, closes):
        if c > o: out.append("🟩")
        elif c < o: out.append("🟥")
        else: out.append("➖")
    return " ".join(out)

def resample_5m(df):
    if df is None or df.empty: return pd.DataFrame()
    return df.resample("5min").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
