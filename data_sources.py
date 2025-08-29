import math, time, random
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from indicators import compute_vwap_from_df


def get_intraday_1m_yf(ticker: str, period: str = "2d", include_ext_hours: bool = True) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1m",
            auto_adjust=False,
            progress=False,
            prepost=include_ext_hours,   # <— extended hours (pre/post)
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df.rename(columns=str.title).dropna()
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York") 
    except Exception:
        return pd.DataFrame()

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.title)
    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    return df[cols].dropna(how="all")

def _to_float(x):
    try:
        if isinstance(x, pd.Series):
            x = x.iloc[-1]
        return float(x)
    except Exception:
        return float("nan")

def _fetch_1m_with_retry(ticker: str, attempts: int = 3, period: str = "2d") -> pd.DataFrame:
    """
    yfinance 1m can be flaky. Try up to `attempts` times with jittered backoff.
    Includes pre/post so we get premkt.
    """
    last_err = None
    for i in range(attempts):
        try:
            df = yf.download(
                ticker, period=period, interval="1m",
                auto_adjust=False, prepost=True, progress=False
            )
            df = _prep(df)
            if not df.empty:
                return df
        except Exception as e:
            last_err = e
        # backoff with a tiny jitter (don’t hammer)
        time.sleep(0.6 * (2 ** i) + random.random() * 0.3)
    # if all attempts failed, return empty frame
    return pd.DataFrame()

def sector_breadth_yf(tickers):
    """
    Returns:
      rows = [(ticker, last, vwap, is_green), ...]
      breadth_pct = % of non-NaN rows where last > vwap
    Strategy:
      - per-ticker downloads with retry
      - small thread pool (avoid yfinance rate limits)
      - daily-reset VWAP with per-day forward-fill
    """
    rows = []
    green = 0
    total = 0

    # modest parallelism; too many threads => throttling / partial data
    max_workers = min(4, max(1, len(tickers)))
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_1m_with_retry, t): t for t in tickers}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                df = fut.result()
            except Exception:
                df = pd.DataFrame()
            results[t] = df

    for t in tickers:
        df = results.get(t, pd.DataFrame())
        if df is None or df.empty or "Close" not in df.columns:
            rows.append((t, float("nan"), float("nan"), False))
            continue

        # latest price as scalar
        last = _to_float(df["Close"].iloc[-1])

        # session-reset VWAP (not RTH-only), fill per-day to bridge zero-volume minutes
        vwap_s = compute_vwap_from_df(df, reset="daily", rth_only=False)
        if vwap_s is None or vwap_s.empty:
            vwap = float("nan")
        else:
            try:
                vwap_s = vwap_s.groupby(vwap_s.index.date).ffill()
            except Exception:
                vwap_s = vwap_s.ffill()
            vwap = _to_float(vwap_s.iloc[-1])

        ok = not (math.isnan(last) or math.isnan(vwap))
        is_green = ok and last > vwap
        rows.append((t, last if ok else float("nan"), vwap if ok else float("nan"), is_green))
        if ok:
            total += 1
            if is_green:
                green += 1

    breadth_pct = (green / total * 100.0) if total else float("nan")
    return rows, breadth_pct


def get_futures_quote(sym):
    try:
        day = yf.download(sym, period="2d", interval="1d", auto_adjust=False, progress=False)
        last_close = float(day["Close"].iloc[-1])
        prev_close = float(day["Close"].iloc[-2]) if len(day)>=2 else last_close
        intr = yf.download(sym, period="1d", interval="1m", auto_adjust=False, progress=False)
        last = float(intr["Close"].iloc[-1]) if not intr.empty else last_close
        chg = ((last - prev_close) / prev_close * 100.0) if prev_close else float("nan")
        return last, chg
    except Exception:
        return float("nan"), float("nan")
