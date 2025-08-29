import math
import pandas as pd
import yfinance as yf
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


def sector_breadth_yf(tickers: list[str]) -> tuple[list[tuple], float]:
    """
    For each sector, compute last and VWAP, and % of sectors above VWAP.
    """
    try:
        data = yf.download(
            tickers, period="1d", interval="1m",
            auto_adjust=False, progress=False, group_by="ticker", prepost=False
        )
    except Exception:
        data = None

    rows = []
    green = 0
    total = 0

    for t in tickers:
        # get a 1m df for this ticker
        try:
            df = data[t] if data is not None else get_intraday_1m_yf(t)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.rename(columns=str.title).dropna()
        except Exception:
            df = pd.DataFrame()

        if df is None or df.empty:
            rows.append((t, float("nan"), float("nan"), False))
            continue

        # robust scalars (avoid FutureWarning)
        last_scalar = pd.to_numeric(df["Close"].iloc[-1], errors="coerce")
        vwap_series = compute_vwap_from_df(df)
        vwap_scalar = pd.to_numeric(vwap_series.iloc[-1], errors="coerce")
        last = float(last_scalar) if pd.notna(last_scalar) else float("nan")
        vwap = float(vwap_scalar) if pd.notna(vwap_scalar) else float("nan")

        ok = not (math.isnan(last) or math.isnan(vwap))
        is_green = (ok and last > vwap)
        rows.append((t, last if ok else float("nan"), vwap if ok else float("nan"), is_green))
        if ok:
            total += 1
            if is_green:
                green += 1

    breadth = (green / total * 100.0) if total else float("nan")
    return rows, breadth

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
