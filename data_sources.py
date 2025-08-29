def get_intraday_1m_yf(ticker, period="2d"):  # keep 2d for buffer
    try:
        df = yf.download(
            ticker, period=period, interval="1m", auto_adjust=False, progress=False, prepost=True
        )
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df.rename(columns=str.title).dropna()
    except Exception:
        return pd.DataFrame()

def sector_breadth_yf(tickers):
    try:
        data = yf.download(tickers, period="1d", interval="1m", auto_adjust=False, progress=False, group_by='ticker')
    except Exception:
        data = None
    rows = []
    green = 0; total = 0
    for t in tickers:
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
        last = float(df["Close"].iloc[-1])
        vwap = float(compute_vwap_from_df(df).iloc[-1])
        ok = not (math.isnan(last) or math.isnan(vwap))
        is_green = (ok and last > vwap)
        rows.append((t, last if ok else float("nan"), vwap if ok else float("nan"), is_green))
        if ok:
            total += 1; green += 1 if is_green else 0
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
