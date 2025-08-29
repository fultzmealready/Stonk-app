import streamlit as st
import os, requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _get_secret(name: str, default: str = "") -> str:
    # exact key
    if name in st.secrets:
        return st.secrets.get(name, default)
    # fallbacks
    for alt in ("TAAPI_SECRETS", "TAAPI_KEY", "TAAPI_API_KEY"):
        if alt in st.secrets:
            return st.secrets.get(alt, default)
    # nested block [taapi]
    tb = st.secrets.get("taapi", {})
    if isinstance(tb, dict):
        for k in ("secret","api_key","key"):
            if k in tb:
                return tb[k]
    # env
    return os.getenv(name, default)

TAAPI_SECRET = _get_secret("TAAPI_SECRET")
TAAPI_URL = os.getenv("TAAPI_URL", "https://api.taapi.io/bulk")

def make_session():
    s = requests.Session()
    retries = Retry(
        total=5, connect=5, read=5, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False, raise_on_redirect=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"Content-Type": "application/json", "Accept": "application/json"})
    return s

SESSION = make_session()

def make_stock_construct(symbol: str, interval: str, indicators: list[dict]) -> dict:
    """
    TAAPI bulk 'construct' for stocks requires:
      { "type":"stocks", "symbol":"SPY", "interval":"1m", "indicators":[...] }
    """
    return {
        "type": "stocks",
        "symbol": symbol.upper(),
        "interval": interval,
        "indicators": indicators,
    }

def taapi_bulk(constructs, timeout_sec=45):
    """POST /bulk with one or more constructs. Return a dict[id -> result]."""
    try:
        payload = {"secret": TAAPI_SECRET, "construct": constructs if len(constructs) != 1 else constructs[0]}
        r = SESSION.post(TAAPI_URL, json=payload, timeout=(5, timeout_sec))
        r.raise_for_status()
        j = r.json()
        data = j.get("data", [])
        out = {}
        for it in data:
            _id = it.get("id")
            res = it.get("result", {})
            if _id:
                out[_id] = res
        if not out and isinstance(j, dict):
            out["__raw__"] = j
        return out
    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = r.text[:800]
        except Exception:
            pass
        st.error(f"TAAPI bulk call failed: HTTP {getattr(r,'status_code','?')} — {e}\n\nResponse: `{body}`")
        return {}
    except requests.exceptions.RequestException as e:
        st.error(f"Network/timeout error calling TAAPI: {e}")
        return {}
    except Exception as e:
        st.error(f"Unexpected TAAPI error: {e}")
        return {}

def get_taapi_candles(symbol, interval="1m", limit=300, timeout_sec=45):
    """Fetch OHLCV candles via TAAPI 'candles' indicator for STOCKS/ETFs."""
    try:
        construct = make_stock_construct(
            symbol, interval,
            [{"indicator": "candles", "backtrack": int(limit)}],
        )
        payload = {"secret": TAAPI_SECRET, "construct": construct}
        r = SESSION.post(TAAPI_URL, json=payload, timeout=(5, timeout_sec))
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return pd.DataFrame()
        item = data[0].get("result", {}) if isinstance(data[0], dict) else {}
        values = item.get("values") or item.get("result") or item.get("data")
        if not isinstance(values, list) or not values:
            return pd.DataFrame()
        records = []
        for c in values:
            ts = c.get("timestamp") or c.get("time") or c.get("t")
            o = c.get("open") or c.get("o")
            h = c.get("high") or c.get("h")
            l = c.get("low")  or c.get("l")
            cl= c.get("close")or c.get("c")
            v = c.get("volume") or c.get("v") or 0.0
            if ts is None or o is None or h is None or l is None or cl is None:
                continue
            ts = int(ts)
            idx = pd.to_datetime(ts, unit=("ms" if ts > 10_000_000_000 else "s"))
            records.append([idx, float(o), float(h), float(l), float(cl), float(v)])
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records, columns=["Datetime","Open","High","Low","Close","Volume"])\
               .set_index("Datetime").sort_index()
        return df
    except Exception:
        return pd.DataFrame()
