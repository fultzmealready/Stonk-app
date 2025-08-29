# sheets_store.py
# Read/write the trade log to Google Sheets using a service account.
# Falls back to local CSV if Sheets isn’t configured.

import os
import uuid
import pandas as pd
import streamlit as st

# Optional: reuse your CSV as a fallback
CSV_PATH = "0dte_trade_log.csv"

# ---- Secrets required ----
# st.secrets["gcp_service_account"]  -> full service account JSON object
# st.secrets["SHEETS_DOC_NAME"]      -> Google Sheet doc name (or provide SHEETS_DOC_KEY)
# st.secrets["SHEETS_TAB_NAME"]      -> Worksheet/tab name (default: "trades")
#
# IMPORTANT: Share your Google Sheet with the service account email (xxxxx@xxxxx.iam.gserviceaccount.com)

def _get_client():
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        sa_info = st.secrets.get("gcp_service_account", None)
        if not sa_info:
            return None, "Missing gcp_service_account in secrets"

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        client = gspread.authorize(creds)
        return client, None
    except Exception as e:
        return None, f"Auth error: {e}"

def _open_sheet():
    """
    Returns (worksheet, error_message). Creates tab/headers if needed.
    """
    client, err = _get_client()
    if err or client is None:
        return None, err or "No client"

    doc_key  = st.secrets.get("SHEETS_DOC_KEY", "").strip()
    doc_name = st.secrets.get("SHEETS_DOC_NAME", "").strip()
    tab_name = st.secrets.get("SHEETS_TAB_NAME", "trades").strip() or "trades"

    try:
        if doc_key:
            sh = client.open_by_key(doc_key)
        elif doc_name:
            sh = client.open(doc_name)
        else:
            return None, "Provide SHEETS_DOC_KEY or SHEETS_DOC_NAME in secrets."

        try:
            ws = sh.worksheet(tab_name)
        except Exception:
            ws = sh.add_worksheet(title=tab_name, rows=2000, cols=20)
            _ensure_headers(ws)

        # Ensure headers exist
        _ensure_headers(ws)
        return ws, None
    except Exception as e:
        return None, f"Open sheet error: {e}"

def _ensure_headers(ws):
    headers = ["id","time","symbol","side","strike","qty","entry","exit","p_l_pct","notes"]
    try:
        current = ws.row_values(1)
        if current != headers:
            ws.update("A1", [headers])
    except Exception:
        pass

def _rows_to_df(rows):
    headers = ["id","time","symbol","side","strike","qty","entry","exit","p_l_pct","notes"]
    if not rows:
        return pd.DataFrame(columns=headers)
    df = pd.DataFrame(rows, columns=headers)
    # Type cleanup
    for c in ["strike","qty","entry","exit","p_l_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------- Public API ----------

def load_trades() -> pd.DataFrame:
    """
    Try Google Sheets; fallback to CSV.
    """
    ws, err = _open_sheet()
    if ws is None:
        # Fallback to CSV
        if os.path.exists(CSV_PATH):
            return pd.read_csv(CSV_PATH)
        else:
            return pd.DataFrame(columns=["id","time","symbol","side","strike","qty","entry","exit","p_l_pct","notes"])

    try:
        values = ws.get_all_values()
        if len(values) <= 1:
            return pd.DataFrame(columns=["id","time","symbol","side","strike","qty","entry","exit","p_l_pct","notes"])
        rows = values[1:]  # skip header
        return _rows_to_df(rows)
    except Exception:
        # Sheets read failed; fallback
        if os.path.exists(CSV_PATH):
            return pd.read_csv(CSV_PATH)
        return pd.DataFrame(columns=["id","time","symbol","side","strike","qty","entry","exit","p_l_pct","notes"])

def save_trades(df: pd.DataFrame) -> None:
    """
    Try writing to Google Sheets; also mirror to CSV locally.
    """
    # Always mirror to CSV
    try:
        df.to_csv(CSV_PATH, index=False)
    except Exception:
        pass

    ws, err = _open_sheet()
    if ws is None:
        return  # no Sheets available; CSV already saved

    # Write whole sheet (cheap for small logs; simple & reliable)
    cols = ["id","time","symbol","side","strike","qty","entry","exit","p_l_pct","notes"]
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[cols]
    values = [cols] + out.astype(str).fillna("").values.tolist()
    ws.clear()
    ws.update("A1", values)

def append_trade_row(time_str: str, symbol: str, side: str, strike: float,
                     qty: int, entry: float, exitp: float, notes: str) -> pd.DataFrame:
    """
    Append a single row using the in-memory df model then save.
    Returns the updated DataFrame.
    """
    df = load_trades()
    row = {
        "id": uuid.uuid4().hex,
        "time": time_str,
        "symbol": symbol,
        "side": side,
        "strike": strike,
        "qty": qty,
        "entry": entry,
        "exit": exitp,
        "p_l_pct": ((exitp - entry)/entry*100.0) if entry and exitp else float("nan"),
        "notes": notes,
    }
    new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_trades(new_df)
    return new_df

def delete_by_ids(ids: list[str]) -> pd.DataFrame:
    df = load_trades()
    if "id" not in df.columns or df.empty:
        return df
    new_df = df[~df["id"].astype(str).isin([str(i) for i in ids])].copy()
    save_trades(new_df)
    return new_df
