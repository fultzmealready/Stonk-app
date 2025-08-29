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

def _get_ws():
    """Return (worksheet, None) or (None, reason). Requires secrets & sharing."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception as e:
        return None, f"gspread/google-auth not importable: {e}"

    sa_info = st.secrets.get("gcp_service_account")
    if not sa_info:
        return None, "Missing st.secrets['gcp_service_account']"

    doc_name = st.secrets.get("SHEETS_DOC_NAME")
    doc_key  = st.secrets.get("SHEETS_DOC_KEY")
    tab_name = st.secrets.get("SHEETS_TAB_NAME", "trades")

    if not (doc_name or doc_key):
        return None, "Missing SHEETS_DOC_NAME or SHEETS_DOC_KEY in secrets"

    try:
        creds = Credentials.from_service_account_info(
            sa_info,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(doc_key) if doc_key else gc.open(doc_name)
        try:
            ws = sh.worksheet(tab_name)
        except Exception:
            ws = sh.add_worksheet(title=tab_name, rows=1000, cols=len(HEADERS))
            ws.append_row(HEADERS, value_input_option="USER_ENTERED")
        # Ensure header exists / is correct
        vals = ws.get_all_values()
        if not vals:
            ws.append_row(HEADERS, value_input_option="USER_ENTERED")
        else:
            header = vals[0]
            if header != HEADERS:
                # Reset header to expected order (simple, safe approach)
                ws.clear()
                ws.append_row(HEADERS, value_input_option="USER_ENTERED")
        return ws, None
    except Exception as e:
        return None, f"Sheets open/authorize error: {e}"

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
    ws, reason = _get_ws()
    if ws is not None:
        try:
            vals = ws.get_all_values()
            if not vals or len(vals) == 1:
                return pd.DataFrame(columns=HEADERS)
            df = pd.DataFrame(vals[1:], columns=vals[0])  # skip header
            # coerce numeric columns
            for c in ("strike","qty","entry","exit","p_l_pct"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception as e:
            st.warning(f"Sheets read failed, falling back to CSV: {e}")
    else:
        if reason:
            st.info(f"Using CSV fallback — {reason}")

    # CSV fallback
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            for c in ("strike","qty","entry","exit","p_l_pct"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    return pd.DataFrame(columns=HEADERS)

def save_trades(df: pd.DataFrame) -> None:
    """Overwrite entire store with df (Sheets if possible; else CSV)."""
    ws, reason = _get_ws()
    # Reorder columns to HEADERS (keep extras at end if present)
    cols = [c for c in HEADERS if c in df.columns] + [c for c in df.columns if c not in HEADERS]
    df = df[cols].copy()

    if ws is not None:
        try:
            # Clear and write header + values
            ws.clear()
            ws.append_row(HEADERS, value_input_option="USER_ENTERED")
            if not df.empty:
                rows = df.reindex(columns=HEADERS, fill_value="").astype(str).values.tolist()
                # gspread prefers chunked updates for large frames; this is fine for small logs
                ws.add_rows(max(0, len(rows) - 1000))
                ws.append_rows(rows, value_input_option="USER_ENTERED")
            st.caption("Storage: Google Sheets")
            return
        except Exception as e:
            st.warning(f"Sheets write failed, falling back to CSV: {e}")
    else:
        if reason:
            st.info(f"Using CSV fallback — {reason}")

    # CSV fallback
    try:
        df.to_csv(CSV_PATH, index=False)
        st.caption("Storage: CSV (fallback)")
    except Exception as e:
        st.error(f"Failed to write CSV: {e}")

def append_trade_row(time_str: str, symbol: str, side: str, strike: float,
                     qty: int, entry: float, exitp: float, notes: str) -> pd.DataFrame:
    """Append one row; prefer Sheets append_row when available."""
    ws, reason = _get_ws()
    row = {
        "id": uuid.uuid4().hex,
        "time": time_str,
        "symbol": symbol,
        "side": side,
        "strike": strike,
        "qty": qty,
        "entry": entry,
        "exit": exitp,
        "p_l_pct": ((exitp - entry)/entry*100.0) if (entry and exitp) else float("nan"),
        "notes": notes,
    }

    if ws is not None:
        try:
            ws.append_row([str(row.get(k, "")) for k in HEADERS], value_input_option="USER_ENTERED")
            # Re-read to return the updated df
            return load_trades()
        except Exception as e:
            st.warning(f"Sheets append failed, will try CSV: {e}")

    # CSV fallback path
    df = load_trades()
    new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_trades(new_df)
    return new_df

def delete_by_ids(ids: list[str]) -> pd.DataFrame:
    """Filter out ids and persist (Sheets if possible; else CSV)."""
    df = load_trades()
    if df.empty or "id" not in df.columns:
        return df
    ids = {str(i) for i in ids}
    new_df = df[~df["id"].astype(str).isin(ids)].copy()
    save_trades(new_df)
    return new_df
