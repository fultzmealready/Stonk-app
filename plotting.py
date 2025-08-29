import math, datetime as dt
import plotly.graph_objs as go
import pandas as pd
from indicators import compute_opening_range, compute_vwap_from_df

def plot_with_orb_em(ticker: str, df: pd.DataFrame, orb_minutes: int = 15):
    fig = go.Figure()

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))

    # VWAP
    vwap_series = compute_vwap_from_df(df)
    fig.add_trace(go.Scatter(x=df.index, y=vwap_series, mode="lines", name="VWAP"))

    # ORB
    s, e, orh, orl = compute_opening_range(df, minutes=orb_minutes)
    if s is not None and e is not None and not (math.isnan(orh) or math.isnan(orl)):
        fig.add_hrect(y0=orl, y1=orh, x0=s, x1=e, opacity=0.15, line_width=0, fillcolor="LightSkyBlue")
        fig.add_hline(y=orh, line_dash="dot", opacity=0.5)
        fig.add_hline(y=orl, line_dash="dot", opacity=0.5)

    # === Remove dead space: skip 00:00 → 08:00 each day ===
    # (Matches what you see on the chart; if you later localize to ET, you can switch this to [20, 4].)
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),          # hide weekends
            dict(bounds=[20, 4], pattern="hour"),  # hide midnight → 08:00 gap you observed
        ]
    )

    # === EOD line at the break (midnight) ===
    try:
        # Midnight of the most recent day in the df's timezone
        last_midnight = df.index[-1].normalize()
        if df.index.min() <= last_midnight <= df.index.max():
            fig.add_vline(
                x=last_midnight,
                line_dash="dot",
                line_color="red",
                annotation_text="EOD",
                annotation_position="top"
            )
    except Exception:
        pass

    # Tighten x range to data window to avoid right-side whitespace
    fig.update_xaxes(range=[df.index.min(), df.index.max()])

    fig.update_layout(
        title=f"{ticker} — 24h",
        height=420,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig
