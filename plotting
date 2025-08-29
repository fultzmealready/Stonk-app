def plot_with_orb_em(ticker, df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))

    # VWAP
    vwap_series = compute_vwap_from_df(df)
    fig.add_trace(go.Scatter(x=df.index, y=vwap_series, mode="lines", name="VWAP"))

    # Regular Trading Hours shading
    for day in df.index.normalize().unique():
        rth_start = day + pd.Timedelta(hours=9, minutes=30)
        rth_end   = day + pd.Timedelta(hours=16, minutes=0)
        fig.add_vrect(
            x0=rth_start, x1=rth_end,
            fillcolor="LightGreen", opacity=0.1, layer="below", line_width=0,
            annotation_text="RTH", annotation_position="top left"
        )
        # Before 9:30 and after 16:00 will show up as unshaded (off-hours)

    fig.update_layout(
        title=f"{ticker} — 24h",
        height=420,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig
