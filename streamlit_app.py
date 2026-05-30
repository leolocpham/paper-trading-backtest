"""
streamlit_app.py  —  Paper Trading Backtest Engine
===================================================
Run locally:  streamlit run streamlit_app.py
Deploy:       push to GitHub, connect at share.streamlit.io
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

from paper_trading_engine import generate_synthetic_ohlcv, PaperTradingEngine

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Paper Trading Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Paper Trading Backtest Engine")
st.caption(
    "Four independent strategies from the CFI Guide to Trading — "
    "fully simulated with no real money at risk."
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    # Data source
    st.subheader("Data Source")
    data_source = st.radio(
        "source",
        ["Synthetic (GBM)", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "OHLCV CSV (columns: open, high, low, close, volume)",
            type=["csv"],
        )
        st.caption("Column names must be lowercase.")

    st.divider()

    # Simulation parameters
    st.subheader("Simulation Parameters")
    starting_capital = st.number_input(
        "Starting Capital ($)",
        min_value=1_000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
        format="%d",
    )
    risk_pct = st.slider(
        "Risk per Trade (%)",
        min_value=0.25,
        max_value=5.0,
        value=1.0,
        step=0.25,
        format="%.2f",
    )

    # Synthetic data settings (only shown for GBM source)
    if data_source == "Synthetic (GBM)":
        st.divider()
        st.subheader("Synthetic Data Settings")
        n_bars   = st.slider("Number of Bars",     300, 3_000, 1_200, 100)
        seed     = int(st.number_input("Random Seed", 1, 9999, 42))
        drift_pct = st.slider("Annual Drift (%)",   -30, 30, 8)
        vol_pct   = st.slider("Daily Volatility (%)", 0.5, 5.0, 1.5, 0.1)

    st.divider()
    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# BACKTEST EXECUTION (CACHED)
# ─────────────────────────────────────────────

def _execute(df: pd.DataFrame, cap: float, risk: float) -> dict:
    """
    Run the engine and return a plain serialisable dict so
    st.cache_data can hash/pickle the result without issues.
    """
    engine = PaperTradingEngine(df=df, starting_capital=cap,
                                risk_per_trade=risk, tick_size=0.01)
    engine.calculate_indicators()
    engine.run()

    out: dict = {"starting_capital": cap, "n_bars": len(df), "strategies": {}}

    for s in engine.STRATEGIES:
        trades  = engine.trades[s]
        equity  = np.array(engine.equity_curve[s])
        n       = len(trades)
        wins    = sum(1 for t in trades if t.pnl > 0)
        win_rt  = wins / n * 100 if n else 0.0
        final   = float(equity[-1])
        ret_pct = (final - cap) / cap * 100.0

        peaks  = np.maximum.accumulate(equity)
        max_dd = float(((equity - peaks) / peaks * 100.0).min())

        rows = [
            {
                "Direction":   t.direction.capitalize(),
                "Entry Price": round(t.entry_price, 4),
                "Exit Price":  round(t.exit_price,  4),
                "Size":        round(t.size, 4),
                "P&L ($)":     round(t.pnl, 2),
                "P&L (%)":     round(t.pnl_pct, 2),
                "Entry Bar":   t.entry_index,
                "Exit Bar":    t.exit_index,
            }
            for t in trades
        ]

        out["strategies"][s] = {
            "equity":   equity.tolist(),
            "n_trades": n,
            "wins":     wins,
            "win_rate": win_rt,
            "final":    final,
            "ret_pct":  ret_pct,
            "max_dd":   max_dd,
            "trades":   rows,
        }

    return out


@st.cache_data(show_spinner=False)
def run_synthetic(n_bars, cap, risk, seed, drift_ann_pct, vol_daily_pct):
    df = generate_synthetic_ohlcv(
        n_bars=n_bars,
        volatility=vol_daily_pct / 100,
        drift=drift_ann_pct / 100 / 252,
        seed=seed,
    )
    return _execute(df, cap, risk)


@st.cache_data(show_spinner=False)
def run_csv(csv_bytes: bytes, cap: float, risk: float) -> dict:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df[sorted(required)].dropna().reset_index(drop=True)
    return _execute(df, cap, risk)


# Determine whether to (re-)run
need_run = run_btn or ("results" not in st.session_state)

if need_run:
    if data_source == "Upload CSV":
        if uploaded_file is None:
            if run_btn:
                st.warning("Please upload a CSV file before clicking Run.")
                st.stop()
            else:
                # First load with CSV selected — auto-run with defaults
                data_source = "Synthetic (GBM)"
                n_bars, seed, drift_pct, vol_pct = 1_200, 42, 8, 1.5

    with st.spinner("Running backtest across all four strategies..."):
        try:
            if data_source == "Upload CSV":
                results = run_csv(uploaded_file.read(), starting_capital, risk_pct / 100)
            else:
                results = run_synthetic(
                    n_bars, starting_capital, risk_pct / 100,
                    seed, drift_pct, vol_pct,
                )
            st.session_state["results"] = results
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.stop()

results = st.session_state["results"]

# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────

STRATEGY_LABELS = {
    "A_PinBar":      "A — Pin Bar Scalping",
    "B_GoldenCross": "B — Golden/Death Cross",
    "C_FibEMA":      "C — 5-8-13 Fib EMA",
    "D_ADXBreakout": "D — ADX Breakout",
}
PALETTE = ["steelblue", "darkorange", "seagreen", "crimson"]

# ─────────────────────────────────────────────
# SUMMARY METRICS ROW
# ─────────────────────────────────────────────

st.divider()
st.subheader("Strategy Summary")

cols = st.columns(4)
for col, (sid, label) in zip(cols, STRATEGY_LABELS.items()):
    s = results["strategies"][sid]
    col.metric(
        label=label,
        value=f"${s['final']:,.0f}",
        delta=f"{s['ret_pct']:+.2f}%",
    )

# ─────────────────────────────────────────────
# EQUITY CURVE CHART
# ─────────────────────────────────────────────

st.divider()
st.subheader("Equity Curves")

fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)
axes      = axes.flatten()

for ax, (sid, label), color in zip(axes, STRATEGY_LABELS.items(), PALETTE):
    s      = results["strategies"][sid]
    equity = np.array(s["equity"])
    x      = np.arange(len(equity))
    peaks  = np.maximum.accumulate(equity)

    ax.fill_between(x, equity, peaks, alpha=0.18, color="red",      label="Drawdown")
    ax.plot(x, equity,                lw=1.3,     color=color,      label="Equity")
    ax.axhline(results["starting_capital"], lw=0.8, color="dimgrey",
               linestyle="--",                                        label="Start")

    ax.set_title(
        f"{label}  |  Return: {s['ret_pct']:+.1f}%  |  Trades: {s['n_trades']}",
        fontsize=9, fontweight="bold",
    )
    ax.set_xlabel("Bar index", fontsize=8)
    ax.set_ylabel("Portfolio ($)", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)

fig.suptitle(
    "Paper Trading Backtest — Equity Curves",
    fontsize=12, fontweight="bold",
)
st.pyplot(fig)
plt.close(fig)

# ─────────────────────────────────────────────
# PER-STRATEGY DETAIL TABS
# ─────────────────────────────────────────────

st.divider()
st.subheader("Strategy Detail")

tabs = st.tabs(list(STRATEGY_LABELS.values()))

for tab, (sid, _) in zip(tabs, STRATEGY_LABELS.items()):
    s = results["strategies"][sid]
    with tab:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Trades",  s["n_trades"])
        m2.metric("Wins",          s["wins"])
        m3.metric("Win Rate",      f"{s['win_rate']:.1f}%")
        m4.metric("Max Drawdown",  f"{s['max_dd']:.2f}%")
        m5.metric("Total Return",  f"{s['ret_pct']:+.2f}%")

        st.write("")  # spacing

        if s["trades"]:
            df_trades = pd.DataFrame(s["trades"])

            # Colour-code P&L column
            def colour_pnl(val):
                colour = "color: limegreen" if val >= 0 else "color: tomato"
                return colour

            st.dataframe(
                df_trades.style.map(colour_pnl, subset=["P&L ($)", "P&L (%)"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No trades were executed for this strategy on this dataset.  "
                "Try increasing the number of bars or adjusting the parameters."
            )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.divider()
st.caption(
    "Paper trading only — no real orders are placed.  "
    f"Backtest ran over {results['n_bars']:,} bars.  "
    "Starting capital: $"
    f"{results['starting_capital']:,.0f}."
)
