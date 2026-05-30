"""
Paper Trading Backtest Engine — CFI Guide to Trading Strategies
================================================================
Implements four independent trading strategies with a built-in
paper trading simulation (no real money, no external API calls).

Strategies
----------
A  Pin Bar Scalping         (10/21 EMA + pin bar structure)
B  Golden / Death Cross     (50 SMA / 200 SMA crossover)
C  5-8-13 Fibonacci EMA     (5/8/13 EMA with fan confirmation)
D  Simple ADX Breakout      (14-period Wilder ADX > 25 trigger)

Usage
-----
    python paper_trading_engine.py

To use real market data, replace the generate_synthetic_ohlcv()
call in __main__ with a DataFrame loader, e.g.:

    import yfinance as yf
    df = yf.download("SPY", start="2018-01-01", end="2024-01-01")
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()

The DataFrame must have lowercase column names and a DatetimeIndex.
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — SYNTHETIC DATA GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_synthetic_ohlcv(
    n_bars: int = 1_200,
    start_price: float = 100.0,
    volatility: float = 0.015,
    drift: float = 0.0003,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Produce realistic OHLCV candle data via Geometric Brownian Motion.

    The intra-bar structure (open / high / low wicks) is generated with
    random noise so pin bars and breakout patterns arise organically,
    giving every strategy a realistic signal environment.
    """
    np.random.seed(seed)

    dates   = pd.date_range("2020-01-01", periods=n_bars, freq="B")
    returns = np.random.normal(drift, volatility, n_bars)
    close   = start_price * np.exp(np.cumsum(returns))

    # Build OHLC — body range and wicks scale proportionally to price level
    body_half = np.abs(np.random.normal(0, volatility * 0.4, n_bars)) * close
    open_     = close + np.random.choice([-1, 1], n_bars) * body_half
    wick_size = np.abs(np.random.normal(0, volatility * 0.35, n_bars)) * close
    high      = np.maximum(open_, close) + wick_size
    low       = np.minimum(open_, close) - wick_size
    volume    = np.random.randint(500_000, 2_000_000, n_bars).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    ).round(4)


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class Position:
    """Tracks one open paper position."""
    strategy:    str
    direction:   str    # "long" | "short"
    entry_price: float
    entry_index: int
    size:        float  # number of units held
    stop_loss:   float  # hard stop price level


@dataclass
class Trade:
    """Immutable record of one completed round-trip trade."""
    strategy:    str
    direction:   str
    entry_price: float
    exit_price:  float
    entry_index: int
    exit_index:  int
    size:        float
    pnl:         float   # realised P&L in dollar terms
    pnl_pct:     float   # P&L as % of entry notional


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — PAPER TRADING ENGINE
# ═══════════════════════════════════════════════════════════════

class PaperTradingEngine:
    """
    Runs four strategy simulations independently over a shared OHLCV
    DataFrame.  Each strategy maintains its own cash ledger, open
    position register, trade log, and equity curve so performance
    metrics are perfectly comparable.

    Cash accounting model
    ---------------------
    Opening any position (long or short) reserves capital equal to
    size × entry_price, deducted from cash.

    Long close  : cash += size × exit_price
    Short close : cash += size × entry_price + pnl
                = size × (2 × entry − exit)

    Equity (mark-to-market while a position is open)
    = cash + size × entry_price + unrealised_pnl
    """

    STRATEGIES = ["A_PinBar", "B_GoldenCross", "C_FibEMA", "D_ADXBreakout"]

    def __init__(
        self,
        df:               pd.DataFrame,
        starting_capital: float = 100_000.0,
        risk_per_trade:   float = 0.01,   # fraction of equity risked per trade
        tick_size:        float = 0.01,   # min price increment for SL placement
    ) -> None:
        self.df               = df.copy().reset_index(drop=True)
        self.starting_capital = starting_capital
        self.risk_per_trade   = risk_per_trade
        self.tick_size        = tick_size

        # Independent state per strategy
        self.cash:         dict[str, float]             = {s: starting_capital for s in self.STRATEGIES}
        self.position:     dict[str, Optional[Position]] = {s: None            for s in self.STRATEGIES}
        self.trades:       dict[str, list[Trade]]        = {s: []              for s in self.STRATEGIES}
        self.equity_curve: dict[str, list[float]]        = {s: [starting_capital] for s in self.STRATEGIES}

        # Populated by calculate_indicators()
        self.ind: pd.DataFrame = pd.DataFrame(index=self.df.index)

    # ─────────────────────────────────────────────
    # 3.1  INDICATORS
    # ─────────────────────────────────────────────

    def calculate_indicators(self) -> None:
        """
        Compute every technical indicator required by all four strategies
        and store them in self.ind.  Call this once before run().
        """
        c, h, l = self.df["close"], self.df["high"], self.df["low"]

        # Strategy A — Pin Bar (10 EMA, 21 EMA)
        self.ind["ema10"] = c.ewm(span=10, adjust=False).mean()
        self.ind["ema21"] = c.ewm(span=21, adjust=False).mean()

        # Strategy B — Golden/Death Cross (50 SMA, 200 SMA)
        self.ind["sma50"]  = c.rolling(50).mean()
        self.ind["sma200"] = c.rolling(200).mean()

        # Strategy C — Fibonacci EMA ribbon (5 EMA, 8 EMA, 13 EMA)
        self.ind["ema5"]  = c.ewm(span=5,  adjust=False).mean()
        self.ind["ema8"]  = c.ewm(span=8,  adjust=False).mean()
        self.ind["ema13"] = c.ewm(span=13, adjust=False).mean()

        # Strategy D — 14-period Wilder ADX
        self.ind["adx"] = self._wilder_adx(h, l, c, period=14)

    @staticmethod
    def _wilder_adx(
        high: pd.Series,
        low:  pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Compute ADX using Wilder exponential smoothing (alpha = 1/period).

        Steps: True Range → +DM / -DM → smooth to ATR / +DI / -DI → DX → ADX.
        """
        prev_close = close.shift(1)
        prev_high  = high.shift(1)
        prev_low   = low.shift(1)

        # True Range
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)

        # Directional movement
        up_move   = high - prev_high
        down_move = prev_low - low

        dm_plus  = pd.Series(
            np.where((up_move   > down_move) & (up_move   > 0), up_move,   0.0),
            index=close.index,
        )
        dm_minus = pd.Series(
            np.where((down_move > up_move)   & (down_move > 0), down_move, 0.0),
            index=close.index,
        )

        alpha = 1.0 / period
        atr        = tr.ewm(alpha=alpha, adjust=False).mean()
        smooth_pos = dm_plus.ewm(alpha=alpha,  adjust=False).mean()
        smooth_neg = dm_minus.ewm(alpha=alpha, adjust=False).mean()

        denom = (smooth_pos + smooth_neg).replace(0.0, np.nan)
        dx    = (100.0 * (smooth_pos - smooth_neg).abs() / denom).fillna(0.0)
        return dx.ewm(alpha=alpha, adjust=False).mean()

    # ─────────────────────────────────────────────
    # 3.2  SIGNAL DETECTION
    # ─────────────────────────────────────────────

    def check_signals(self, i: int) -> dict[str, Optional[str]]:
        """
        Evaluate all four strategy signal conditions at bar index i.

        Signals are evaluated independently on the same data;
        none block or interact with one another.

        Returns
        -------
        dict mapping each strategy name to "long", "short", or None.
        """
        signals: dict[str, Optional[str]] = {s: None for s in self.STRATEGIES}

        if i < 1:
            return signals

        df  = self.df
        ind = self.ind
        p   = i - 1   # index of the previous (completed) bar

        # ── A  Pin Bar Scalping ──────────────────────────────────────────
        # Inspect the completed bar at p; if it qualifies as a pin bar
        # that is extended away from both EMAs, signal entry at bar i open.

        o_p, h_p, l_p, c_p = df.loc[p, ["open", "high", "low", "close"]]
        body       = abs(c_p - o_p)
        full_range = h_p - l_p

        if full_range > 1e-9:
            upper_wick = h_p - max(o_p, c_p)
            lower_wick = min(o_p, c_p) - l_p

            ema10_p = ind.loc[p, "ema10"]
            ema21_p = ind.loc[p, "ema21"]

            # Structural pin bar: tail ≥ 2.5× body, opposite wick ≤ 10% of range
            bullish_pin = (
                lower_wick >= 2.5 * max(body, 1e-9) and
                upper_wick <= 0.10 * full_range      and
                l_p < min(ema10_p, ema21_p)          # candle low below both EMAs
            )
            bearish_pin = (
                upper_wick >= 2.5 * max(body, 1e-9) and
                lower_wick <= 0.10 * full_range      and
                h_p > max(ema10_p, ema21_p)           # candle high above both EMAs
            )

            if bullish_pin:
                signals["A_PinBar"] = "long"
            elif bearish_pin:
                signals["A_PinBar"] = "short"

        # ── B  Golden / Death Cross ──────────────────────────────────────
        s50_now,  s50_prev  = ind.loc[i, "sma50"],  ind.loc[p, "sma50"]
        s200_now, s200_prev = ind.loc[i, "sma200"], ind.loc[p, "sma200"]

        if all(pd.notna(v) for v in (s50_now, s50_prev, s200_now, s200_prev)):
            if s50_prev <= s200_prev and s50_now > s200_now:   # Golden Cross
                signals["B_GoldenCross"] = "long"
            elif s50_prev >= s200_prev and s50_now < s200_now: # Death Cross
                signals["B_GoldenCross"] = "short"

        # ── C  5-8-13 Fibonacci EMA ──────────────────────────────────────
        e5_now,  e5_prev  = ind.loc[i, "ema5"],  ind.loc[p, "ema5"]
        e8_now            = ind.loc[i, "ema8"]
        e13_now, e13_prev = ind.loc[i, "ema13"], ind.loc[p, "ema13"]

        spread_now  = abs(e5_now  - e13_now)
        spread_prev = abs(e5_prev - e13_prev)
        candle_range = df.loc[i, "high"] - df.loc[i, "low"]

        # Consolidation filter: suppress signals when the EMA ribbon is tight
        compressed = (spread_now < 0.05 * candle_range) if candle_range > 1e-9 else True

        if not compressed:
            fanning       = spread_now > spread_prev
            aligned_long  = e5_now > e8_now > e13_now   # stacked bullishly
            aligned_short = e5_now < e8_now < e13_now   # stacked bearishly

            bull_cross = e5_prev <= e13_prev and e5_now > e13_now
            bear_cross = e5_prev >= e13_prev and e5_now < e13_now

            if bull_cross and fanning and aligned_long:
                signals["C_FibEMA"] = "long"
            elif bear_cross and fanning and aligned_short:
                signals["C_FibEMA"] = "short"

        # ── D  ADX Breakout ──────────────────────────────────────────────
        adx_now  = ind.loc[i, "adx"]
        adx_prev = ind.loc[p, "adx"]

        if pd.notna(adx_now) and pd.notna(adx_prev):
            crossed_above_25 = adx_prev <= 25.0 and adx_now > 25.0
            slope_positive   = adx_now > adx_prev

            if crossed_above_25 and slope_positive and i >= 3:
                # Direction determined by OLS slope of the last 4 closes
                closes      = df.loc[i - 3: i, "close"].values
                price_slope = np.polyfit(range(len(closes)), closes, 1)[0]

                if price_slope > 0:
                    signals["D_ADXBreakout"] = "long"
                elif price_slope < 0:
                    signals["D_ADXBreakout"] = "short"

        return signals

    # ─────────────────────────────────────────────
    # 3.3  TRADE EXECUTION
    # ─────────────────────────────────────────────

    def execute_paper_trade(self, i: int, signals: dict[str, Optional[str]]) -> None:
        """
        For every strategy at bar i:
          1. Check exit conditions on the open position (if any).
          2. If flat after exits, check for a new entry signal.
          3. Append mark-to-market equity to the curve.

        Exits are evaluated before entries so a flip trade (e.g.
        Golden Cross long → Death Cross short) completes correctly
        within the same bar.
        """
        for strategy in self.STRATEGIES:
            # ── EXIT ──────────────────────────────────────────────────────
            pos = self.position[strategy]
            if pos is not None:
                exited = self._check_exit(i, strategy, pos, signals)
                if exited:
                    self.position[strategy] = None

            # ── ENTRY ──────────────────────────────────────────────────────
            sig = signals.get(strategy)
            if sig is not None and self.position[strategy] is None:
                self._open_position(i, strategy, sig)

            # ── EQUITY MARK-TO-MARKET ──────────────────────────────────────
            pos   = self.position[strategy]
            price = self.df.loc[i, "close"]
            if pos is not None:
                # Cash already excludes the reserved capital (size × entry_price),
                # so we add it back plus unrealised P&L to get true equity.
                equity = (
                    self.cash[strategy]
                    + pos.size * pos.entry_price
                    + self._unrealised_pnl(pos, price)
                )
            else:
                equity = self.cash[strategy]

            self.equity_curve[strategy].append(equity)

    # ── Exit dispatcher ──────────────────────────────────────────────────

    def _check_exit(
        self,
        i:        int,
        strategy: str,
        pos:      Position,
        signals:  dict[str, Optional[str]],
    ) -> bool:
        """Dispatch to the per-strategy exit handler. Returns True if closed."""
        if strategy == "A_PinBar":
            return self._exit_pin_bar(i, pos)
        if strategy in ("B_GoldenCross", "C_FibEMA"):
            return self._exit_on_opposite_signal(i, strategy, pos, signals)
        if strategy == "D_ADXBreakout":
            return self._exit_adx(i, pos)
        return False

    def _exit_pin_bar(self, i: int, pos: Position) -> bool:
        """
        Exit conditions for Strategy A:
        - Stop-loss : 1 tick beyond the tail tip of the pin bar.
        - Take-profit: price returns to touch the 10 EMA.
        """
        bar   = self.df.loc[i]
        ema10 = self.ind.loc[i, "ema10"]

        if pos.direction == "long":
            sl_hit = bar["low"]  <= pos.stop_loss
            tp_hit = bar["high"] >= ema10
            exit_p = pos.stop_loss if sl_hit else (ema10 if tp_hit else None)
        else:
            sl_hit = bar["high"] >= pos.stop_loss
            tp_hit = bar["low"]  <= ema10
            exit_p = pos.stop_loss if sl_hit else (ema10 if tp_hit else None)

        if exit_p is not None:
            self._close_position(pos, exit_p, i, "A_PinBar")
            return True
        return False

    def _exit_on_opposite_signal(
        self,
        i:        int,
        strategy: str,
        pos:      Position,
        signals:  dict[str, Optional[str]],
    ) -> bool:
        """
        Exit conditions for Strategies B and C:
        Hold until the opposite crossover signal fires, then exit at
        the open of the current bar (crossover was detected at the
        prior close).
        """
        sig = signals.get(strategy)
        flip = (pos.direction == "long" and sig == "short") or \
               (pos.direction == "short" and sig == "long")
        if flip:
            self._close_position(pos, self.df.loc[i, "open"], i, strategy)
            return True
        return False

    def _exit_adx(self, i: int, pos: Position) -> bool:
        """
        Exit condition for Strategy D:
        Close immediately when ADX drops below 25 OR the ADX slope
        turns negative (momentum fading).
        """
        adx_now  = self.ind.loc[i,   "adx"]
        adx_prev = self.ind.loc[i-1, "adx"] if i > 0 else adx_now

        if adx_now < 25.0 or adx_now < adx_prev:
            self._close_position(pos, self.df.loc[i, "close"], i, "D_ADXBreakout")
            return True
        return False

    # ── Position open / close helpers ────────────────────────────────────

    def _open_position(self, i: int, strategy: str, direction: str) -> None:
        """
        Compute stop-loss, position size via fixed-fractional risk,
        reserve capital, and record the new Position.
        """
        df  = self.df
        ind = self.ind

        # Strategy A enters at the open of bar i (the first candle after the pin)
        entry_price = df.loc[i, "open"] if strategy == "A_PinBar" else df.loc[i, "close"]

        # ── Stop-loss placement ──────────────────────────────────────────
        if strategy == "A_PinBar":
            p    = i - 1
            stop = (df.loc[p, "low"]  - self.tick_size) if direction == "long" \
              else (df.loc[p, "high"] + self.tick_size)
        else:
            # ATR stop: mean of (high − low) over the last 14 bars
            start = max(0, i - 13)
            atr   = (df.loc[start:i, "high"] - df.loc[start:i, "low"]).mean()
            atr   = max(atr, self.tick_size)  # floor to avoid zero stop distance
            stop  = (entry_price - atr) if direction == "long" else (entry_price + atr)

        risk_per_unit = abs(entry_price - stop)
        if risk_per_unit < 1e-9:
            return

        # ── Fixed-fractional position sizing ────────────────────────────
        # Risk dollars = current equity × risk_per_trade
        # We approximate equity with cash (conservative when no position is open)
        risk_dollars = self.cash[strategy] * self.risk_per_trade
        size         = risk_dollars / risk_per_unit

        # Cap at available cash so there is no implicit leverage
        max_size = self.cash[strategy] / entry_price
        size     = min(size, max_size)
        if size <= 0.0:
            return

        # Reserve capital (identical treatment for long and short)
        self.cash[strategy] -= size * entry_price

        self.position[strategy] = Position(
            strategy    = strategy,
            direction   = direction,
            entry_price = entry_price,
            entry_index = i,
            size        = size,
            stop_loss   = stop,
        )

    def _close_position(
        self,
        pos:        Position,
        exit_price: float,
        exit_index: int,
        strategy:   str,
    ) -> None:
        """
        Settle the trade: return capital + P&L to cash and append a
        Trade record to the strategy's log.
        """
        if pos.direction == "long":
            pnl = pos.size * (exit_price - pos.entry_price)
            # Return sale proceeds (cost basis + P&L)
            self.cash[strategy] += pos.size * exit_price
        else:
            pnl = pos.size * (pos.entry_price - exit_price)
            # Return reserved margin + realised P&L
            self.cash[strategy] += pos.size * pos.entry_price + pnl

        pnl_pct = pnl / (pos.size * pos.entry_price) * 100.0

        self.trades[strategy].append(Trade(
            strategy    = strategy,
            direction   = pos.direction,
            entry_price = pos.entry_price,
            exit_price  = exit_price,
            entry_index = pos.entry_index,
            exit_index  = exit_index,
            size        = pos.size,
            pnl         = pnl,
            pnl_pct     = pnl_pct,
        ))

    @staticmethod
    def _unrealised_pnl(pos: Position, current_price: float) -> float:
        """Mark-to-market P&L for an open position."""
        if pos.direction == "long":
            return pos.size * (current_price - pos.entry_price)
        return pos.size * (pos.entry_price - current_price)

    # ─────────────────────────────────────────────
    # 3.4  MAIN BACKTEST LOOP
    # ─────────────────────────────────────────────

    def run(self) -> None:
        """
        Iterate over the OHLCV DataFrame bar by bar.  For each bar,
        compute signals and execute the paper-trade logic for all
        four strategies.  Open positions at end-of-data are closed
        at the final bar's closing price.
        """
        n = len(self.df)
        print(f"Backtesting {n:,} bars across {len(self.STRATEGIES)} strategies...")

        for i in range(1, n):
            sigs = self.check_signals(i)
            self.execute_paper_trade(i, sigs)

        # Force-close any positions still open at the end of the dataset
        last_i     = n - 1
        last_price = self.df.loc[last_i, "close"]
        for strategy in self.STRATEGIES:
            pos = self.position[strategy]
            if pos is not None:
                self._close_position(pos, last_price, last_i, strategy)
                self.position[strategy] = None

        print("Backtest complete.\n")

    # ─────────────────────────────────────────────
    # 3.5  PERFORMANCE REPORTING
    # ─────────────────────────────────────────────

    def performance_report(self) -> None:
        """
        Print a formatted performance summary for each strategy listing:
          - Total Trades Executed
          - Win Rate Percentage
          - Starting Capital vs Final Capital Balance
          - Maximum Drawdown experienced during the run
        """
        divider = "=" * 62

        for strategy in self.STRATEGIES:
            trades  = self.trades[strategy]
            equity  = np.array(self.equity_curve[strategy])
            n_total = len(trades)

            win_rate  = (sum(1 for t in trades if t.pnl > 0) / n_total * 100) \
                        if n_total else 0.0
            final_cap = equity[-1]
            total_ret = (final_cap - self.starting_capital) / self.starting_capital * 100.0

            # Maximum drawdown: largest peak-to-trough decline in equity
            peaks  = np.maximum.accumulate(equity)
            dd_pct = (equity - peaks) / peaks * 100.0
            max_dd = dd_pct.min()

            print(divider)
            print(f"  STRATEGY : {strategy}")
            print(divider)
            print(f"  Total Trades Executed  : {n_total:>6}")
            print(f"  Win Rate               : {win_rate:>6.1f}%")
            print(f"  Starting Capital       : ${self.starting_capital:>12,.2f}")
            print(f"  Final Capital          : ${final_cap:>12,.2f}")
            print(f"  Total Return           : {total_ret:>+7.2f}%")
            print(f"  Max Drawdown           : {max_dd:>7.2f}%")
            print()

    # ─────────────────────────────────────────────
    # 3.6  EQUITY CURVE PLOTTING
    # ─────────────────────────────────────────────

    def plot_equity_curves(self, save_path: str = "equity_curves.png") -> None:
        """
        Render a 2×2 grid of equity curve charts, one per strategy.
        Drawdown periods are shaded in red behind the equity line.
        The chart is saved to disk and displayed interactively.
        """
        palette = ["steelblue", "darkorange", "seagreen", "crimson"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
        axes = axes.flatten()

        for ax, strategy, color in zip(axes, self.STRATEGIES, palette):
            equity = np.array(self.equity_curve[strategy])
            x      = np.arange(len(equity))
            peaks  = np.maximum.accumulate(equity)

            ax.fill_between(x, equity, peaks,
                            alpha=0.20, color="red",   label="Drawdown region")
            ax.plot(x, equity,
                    linewidth=1.3, color=color,        label="Portfolio equity")
            ax.axhline(self.starting_capital,
                       linewidth=0.8, color="dimgrey",
                       linestyle="--",                 label="Starting capital")

            final_ret = (equity[-1] - self.starting_capital) / self.starting_capital * 100.0
            n_trades  = len(self.trades[strategy])
            ax.set_title(
                f"{strategy}   |   Return: {final_ret:+.1f}%   |   Trades: {n_trades}",
                fontsize=9,
                fontweight="bold",
            )
            ax.set_xlabel("Bar Index", fontsize=8)
            ax.set_ylabel("Portfolio Value ($)", fontsize=8)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, alpha=0.25)

        fig.suptitle(
            "Paper Trading Backtest — Equity Curves (No Real Money)",
            fontsize=13,
            fontweight="bold",
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Equity curve chart saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── 1. Load / generate price data ────────────────────────────────────
    # Swap this for a real DataFrame to run the engine on live market data.
    # Required columns (lowercase): open, high, low, close, volume
    # Required index: DatetimeIndex
    df = generate_synthetic_ohlcv(n_bars=1_200)

    print(f"Data loaded: {len(df):,} bars  "
          f"({df.index[0].date()} to {df.index[-1].date()})\n")

    # ── 2. Initialise engine ─────────────────────────────────────────────
    engine = PaperTradingEngine(
        df               = df,
        starting_capital = 100_000.0,
        risk_per_trade   = 0.01,    # risk 1% of equity per trade
        tick_size        = 0.01,    # 1 cent minimum stop distance
    )

    # ── 3. Compute all indicators ────────────────────────────────────────
    engine.calculate_indicators()

    # ── 4. Run the backtest ──────────────────────────────────────────────
    engine.run()

    # ── 5. Print the performance report ─────────────────────────────────
    engine.performance_report()

    # ── 6. Plot and save equity curves ──────────────────────────────────
    engine.plot_equity_curves(save_path="equity_curves.png")
