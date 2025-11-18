"""
Trading Performance Metrics Module

Provides comprehensive performance analysis for trading strategies including:
- Sharpe Ratio, Sortino Ratio
- Maximum Drawdown, Calmar Ratio
- Win Rate, Profit Factor
- Risk-adjusted returns

Usage:
    from performance_metrics import PerformanceAnalyzer

    analyzer = PerformanceAnalyzer(returns, trades)
    metrics = analyzer.calculate_all_metrics()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class PerformanceAnalyzer:
    """
    Comprehensive performance metrics calculator for trading strategies.
    """

    def __init__(self, returns: np.ndarray, trades: Optional[pd.DataFrame] = None,
                 risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.

        Args:
            returns: Array of period returns (e.g., daily returns)
            trades: Optional DataFrame with trade details (entry, exit, pnl)
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.returns = np.array(returns)
        self.trades = trades
        self.risk_free_rate = risk_free_rate

        # Calculate cumulative returns
        self.cumulative_returns = np.cumprod(1 + self.returns) - 1

    def calculate_total_return(self) -> float:
        """
        Calculate total return over the period.

        Returns:
            Total return as decimal (e.g., 0.25 = 25%)
        """
        return self.cumulative_returns[-1] if len(self.cumulative_returns) > 0 else 0.0

    def calculate_annualized_return(self, periods_per_year: int = 252) -> float:
        """
        Calculate annualized return.

        Args:
            periods_per_year: Number of periods in a year (252 for daily, 52 for weekly)

        Returns:
            Annualized return
        """
        n_periods = len(self.returns)
        if n_periods == 0:
            return 0.0

        total_return = self.calculate_total_return()
        years = n_periods / periods_per_year

        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    def calculate_volatility(self, periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility (standard deviation of returns).

        Args:
            periods_per_year: Number of periods in a year

        Returns:
            Annualized volatility
        """
        if len(self.returns) == 0:
            return 0.0

        return np.std(self.returns, ddof=1) * np.sqrt(periods_per_year)

    def calculate_sharpe_ratio(self, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe Ratio (risk-adjusted return).

        Formula: (Return - Risk_Free_Rate) / Volatility

        Args:
            periods_per_year: Number of periods in a year

        Returns:
            Sharpe ratio
        """
        annual_return = self.calculate_annualized_return(periods_per_year)
        volatility = self.calculate_volatility(periods_per_year)

        if volatility == 0:
            return 0.0

        return (annual_return - self.risk_free_rate) / volatility

    def calculate_sortino_ratio(self, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino Ratio (downside risk-adjusted return).

        Uses only negative returns for volatility calculation.

        Args:
            periods_per_year: Number of periods in a year

        Returns:
            Sortino ratio
        """
        annual_return = self.calculate_annualized_return(periods_per_year)

        # Calculate downside deviation (only negative returns)
        negative_returns = self.returns[self.returns < 0]
        if len(negative_returns) == 0:
            return np.inf

        downside_std = np.std(negative_returns, ddof=1) * np.sqrt(periods_per_year)

        if downside_std == 0:
            return 0.0

        return (annual_return - self.risk_free_rate) / downside_std

    def calculate_max_drawdown(self) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown and its duration.

        Returns:
            Tuple of (max_drawdown, start_index, end_index)
        """
        if len(self.cumulative_returns) == 0:
            return 0.0, 0, 0

        # Calculate cumulative wealth
        wealth = 1 + self.cumulative_returns

        # Calculate running maximum
        running_max = np.maximum.accumulate(wealth)

        # Calculate drawdown
        drawdown = (wealth - running_max) / running_max

        # Find maximum drawdown
        max_dd = np.min(drawdown)
        end_idx = np.argmin(drawdown)

        # Find start of drawdown (last peak before max drawdown)
        start_idx = np.argmax(wealth[:end_idx + 1]) if end_idx > 0 else 0

        return abs(max_dd), start_idx, end_idx

    def calculate_calmar_ratio(self, periods_per_year: int = 252) -> float:
        """
        Calculate Calmar Ratio (return / max drawdown).

        Args:
            periods_per_year: Number of periods in a year

        Returns:
            Calmar ratio
        """
        annual_return = self.calculate_annualized_return(periods_per_year)
        max_dd, _, _ = self.calculate_max_drawdown()

        if max_dd == 0:
            return np.inf if annual_return > 0 else 0.0

        return annual_return / max_dd

    def calculate_win_rate(self) -> float:
        """
        Calculate win rate from trades.

        Returns:
            Win rate (0.0 to 1.0), or 0.0 if no trades data
        """
        if self.trades is None or len(self.trades) == 0:
            # Calculate from returns
            winning_periods = np.sum(self.returns > 0)
            total_periods = len(self.returns)
            return winning_periods / total_periods if total_periods > 0 else 0.0

        winning_trades = np.sum(self.trades['pnl'] > 0)
        total_trades = len(self.trades)

        return winning_trades / total_trades if total_trades > 0 else 0.0

    def calculate_profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Returns:
            Profit factor, or 0.0 if no trades
        """
        if self.trades is None or len(self.trades) == 0:
            # Calculate from returns
            gross_profit = np.sum(self.returns[self.returns > 0])
            gross_loss = abs(np.sum(self.returns[self.returns < 0]))
        else:
            gross_profit = np.sum(self.trades[self.trades['pnl'] > 0]['pnl'])
            gross_loss = abs(np.sum(self.trades[self.trades['pnl'] < 0]['pnl']))

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def calculate_average_win_loss_ratio(self) -> float:
        """
        Calculate average win to average loss ratio.

        Returns:
            Win/Loss ratio
        """
        if self.trades is None or len(self.trades) == 0:
            wins = self.returns[self.returns > 0]
            losses = self.returns[self.returns < 0]
        else:
            wins = self.trades[self.trades['pnl'] > 0]['pnl']
            losses = self.trades[self.trades['pnl'] < 0]['pnl']

        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0

        if avg_loss == 0:
            return np.inf if avg_win > 0 else 0.0

        return avg_win / avg_loss

    def calculate_max_consecutive_wins(self) -> int:
        """
        Calculate maximum consecutive winning periods.

        Returns:
            Max consecutive wins
        """
        if len(self.returns) == 0:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for ret in self.returns:
            if ret > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def calculate_max_consecutive_losses(self) -> int:
        """
        Calculate maximum consecutive losing periods.

        Returns:
            Max consecutive losses
        """
        if len(self.returns) == 0:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for ret in self.returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def calculate_expectancy(self) -> float:
        """
        Calculate expectancy (average expected return per trade).

        Formula: (Win_Rate * Avg_Win) - (Loss_Rate * Avg_Loss)

        Returns:
            Expectancy value
        """
        win_rate = self.calculate_win_rate()
        loss_rate = 1 - win_rate

        if self.trades is None or len(self.trades) == 0:
            wins = self.returns[self.returns > 0]
            losses = self.returns[self.returns < 0]
        else:
            wins = self.trades[self.trades['pnl'] > 0]['pnl']
            losses = self.trades[self.trades['pnl'] < 0]['pnl']

        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0

        return (win_rate * avg_win) - (loss_rate * avg_loss)

    def calculate_ulcer_index(self) -> float:
        """
        Calculate Ulcer Index (measure of downside volatility).

        Returns:
            Ulcer Index
        """
        if len(self.cumulative_returns) == 0:
            return 0.0

        wealth = 1 + self.cumulative_returns
        running_max = np.maximum.accumulate(wealth)
        drawdown = (wealth - running_max) / running_max * 100

        # Square the drawdowns, average them, take square root
        ulcer = np.sqrt(np.mean(drawdown ** 2))

        return ulcer

    def calculate_all_metrics(self, periods_per_year: int = 252) -> Dict[str, float]:
        """
        Calculate all performance metrics.

        Args:
            periods_per_year: Number of periods in a year

        Returns:
            Dictionary of all metrics
        """
        max_dd, dd_start, dd_end = self.calculate_max_drawdown()

        metrics = {
            'total_return': self.calculate_total_return(),
            'annualized_return': self.calculate_annualized_return(periods_per_year),
            'volatility': self.calculate_volatility(periods_per_year),
            'sharpe_ratio': self.calculate_sharpe_ratio(periods_per_year),
            'sortino_ratio': self.calculate_sortino_ratio(periods_per_year),
            'max_drawdown': max_dd,
            'calmar_ratio': self.calculate_calmar_ratio(periods_per_year),
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor(),
            'avg_win_loss_ratio': self.calculate_average_win_loss_ratio(),
            'expectancy': self.calculate_expectancy(),
            'max_consecutive_wins': self.calculate_max_consecutive_wins(),
            'max_consecutive_losses': self.calculate_max_consecutive_losses(),
            'ulcer_index': self.calculate_ulcer_index(),
            'total_trades': len(self.trades) if self.trades is not None else len(self.returns)
        }

        return metrics

    def print_report(self, periods_per_year: int = 252):
        """
        Print formatted performance report.

        Args:
            periods_per_year: Number of periods in a year
        """
        metrics = self.calculate_all_metrics(periods_per_year)

        print("=" * 60)
        print("PERFORMANCE METRICS REPORT")
        print("=" * 60)
        print(f"\nReturn Metrics:")
        print(f"  Total Return:       {metrics['total_return']:>10.2%}")
        print(f"  Annualized Return:  {metrics['annualized_return']:>10.2%}")
        print(f"  Volatility:         {metrics['volatility']:>10.2%}")

        print(f"\nRisk-Adjusted Metrics:")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:      {metrics['sortino_ratio']:>10.2f}")
        print(f"  Calmar Ratio:       {metrics['calmar_ratio']:>10.2f}")

        print(f"\nDrawdown Metrics:")
        print(f"  Max Drawdown:       {metrics['max_drawdown']:>10.2%}")
        print(f"  Ulcer Index:        {metrics['ulcer_index']:>10.2f}")

        print(f"\nTrade Statistics:")
        print(f"  Total Trades:       {metrics['total_trades']:>10.0f}")
        print(f"  Win Rate:           {metrics['win_rate']:>10.2%}")
        print(f"  Profit Factor:      {metrics['profit_factor']:>10.2f}")
        print(f"  Avg W/L Ratio:      {metrics['avg_win_loss_ratio']:>10.2f}")
        print(f"  Expectancy:         {metrics['expectancy']:>10.4f}")

        print(f"\nConsecutive Streaks:")
        print(f"  Max Consecutive Wins:   {metrics['max_consecutive_wins']:>6.0f}")
        print(f"  Max Consecutive Losses: {metrics['max_consecutive_losses']:>6.0f}")
        print("=" * 60)


def compare_strategies(strategies: Dict[str, PerformanceAnalyzer],
                      periods_per_year: int = 252) -> pd.DataFrame:
    """
    Compare multiple trading strategies.

    Args:
        strategies: Dictionary of strategy_name -> PerformanceAnalyzer
        periods_per_year: Number of periods in a year

    Returns:
        DataFrame comparing all metrics across strategies
    """
    comparison = {}

    for name, analyzer in strategies.items():
        comparison[name] = analyzer.calculate_all_metrics(periods_per_year)

    df = pd.DataFrame(comparison).T
    return df
