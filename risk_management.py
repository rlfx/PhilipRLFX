"""
Risk Management Module

Provides comprehensive risk management tools:
- Position sizing (Fixed, Kelly Criterion, Risk Parity)
- Stop-loss and take-profit levels
- Maximum drawdown limits
- Portfolio risk controls

Usage:
    from risk_management import RiskManager

    rm = RiskManager(account_balance=100000)
    position_size = rm.calculate_position_size(entry_price, stop_loss)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class RiskManager:
    """
    Comprehensive risk management for trading strategies.
    """

    def __init__(self, account_balance: float, max_risk_per_trade: float = 0.02,
                 max_portfolio_risk: float = 0.06, max_drawdown: float = 0.20):
        """
        Initialize risk manager.

        Args:
            account_balance: Current account balance
            max_risk_per_trade: Maximum risk per trade as fraction (default: 2%)
            max_portfolio_risk: Maximum total portfolio risk (default: 6%)
            max_drawdown: Maximum allowed drawdown before stopping (default: 20%)
        """
        self.account_balance = account_balance
        self.initial_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.max_drawdown = max_drawdown

        self.peak_balance = account_balance
        self.current_drawdown = 0.0
        self.open_positions = []

    def update_balance(self, new_balance: float):
        """
        Update account balance and drawdown tracking.

        Args:
            new_balance: New account balance
        """
        self.account_balance = new_balance

        # Update peak and drawdown
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

        self.current_drawdown = (self.peak_balance - new_balance) / self.peak_balance

    def is_max_drawdown_exceeded(self) -> bool:
        """
        Check if maximum drawdown limit is exceeded.

        Returns:
            True if drawdown exceeded, False otherwise
        """
        return self.current_drawdown >= self.max_drawdown

    def calculate_position_size_fixed_risk(self, entry_price: float,
                                          stop_loss: float,
                                          risk_fraction: Optional[float] = None) -> float:
        """
        Calculate position size using fixed risk method.

        Position Size = (Account Balance * Risk%) / (Entry - Stop Loss)

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_fraction: Risk fraction override (uses default if None)

        Returns:
            Position size in units
        """
        if risk_fraction is None:
            risk_fraction = self.max_risk_per_trade

        risk_amount = self.account_balance * risk_fraction
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            return 0.0

        position_size = risk_amount / price_risk

        return position_size

    def calculate_position_size_kelly(self, win_rate: float, avg_win: float,
                                     avg_loss: float, fraction: float = 0.5) -> float:
        """
        Calculate position size using Kelly Criterion.

        Kelly% = (Win_Rate * Avg_Win - Loss_Rate * Avg_Loss) / Avg_Win

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade size
            avg_loss: Average losing trade size (positive number)
            fraction: Fraction of Kelly to use (default: 0.5 for half-Kelly)

        Returns:
            Position size as fraction of account balance
        """
        loss_rate = 1 - win_rate

        if avg_win == 0:
            return 0.0

        kelly_pct = (win_rate * avg_win - loss_rate * avg_loss) / avg_win

        # Apply fractional Kelly and cap at max risk
        kelly_pct = max(0, min(kelly_pct * fraction, self.max_risk_per_trade * 2))

        return kelly_pct

    def calculate_position_size_volatility_based(self, prices: pd.Series,
                                                 target_volatility: float = 0.15) -> float:
        """
        Calculate position size based on volatility targeting.

        Args:
            prices: Price series for volatility calculation
            target_volatility: Target portfolio volatility (annualized)

        Returns:
            Position size multiplier
        """
        if len(prices) < 20:
            return 1.0

        # Calculate realized volatility
        returns = prices.pct_change().dropna()
        realized_vol = returns.std() * np.sqrt(252)  # Annualize

        if realized_vol == 0:
            return 1.0

        # Scale position to achieve target volatility
        position_multiplier = target_volatility / realized_vol

        # Cap position size
        position_multiplier = min(position_multiplier, 2.0)

        return position_multiplier

    def calculate_stop_loss_atr(self, current_price: float, atr: float,
                               multiplier: float = 2.0, direction: str = 'long') -> float:
        """
        Calculate stop loss using Average True Range (ATR).

        Args:
            current_price: Current entry price
            atr: Average True Range value
            multiplier: ATR multiplier (default: 2.0)
            direction: 'long' or 'short'

        Returns:
            Stop loss price
        """
        if direction == 'long':
            stop_loss = current_price - (atr * multiplier)
        else:  # short
            stop_loss = current_price + (atr * multiplier)

        return stop_loss

    def calculate_take_profit_atr(self, current_price: float, atr: float,
                                  multiplier: float = 3.0, direction: str = 'long') -> float:
        """
        Calculate take profit using Average True Range (ATR).

        Args:
            current_price: Current entry price
            atr: Average True Range value
            multiplier: ATR multiplier (default: 3.0 for 1.5:1 risk-reward)
            direction: 'long' or 'short'

        Returns:
            Take profit price
        """
        if direction == 'long':
            take_profit = current_price + (atr * multiplier)
        else:  # short
            take_profit = current_price - (atr * multiplier)

        return take_profit

    def calculate_stop_loss_percentage(self, entry_price: float,
                                      percentage: float = 0.02,
                                      direction: str = 'long') -> float:
        """
        Calculate stop loss using fixed percentage.

        Args:
            entry_price: Entry price
            percentage: Stop loss percentage (e.g., 0.02 = 2%)
            direction: 'long' or 'short'

        Returns:
            Stop loss price
        """
        if direction == 'long':
            return entry_price * (1 - percentage)
        else:
            return entry_price * (1 + percentage)

    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float,
                                   take_profit: float) -> float:
        """
        Calculate risk-reward ratio.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Risk-reward ratio
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        if risk == 0:
            return 0.0

        return reward / risk

    def should_enter_trade(self, signal_strength: float = 1.0) -> bool:
        """
        Determine if new trade should be entered based on risk limits.

        Args:
            signal_strength: Strength of trading signal (0-1)

        Returns:
            True if trade should be entered, False otherwise
        """
        # Check drawdown limit
        if self.is_max_drawdown_exceeded():
            return False

        # Check portfolio risk limit
        current_risk = self.calculate_current_portfolio_risk()
        if current_risk >= self.max_portfolio_risk:
            return False

        # Require minimum signal strength
        if signal_strength < 0.5:
            return False

        return True

    def calculate_current_portfolio_risk(self) -> float:
        """
        Calculate current portfolio risk from open positions.

        Returns:
            Current portfolio risk as fraction of account
        """
        if not self.open_positions:
            return 0.0

        total_risk = sum(pos['risk_amount'] for pos in self.open_positions)
        return total_risk / self.account_balance

    def add_position(self, entry_price: float, stop_loss: float, position_size: float):
        """
        Add new position to tracking.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            position_size: Position size in units
        """
        risk_per_unit = abs(entry_price - stop_loss)
        risk_amount = risk_per_unit * position_size

        position = {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'position_size': position_size,
            'risk_amount': risk_amount
        }

        self.open_positions.append(position)

    def remove_position(self, index: int = 0):
        """
        Remove position from tracking.

        Args:
            index: Index of position to remove
        """
        if 0 <= index < len(self.open_positions):
            self.open_positions.pop(index)

    def get_position_sizing_recommendation(self, entry_price: float, stop_loss: float,
                                          historical_performance: Optional[Dict] = None
                                          ) -> Dict[str, float]:
        """
        Get comprehensive position sizing recommendation.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            historical_performance: Optional dict with win_rate, avg_win, avg_loss

        Returns:
            Dictionary with different position sizing methods
        """
        recommendation = {}

        # Fixed risk method
        recommendation['fixed_risk'] = self.calculate_position_size_fixed_risk(
            entry_price, stop_loss
        )

        # Kelly criterion (if historical performance available)
        if historical_performance:
            kelly_fraction = self.calculate_position_size_kelly(
                historical_performance['win_rate'],
                historical_performance['avg_win'],
                historical_performance['avg_loss']
            )
            recommendation['kelly_size'] = self.account_balance * kelly_fraction
        else:
            recommendation['kelly_size'] = None

        # Conservative (half of fixed risk)
        recommendation['conservative'] = recommendation['fixed_risk'] * 0.5

        # Aggressive (1.5x fixed risk, capped)
        max_aggressive = self.account_balance * (self.max_risk_per_trade * 1.5)
        price_risk = abs(entry_price - stop_loss)
        recommendation['aggressive'] = min(
            recommendation['fixed_risk'] * 1.5,
            max_aggressive / price_risk if price_risk > 0 else 0
        )

        return recommendation

    def trailing_stop_loss(self, current_price: float, entry_price: float,
                          current_stop: float, trail_percent: float = 0.02,
                          direction: str = 'long') -> float:
        """
        Calculate trailing stop loss.

        Args:
            current_price: Current market price
            entry_price: Original entry price
            current_stop: Current stop loss level
            trail_percent: Trailing percentage (e.g., 0.02 = 2%)
            direction: 'long' or 'short'

        Returns:
            Updated stop loss price
        """
        if direction == 'long':
            # Only move stop up, never down
            new_stop = current_price * (1 - trail_percent)
            return max(current_stop, new_stop)
        else:  # short
            # Only move stop down, never up
            new_stop = current_price * (1 + trail_percent)
            return min(current_stop, new_stop)

    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Get current risk metrics.

        Returns:
            Dictionary of risk metrics
        """
        return {
            'account_balance': self.account_balance,
            'peak_balance': self.peak_balance,
            'current_drawdown': self.current_drawdown,
            'max_drawdown_limit': self.max_drawdown,
            'drawdown_remaining': self.max_drawdown - self.current_drawdown,
            'portfolio_risk': self.calculate_current_portfolio_risk(),
            'max_portfolio_risk': self.max_portfolio_risk,
            'risk_remaining': self.max_portfolio_risk - self.calculate_current_portfolio_risk(),
            'open_positions': len(self.open_positions),
            'total_return': (self.account_balance - self.initial_balance) / self.initial_balance
        }

    def print_risk_report(self):
        """Print formatted risk management report."""
        metrics = self.get_risk_metrics()

        print("=" * 60)
        print("RISK MANAGEMENT REPORT")
        print("=" * 60)
        print(f"\nAccount Status:")
        print(f"  Current Balance:    ${metrics['account_balance']:>12,.2f}")
        print(f"  Peak Balance:       ${metrics['peak_balance']:>12,.2f}")
        print(f"  Total Return:       {metrics['total_return']:>12.2%}")

        print(f"\nDrawdown Status:")
        print(f"  Current Drawdown:   {metrics['current_drawdown']:>12.2%}")
        print(f"  Max Allowed:        {metrics['max_drawdown_limit']:>12.2%}")
        print(f"  Remaining Buffer:   {metrics['drawdown_remaining']:>12.2%}")

        print(f"\nPortfolio Risk:")
        print(f"  Current Risk:       {metrics['portfolio_risk']:>12.2%}")
        print(f"  Max Allowed:        {metrics['max_portfolio_risk']:>12.2%}")
        print(f"  Remaining Capacity: {metrics['risk_remaining']:>12.2%}")

        print(f"\nPosition Status:")
        print(f"  Open Positions:     {metrics['open_positions']:>12.0f}")
        print("=" * 60)
