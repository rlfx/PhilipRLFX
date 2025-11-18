"""
Unit tests for risk management module.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_management import RiskManager


class TestRiskManager(unittest.TestCase):
    """Test cases for RiskManager class."""

    def setUp(self):
        """Set up test data."""
        self.rm = RiskManager(
            account_balance=100000,
            max_risk_per_trade=0.02,
            max_portfolio_risk=0.06,
            max_drawdown=0.20
        )

    def test_initialization(self):
        """Test risk manager initialization."""
        self.assertEqual(self.rm.account_balance, 100000)
        self.assertEqual(self.rm.max_risk_per_trade, 0.02)
        self.assertEqual(self.rm.max_portfolio_risk, 0.06)
        self.assertEqual(self.rm.max_drawdown, 0.20)

    def test_update_balance(self):
        """Test balance update and drawdown tracking."""
        # Increase balance
        self.rm.update_balance(110000)
        self.assertEqual(self.rm.account_balance, 110000)
        self.assertEqual(self.rm.peak_balance, 110000)
        self.assertEqual(self.rm.current_drawdown, 0.0)

        # Decrease balance
        self.rm.update_balance(100000)
        self.assertAlmostEqual(self.rm.current_drawdown, 0.0909, places=3)

    def test_is_max_drawdown_exceeded(self):
        """Test max drawdown check."""
        self.assertFalse(self.rm.is_max_drawdown_exceeded())

        # Simulate large loss
        self.rm.update_balance(75000)  # 25% drawdown
        self.assertTrue(self.rm.is_max_drawdown_exceeded())

    def test_calculate_position_size_fixed_risk(self):
        """Test fixed risk position sizing."""
        entry_price = 100
        stop_loss = 95

        position_size = self.rm.calculate_position_size_fixed_risk(entry_price, stop_loss)

        # Risk amount should be 2% of 100000 = 2000
        # Price risk = 100 - 95 = 5
        # Position size = 2000 / 5 = 400
        self.assertAlmostEqual(position_size, 400, places=0)

    def test_calculate_position_size_kelly(self):
        """Test Kelly criterion position sizing."""
        win_rate = 0.6
        avg_win = 100
        avg_loss = 50

        kelly_pct = self.rm.calculate_position_size_kelly(win_rate, avg_win, avg_loss)

        self.assertIsInstance(kelly_pct, float)
        self.assertGreater(kelly_pct, 0)

    def test_calculate_stop_loss_atr(self):
        """Test ATR-based stop loss calculation."""
        current_price = 100
        atr = 2.0

        # Long position
        stop_loss_long = self.rm.calculate_stop_loss_atr(current_price, atr, direction='long')
        self.assertLess(stop_loss_long, current_price)
        self.assertAlmostEqual(stop_loss_long, 96.0, places=1)

        # Short position
        stop_loss_short = self.rm.calculate_stop_loss_atr(current_price, atr, direction='short')
        self.assertGreater(stop_loss_short, current_price)
        self.assertAlmostEqual(stop_loss_short, 104.0, places=1)

    def test_calculate_stop_loss_percentage(self):
        """Test percentage-based stop loss calculation."""
        entry_price = 100
        percentage = 0.02

        # Long position
        stop_loss_long = self.rm.calculate_stop_loss_percentage(
            entry_price, percentage, direction='long'
        )
        self.assertAlmostEqual(stop_loss_long, 98.0, places=1)

        # Short position
        stop_loss_short = self.rm.calculate_stop_loss_percentage(
            entry_price, percentage, direction='short'
        )
        self.assertAlmostEqual(stop_loss_short, 102.0, places=1)

    def test_calculate_risk_reward_ratio(self):
        """Test risk-reward ratio calculation."""
        entry_price = 100
        stop_loss = 95
        take_profit = 110

        rr_ratio = self.rm.calculate_risk_reward_ratio(entry_price, stop_loss, take_profit)

        # Risk = 5, Reward = 10, Ratio = 2.0
        self.assertAlmostEqual(rr_ratio, 2.0, places=1)

    def test_should_enter_trade(self):
        """Test trade entry decision."""
        # Should allow trade initially
        self.assertTrue(self.rm.should_enter_trade(signal_strength=1.0))

        # Should not allow with weak signal
        self.assertFalse(self.rm.should_enter_trade(signal_strength=0.3))

        # Should not allow when max drawdown exceeded
        self.rm.update_balance(75000)
        self.assertFalse(self.rm.should_enter_trade(signal_strength=1.0))

    def test_add_remove_position(self):
        """Test position tracking."""
        self.assertEqual(len(self.rm.open_positions), 0)

        # Add position
        self.rm.add_position(entry_price=100, stop_loss=95, position_size=100)
        self.assertEqual(len(self.rm.open_positions), 1)

        # Remove position
        self.rm.remove_position(0)
        self.assertEqual(len(self.rm.open_positions), 0)

    def test_trailing_stop_loss(self):
        """Test trailing stop loss calculation."""
        entry_price = 100
        current_stop = 95

        # Price moves up
        new_price = 110
        new_stop = self.rm.trailing_stop_loss(
            new_price, entry_price, current_stop, trail_percent=0.02, direction='long'
        )

        # Stop should move up
        self.assertGreater(new_stop, current_stop)
        self.assertAlmostEqual(new_stop, 107.8, places=1)

        # Price moves down (stop should not move down)
        lower_price = 105
        new_stop2 = self.rm.trailing_stop_loss(
            lower_price, entry_price, new_stop, trail_percent=0.02, direction='long'
        )
        self.assertEqual(new_stop2, new_stop)

    def test_get_risk_metrics(self):
        """Test risk metrics retrieval."""
        metrics = self.rm.get_risk_metrics()

        self.assertIsInstance(metrics, dict)
        self.assertIn('account_balance', metrics)
        self.assertIn('current_drawdown', metrics)
        self.assertIn('portfolio_risk', metrics)
        self.assertIn('open_positions', metrics)


if __name__ == '__main__':
    unittest.main()
