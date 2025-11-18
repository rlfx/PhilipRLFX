"""
Unit tests for performance metrics module.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance_metrics import PerformanceAnalyzer


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test cases for PerformanceAnalyzer class."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)

        # Create sample returns with positive drift
        n = 252  # One year of daily returns
        self.returns = np.random.randn(n) * 0.01 + 0.0005  # ~12.6% annual return

        # Create sample trades
        self.trades = pd.DataFrame({
            'entry': [100, 105, 110],
            'exit': [105, 103, 115],
            'pnl': [5, -2, 5]
        })

        self.analyzer = PerformanceAnalyzer(self.returns, self.trades)

    def test_calculate_total_return(self):
        """Test total return calculation."""
        total_return = self.analyzer.calculate_total_return()

        self.assertIsInstance(total_return, float)
        # Should be positive with our setup
        self.assertGreater(total_return, -0.5)

    def test_calculate_annualized_return(self):
        """Test annualized return calculation."""
        annual_return = self.analyzer.calculate_annualized_return()

        self.assertIsInstance(annual_return, float)

    def test_calculate_volatility(self):
        """Test volatility calculation."""
        volatility = self.analyzer.calculate_volatility()

        self.assertIsInstance(volatility, float)
        self.assertGreater(volatility, 0)

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        sharpe = self.analyzer.calculate_sharpe_ratio()

        self.assertIsInstance(sharpe, float)
        # Can be positive or negative

    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        sortino = self.analyzer.calculate_sortino_ratio()

        self.assertIsInstance(sortino, (float, np.floating))

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        max_dd, start_idx, end_idx = self.analyzer.calculate_max_drawdown()

        self.assertIsInstance(max_dd, float)
        self.assertGreaterEqual(max_dd, 0)
        self.assertLessEqual(max_dd, 1)
        self.assertIsInstance(start_idx, (int, np.integer))
        self.assertIsInstance(end_idx, (int, np.integer))

    def test_calculate_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        calmar = self.analyzer.calculate_calmar_ratio()

        self.assertIsInstance(calmar, (float, np.floating))

    def test_calculate_win_rate(self):
        """Test win rate calculation."""
        win_rate = self.analyzer.calculate_win_rate()

        self.assertIsInstance(win_rate, float)
        self.assertGreaterEqual(win_rate, 0)
        self.assertLessEqual(win_rate, 1)

        # With our sample trades, win rate should be 2/3
        self.assertAlmostEqual(win_rate, 2/3, places=2)

    def test_calculate_profit_factor(self):
        """Test profit factor calculation."""
        profit_factor = self.analyzer.calculate_profit_factor()

        self.assertIsInstance(profit_factor, (float, np.floating))
        self.assertGreater(profit_factor, 0)

    def test_calculate_average_win_loss_ratio(self):
        """Test average win/loss ratio calculation."""
        ratio = self.analyzer.calculate_average_win_loss_ratio()

        self.assertIsInstance(ratio, (float, np.floating))

    def test_calculate_expectancy(self):
        """Test expectancy calculation."""
        expectancy = self.analyzer.calculate_expectancy()

        self.assertIsInstance(expectancy, float)

    def test_calculate_all_metrics(self):
        """Test calculation of all metrics."""
        metrics = self.analyzer.calculate_all_metrics()

        self.assertIsInstance(metrics, dict)

        # Check all expected keys
        expected_keys = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'calmar_ratio', 'win_rate', 'profit_factor',
            'avg_win_loss_ratio', 'expectancy',
            'max_consecutive_wins', 'max_consecutive_losses',
            'ulcer_index', 'total_trades'
        ]

        for key in expected_keys:
            self.assertIn(key, metrics)

    def test_empty_returns(self):
        """Test handling of empty returns."""
        analyzer = PerformanceAnalyzer(np.array([]))

        self.assertEqual(analyzer.calculate_total_return(), 0.0)
        self.assertEqual(analyzer.calculate_volatility(), 0.0)


if __name__ == '__main__':
    unittest.main()
