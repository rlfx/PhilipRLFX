"""
Unit tests for Q-Learning trading module.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl import QLearningTrader


class TestQLearningTrader(unittest.TestCase):
    """Test cases for QLearningTrader class."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)

        # Create sample price data
        n = 200
        dates = pd.date_range('2020-01-01', periods=n)
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_price = close + np.random.randn(n) * 0.5

        self.data = pd.DataFrame({
            'DATE': dates,
            'OPEN': open_price,
            'HIGH': high,
            'LOW': low,
            'CLOSE': close
        })

        self.trader = QLearningTrader(
            train_ratio=0.7,
            cci_period=14,
            alpha=0.4,
            gamma=0.9,
            transaction_cost=0.001
        )

    def test_initialization(self):
        """Test trader initialization."""
        self.assertEqual(self.trader.train_ratio, 0.7)
        self.assertEqual(self.trader.cci_period, 14)
        self.assertEqual(self.trader.alpha, 0.4)
        self.assertEqual(self.trader.gamma, 0.9)
        self.assertEqual(len(self.trader.actions), 3)
        self.assertEqual(self.trader.n_states, 21)

    def test_discretize_state(self):
        """Test state discretization."""
        # Test boundary conditions
        state1 = self.trader.discretize_state(-1500)
        self.assertEqual(state1, 0)

        state2 = self.trader.discretize_state(1500)
        self.assertEqual(state2, 20)

        state3 = self.trader.discretize_state(0)
        self.assertGreaterEqual(state3, 0)
        self.assertLessEqual(state3, 20)

    def test_calculate_reward(self):
        """Test reward calculation."""
        # Long position, price goes up - positive reward
        reward1 = self.trader.calculate_reward(
            prev_price=100,
            curr_price=101,
            prev_action=1,  # long
            curr_action=1   # stay long
        )
        self.assertGreater(reward1, 0)

        # Long position, price goes down - negative reward
        reward2 = self.trader.calculate_reward(
            prev_price=100,
            curr_price=99,
            prev_action=1,
            curr_action=1
        )
        self.assertLess(reward2, 0)

        # Transaction cost when changing position
        reward3 = self.trader.calculate_reward(
            prev_price=100,
            curr_price=100,
            prev_action=0,
            curr_action=1  # entering position
        )
        self.assertLess(reward3, 0)  # Should have transaction cost

    def test_get_optimal_action(self):
        """Test optimal action selection."""
        action = self.trader.get_optimal_action(state=5)

        self.assertIn(action, [0, 1, 2])

    def test_update_q_value(self):
        """Test Q-value update."""
        initial_q = self.trader.Q[5, 1]

        new_q = self.trader.update_q_value(
            state=5,
            action_idx=1,
            reward=0.01,
            next_state=6,
            next_action_idx=1
        )

        self.assertIsInstance(new_q, float)

    def test_train(self):
        """Test training method."""
        results = self.trader.train(self.data)

        self.assertIsInstance(results, dict)
        self.assertIn('final_reward', results)
        self.assertIn('avg_reward', results)
        self.assertIn('Q_table', results)

        # Q-table should have correct shape
        self.assertEqual(results['Q_table'].shape, (21, 3))

    def test_test(self):
        """Test testing method."""
        # Train first
        self.trader.train(self.data)

        # Then test
        results = self.trader.test(self.data)

        self.assertIsInstance(results, dict)
        self.assertIn('cumulative_reward', results)
        self.assertIn('avg_reward', results)
        self.assertIn('total_trades', results)
        self.assertIn('actions', results)
        self.assertIn('rewards', results)


if __name__ == '__main__':
    unittest.main()
