"""
Unit tests for technical indicators module.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import TechnicalIndicators, calculate_support_resistance, detect_trend


class TestTechnicalIndicators(unittest.TestCase):
    """Test cases for TechnicalIndicators class."""

    def setUp(self):
        """Set up test data."""
        # Create sample price data
        np.random.seed(42)
        n = 100

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

        self.ti = TechnicalIndicators(self.data)

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        rsi = self.ti.calculate_rsi(period=14)

        self.assertIsNotNone(rsi)
        self.assertEqual(len(rsi), len(self.data))

        # RSI should be between 0 and 100
        valid_rsi = rsi[~np.isnan(rsi)]
        self.assertTrue(np.all(valid_rsi >= 0))
        self.assertTrue(np.all(valid_rsi <= 100))

    def test_calculate_macd(self):
        """Test MACD calculation."""
        macd, signal, hist = self.ti.calculate_macd()

        self.assertIsNotNone(macd)
        self.assertIsNotNone(signal)
        self.assertIsNotNone(hist)

        self.assertEqual(len(macd), len(self.data))
        self.assertEqual(len(signal), len(self.data))
        self.assertEqual(len(hist), len(self.data))

    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = self.ti.calculate_bollinger_bands()

        self.assertEqual(len(upper), len(self.data))
        self.assertEqual(len(middle), len(self.data))
        self.assertEqual(len(lower), len(self.data))

        # Upper should be > middle > lower
        valid_idx = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        self.assertTrue(np.all(upper[valid_idx] >= middle[valid_idx]))
        self.assertTrue(np.all(middle[valid_idx] >= lower[valid_idx]))

    def test_calculate_atr(self):
        """Test ATR calculation."""
        atr = self.ti.calculate_atr(period=14)

        self.assertIsNotNone(atr)
        self.assertEqual(len(atr), len(self.data))

        # ATR should be positive
        valid_atr = atr[~np.isnan(atr)]
        self.assertTrue(np.all(valid_atr >= 0))

    def test_calculate_stochastic(self):
        """Test Stochastic oscillator calculation."""
        k, d = self.ti.calculate_stochastic()

        self.assertEqual(len(k), len(self.data))
        self.assertEqual(len(d), len(self.data))

        # Stochastic should be between 0 and 100
        valid_k = k[~np.isnan(k)]
        valid_d = d[~np.isnan(d)]
        self.assertTrue(np.all(valid_k >= 0))
        self.assertTrue(np.all(valid_k <= 100))
        self.assertTrue(np.all(valid_d >= 0))
        self.assertTrue(np.all(valid_d <= 100))

    def test_get_rsi_signal(self):
        """Test RSI signal generation."""
        signal = self.ti.get_rsi_signal()

        self.assertIn(signal, [-1, 0, 1])

    def test_get_macd_signal(self):
        """Test MACD signal generation."""
        signal = self.ti.get_macd_signal()

        self.assertIn(signal, [-1, 0, 1])

    def test_get_all_signals(self):
        """Test getting all signals."""
        signals = self.ti.get_all_signals()

        self.assertIsInstance(signals, dict)
        self.assertIn('rsi', signals)
        self.assertIn('macd', signals)
        self.assertIn('stochastic', signals)
        self.assertIn('bollinger', signals)

        for signal in signals.values():
            self.assertIn(signal, [-1, 0, 1])

    def test_get_composite_signal(self):
        """Test composite signal calculation."""
        composite = self.ti.get_composite_signal()

        self.assertIsInstance(composite, float)
        self.assertGreaterEqual(composite, -1.0)
        self.assertLessEqual(composite, 1.0)

    def test_support_resistance(self):
        """Test support and resistance calculation."""
        support, resistance = calculate_support_resistance(self.data, window=20)

        self.assertIsInstance(support, float)
        self.assertIsInstance(resistance, float)
        self.assertLess(support, resistance)

    def test_detect_trend(self):
        """Test trend detection."""
        trend = detect_trend(self.data)

        self.assertIn(trend, ['uptrend', 'downtrend', 'sideways'])


if __name__ == '__main__':
    unittest.main()
