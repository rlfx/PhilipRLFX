"""
Ensemble Trading Strategy Module

Combines multiple trading signals using various ensemble methods:
- Voting (majority vote)
- Weighted average
- Stacking (meta-learning)
- Confidence-based selection

Usage:
    from ensemble_strategy import EnsembleStrategy

    ensemble = EnsembleStrategy()
    signal = ensemble.get_ensemble_signal(prices)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from indicators import TechnicalIndicators


class EnsembleStrategy:
    """
    Ensemble trading strategy that combines multiple indicators and strategies.
    """

    def __init__(self, method: str = 'weighted', weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble strategy.

        Args:
            method: Ensemble method ('voting', 'weighted', 'adaptive')
            weights: Optional weights for each signal (used in weighted method)
        """
        self.method = method
        self.weights = weights or {}

        # Default weights if not provided
        if method == 'weighted' and not self.weights:
            self.weights = {
                'rsi': 1.0,
                'macd': 1.5,
                'stochastic': 1.0,
                'bollinger': 1.2,
                'trend': 2.0
            }

    def majority_voting(self, signals: Dict[str, int]) -> int:
        """
        Get trading signal using majority voting.

        Args:
            signals: Dictionary of signal_name -> signal_value (-1, 0, 1)

        Returns:
            Ensemble signal (-1, 0, 1)
        """
        signal_values = list(signals.values())

        if not signal_values:
            return 0

        # Count votes for each action
        buy_votes = signal_values.count(1)
        sell_votes = signal_values.count(-1)
        neutral_votes = signal_values.count(0)

        # Return majority
        if buy_votes > sell_votes and buy_votes > neutral_votes:
            return 1
        elif sell_votes > buy_votes and sell_votes > neutral_votes:
            return -1
        else:
            return 0

    def weighted_average(self, signals: Dict[str, int], weights: Dict[str, float]) -> int:
        """
        Get trading signal using weighted average.

        Args:
            signals: Dictionary of signal_name -> signal_value
            weights: Dictionary of signal_name -> weight

        Returns:
            Ensemble signal (-1, 0, 1)
        """
        if not signals:
            return 0

        weighted_sum = 0.0
        total_weight = 0.0

        for signal_name, signal_value in signals.items():
            weight = weights.get(signal_name, 1.0)
            weighted_sum += signal_value * weight
            total_weight += weight

        if total_weight == 0:
            return 0

        avg_signal = weighted_sum / total_weight

        # Convert to discrete signal
        if avg_signal > 0.3:
            return 1
        elif avg_signal < -0.3:
            return -1
        else:
            return 0

    def adaptive_weighting(self, signals: Dict[str, int],
                          performance_history: Dict[str, List[float]]) -> int:
        """
        Adaptive weighting based on recent performance of each indicator.

        Args:
            signals: Current signals from each indicator
            performance_history: Historical performance of each indicator

        Returns:
            Ensemble signal (-1, 0, 1)
        """
        # Calculate weights based on recent performance
        adaptive_weights = {}

        for signal_name in signals.keys():
            if signal_name in performance_history and len(performance_history[signal_name]) > 0:
                # Use recent performance (e.g., last 20 signals)
                recent_performance = performance_history[signal_name][-20:]
                avg_performance = np.mean(recent_performance)
                # Weight proportional to performance (with minimum threshold)
                adaptive_weights[signal_name] = max(0.1, avg_performance)
            else:
                adaptive_weights[signal_name] = 1.0

        return self.weighted_average(signals, adaptive_weights)

    def confidence_based_selection(self, signals: Dict[str, int],
                                  confidences: Dict[str, float]) -> int:
        """
        Select signal based on confidence scores.

        Args:
            signals: Dictionary of signals
            confidences: Dictionary of confidence scores (0.0 to 1.0)

        Returns:
            Signal from the most confident indicator
        """
        if not signals or not confidences:
            return 0

        # Find indicator with highest confidence
        max_confidence = -1
        best_signal = 0

        for signal_name, confidence in confidences.items():
            if confidence > max_confidence and signal_name in signals:
                max_confidence = confidence
                best_signal = signals[signal_name]

        return best_signal

    def get_ensemble_signal(self, prices: pd.DataFrame,
                           performance_history: Optional[Dict[str, List[float]]] = None,
                           confidences: Optional[Dict[str, float]] = None) -> int:
        """
        Get ensemble trading signal from multiple indicators.

        Args:
            prices: DataFrame with OHLC data
            performance_history: Optional performance history for adaptive weighting
            confidences: Optional confidence scores for each indicator

        Returns:
            Ensemble signal (-1, 0, 1)
        """
        # Calculate signals from all indicators
        ti = TechnicalIndicators(prices)
        signals = ti.get_all_signals()

        # Add trend signal
        from indicators import detect_trend
        trend = detect_trend(prices)
        if trend == 'uptrend':
            signals['trend'] = 1
        elif trend == 'downtrend':
            signals['trend'] = -1
        else:
            signals['trend'] = 0

        # Apply ensemble method
        if self.method == 'voting':
            return self.majority_voting(signals)

        elif self.method == 'weighted':
            return self.weighted_average(signals, self.weights)

        elif self.method == 'adaptive' and performance_history:
            return self.adaptive_weighting(signals, performance_history)

        elif self.method == 'confidence' and confidences:
            return self.confidence_based_selection(signals, confidences)

        else:
            # Default to voting
            return self.majority_voting(signals)


class StackingEnsemble:
    """
    Stacking ensemble that uses a meta-learner to combine predictions.
    """

    def __init__(self):
        """Initialize stacking ensemble."""
        self.meta_model = None
        self.base_indicators = ['rsi', 'macd', 'stochastic', 'bollinger']

    def prepare_features(self, prices: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix from base indicators.

        Args:
            prices: DataFrame with OHLC data

        Returns:
            Feature matrix (n_samples, n_features)
        """
        ti = TechnicalIndicators(prices)

        # Get all indicator values (not just signals)
        features = []

        # RSI
        rsi = ti.calculate_rsi()
        features.append(rsi)

        # MACD
        macd, signal, hist = ti.calculate_macd()
        features.extend([macd, signal, hist])

        # Stochastic
        k, d = ti.calculate_stochastic()
        features.extend([k, d])

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = ti.calculate_bollinger_bands()
        bb_width = (bb_upper - bb_lower) / bb_mid
        bb_position = (prices['CLOSE'].values - bb_lower) / (bb_upper - bb_lower)
        features.extend([bb_width, bb_position])

        # Stack features
        feature_matrix = np.column_stack(features)

        return feature_matrix

    def train_meta_model(self, X: np.ndarray, y: np.ndarray):
        """
        Train meta-model on base indicator predictions.

        Args:
            X: Feature matrix from base indicators
            y: Target labels (1 = buy, -1 = sell, 0 = hold)
        """
        # Simple logistic regression as meta-model
        # In practice, you could use more sophisticated models
        from sklearn.linear_model import LogisticRegression

        self.meta_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
        self.meta_model.fit(X, y)

    def predict(self, prices: pd.DataFrame) -> int:
        """
        Predict trading signal using trained meta-model.

        Args:
            prices: DataFrame with OHLC data

        Returns:
            Trading signal (-1, 0, 1)
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Call train_meta_model first.")

        features = self.prepare_features(prices)
        prediction = self.meta_model.predict(features[-1:])

        return int(prediction[0])


class MultiTimeframeEnsemble:
    """
    Ensemble strategy across multiple timeframes.
    """

    def __init__(self, timeframes: List[int] = [20, 50, 100]):
        """
        Initialize multi-timeframe ensemble.

        Args:
            timeframes: List of lookback periods (e.g., [short, medium, long])
        """
        self.timeframes = timeframes

    def get_signal_for_timeframe(self, prices: pd.DataFrame, lookback: int) -> int:
        """
        Get signal for specific timeframe.

        Args:
            prices: Full price DataFrame
            lookback: Lookback period

        Returns:
            Signal for this timeframe
        """
        if len(prices) < lookback:
            return 0

        # Use only recent data for this timeframe
        recent_prices = prices.tail(lookback)

        ti = TechnicalIndicators(recent_prices)
        signals = ti.get_all_signals()

        # Simple majority vote for this timeframe
        signal_values = list(signals.values())
        buy_votes = signal_values.count(1)
        sell_votes = signal_values.count(-1)

        if buy_votes > sell_votes:
            return 1
        elif sell_votes > buy_votes:
            return -1
        else:
            return 0

    def get_ensemble_signal(self, prices: pd.DataFrame) -> int:
        """
        Get ensemble signal across all timeframes.

        Args:
            prices: DataFrame with OHLC data

        Returns:
            Ensemble signal (-1, 0, 1)
        """
        signals = []

        for timeframe in self.timeframes:
            signal = self.get_signal_for_timeframe(prices, timeframe)
            signals.append(signal)

        # All timeframes must agree for strong signal
        if all(s == 1 for s in signals):
            return 1
        elif all(s == -1 for s in signals):
            return -1
        # Majority vote
        elif signals.count(1) > signals.count(-1):
            return 1 if signals.count(1) >= 2 else 0
        elif signals.count(-1) > signals.count(1):
            return -1 if signals.count(-1) >= 2 else 0
        else:
            return 0


def create_ensemble(method: str = 'weighted', **kwargs) -> EnsembleStrategy:
    """
    Factory function to create ensemble strategy.

    Args:
        method: Ensemble method ('voting', 'weighted', 'adaptive', 'stacking', 'multi_timeframe')
        **kwargs: Additional arguments for specific ensemble types

    Returns:
        Ensemble strategy instance
    """
    if method == 'stacking':
        return StackingEnsemble()
    elif method == 'multi_timeframe':
        timeframes = kwargs.get('timeframes', [20, 50, 100])
        return MultiTimeframeEnsemble(timeframes)
    else:
        weights = kwargs.get('weights', None)
        return EnsembleStrategy(method=method, weights=weights)
