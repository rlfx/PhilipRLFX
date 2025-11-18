"""
Advanced Technical Indicators Module

This module provides a comprehensive set of technical indicators for trading strategies,
including momentum, volatility, trend, and volume-based indicators.

Usage:
    from indicators import TechnicalIndicators

    ti = TechnicalIndicators(prices_df)
    signals = ti.get_all_signals()
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Tuple, Optional


class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator.

    Supports:
        - Trend: EMA, SMA, MACD, ADX
        - Momentum: RSI, Stochastic, CCI, Williams %R
        - Volatility: Bollinger Bands, ATR, Keltner Channels
        - Volume: OBV, MFI
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with price data.

        Args:
            data: DataFrame with columns: OPEN, HIGH, LOW, CLOSE, VOLUME (optional)
        """
        self.data = data.copy()
        self.high = data['HIGH'].values
        self.low = data['LOW'].values
        self.close = data['CLOSE'].values
        self.open = data['OPEN'].values
        self.volume = data['VOLUME'].values if 'VOLUME' in data.columns else None

    def calculate_rsi(self, period: int = 14) -> np.ndarray:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            period: Lookback period (default: 14)

        Returns:
            RSI values (0-100)
        """
        return talib.RSI(self.close, timeperiod=period)

    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        return talib.MACD(self.close, fastperiod=fast, slowperiod=slow, signalperiod=signal)

    def calculate_stochastic(self, k_period: int = 14, d_period: int = 3
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Stochastic Oscillator.

        Args:
            k_period: %K period
            d_period: %D period

        Returns:
            Tuple of (%K, %D)
        """
        slowk, slowd = talib.STOCH(
            self.high, self.low, self.close,
            fastk_period=k_period,
            slowk_period=d_period,
            slowd_period=d_period
        )
        return slowk, slowd

    def calculate_atr(self, period: int = 14) -> np.ndarray:
        """
        Calculate Average True Range (ATR) - volatility indicator.

        Args:
            period: Lookback period

        Returns:
            ATR values
        """
        return talib.ATR(self.high, self.low, self.close, timeperiod=period)

    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands.

        Args:
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        upper, middle, lower = talib.BBANDS(
            self.close,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=0
        )
        return upper, middle, lower

    def calculate_adx(self, period: int = 14) -> np.ndarray:
        """
        Calculate Average Directional Index (ADX) - trend strength.

        Args:
            period: Lookback period

        Returns:
            ADX values (0-100)
        """
        return talib.ADX(self.high, self.low, self.close, timeperiod=period)

    def calculate_cci(self, period: int = 14) -> np.ndarray:
        """
        Calculate Commodity Channel Index (CCI).

        Args:
            period: Lookback period

        Returns:
            CCI values
        """
        return talib.CCI(self.high, self.low, self.close, timeperiod=period)

    def calculate_williams_r(self, period: int = 14) -> np.ndarray:
        """
        Calculate Williams %R - momentum indicator.

        Args:
            period: Lookback period

        Returns:
            Williams %R values (-100 to 0)
        """
        return talib.WILLR(self.high, self.low, self.close, timeperiod=period)

    def calculate_obv(self) -> Optional[np.ndarray]:
        """
        Calculate On-Balance Volume (OBV).

        Returns:
            OBV values if volume data available, else None
        """
        if self.volume is None:
            return None
        return talib.OBV(self.close, self.volume)

    def calculate_mfi(self, period: int = 14) -> Optional[np.ndarray]:
        """
        Calculate Money Flow Index (MFI) - volume-weighted RSI.

        Args:
            period: Lookback period

        Returns:
            MFI values (0-100) if volume available, else None
        """
        if self.volume is None:
            return None
        return talib.MFI(self.high, self.low, self.close, self.volume, timeperiod=period)

    def calculate_keltner_channels(self, period: int = 20, atr_mult: float = 2.0
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Keltner Channels (volatility-based envelope).

        Args:
            period: EMA period
            atr_mult: ATR multiplier

        Returns:
            Tuple of (Upper channel, Middle EMA, Lower channel)
        """
        ema = talib.EMA(self.close, timeperiod=period)
        atr = self.calculate_atr(period)

        upper = ema + (atr * atr_mult)
        lower = ema - (atr * atr_mult)

        return upper, ema, lower

    def get_rsi_signal(self, period: int = 14, oversold: float = 30, overbought: float = 70
                       ) -> int:
        """
        Get trading signal from RSI.

        Args:
            period: RSI period
            oversold: Oversold threshold (default: 30)
            overbought: Overbought threshold (default: 70)

        Returns:
            1 (buy), -1 (sell), 0 (neutral)
        """
        rsi = self.calculate_rsi(period)
        current_rsi = rsi[-1]

        if np.isnan(current_rsi):
            return 0

        if current_rsi < oversold:
            return 1  # Oversold - buy signal
        elif current_rsi > overbought:
            return -1  # Overbought - sell signal
        else:
            return 0

    def get_macd_signal(self) -> int:
        """
        Get trading signal from MACD crossover.

        Returns:
            1 (bullish crossover), -1 (bearish crossover), 0 (no signal)
        """
        macd, signal, hist = self.calculate_macd()

        if len(hist) < 2:
            return 0

        # Bullish crossover: MACD crosses above signal
        if hist[-2] < 0 and hist[-1] > 0:
            return 1
        # Bearish crossover: MACD crosses below signal
        elif hist[-2] > 0 and hist[-1] < 0:
            return -1
        else:
            return 0

    def get_stochastic_signal(self, oversold: float = 20, overbought: float = 80) -> int:
        """
        Get trading signal from Stochastic oscillator.

        Args:
            oversold: Oversold threshold
            overbought: Overbought threshold

        Returns:
            1 (buy), -1 (sell), 0 (neutral)
        """
        k, d = self.calculate_stochastic()

        if len(k) < 2:
            return 0

        # Bullish crossover in oversold region
        if k[-1] < oversold and k[-2] < d[-2] and k[-1] > d[-1]:
            return 1
        # Bearish crossover in overbought region
        elif k[-1] > overbought and k[-2] > d[-2] and k[-1] < d[-1]:
            return -1
        else:
            return 0

    def get_bollinger_signal(self) -> int:
        """
        Get trading signal from Bollinger Bands.

        Returns:
            1 (price near lower band - buy), -1 (price near upper band - sell), 0 (neutral)
        """
        upper, middle, lower = self.calculate_bollinger_bands()

        if len(upper) == 0:
            return 0

        current_price = self.close[-1]
        bb_upper = upper[-1]
        bb_lower = lower[-1]
        bb_width = bb_upper - bb_lower

        # Buy when price is near lower band
        if current_price < bb_lower + 0.2 * bb_width:
            return 1
        # Sell when price is near upper band
        elif current_price > bb_upper - 0.2 * bb_width:
            return -1
        else:
            return 0

    def get_all_signals(self) -> Dict[str, int]:
        """
        Get all trading signals from different indicators.

        Returns:
            Dictionary with indicator names as keys and signals as values
        """
        signals = {
            'rsi': self.get_rsi_signal(),
            'macd': self.get_macd_signal(),
            'stochastic': self.get_stochastic_signal(),
            'bollinger': self.get_bollinger_signal()
        }

        return signals

    def get_composite_signal(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Get weighted composite signal from all indicators.

        Args:
            weights: Dictionary of indicator weights (default: equal weights)

        Returns:
            Composite signal score (-1 to 1)
        """
        signals = self.get_all_signals()

        if weights is None:
            weights = {k: 1.0 for k in signals.keys()}

        total_weight = sum(weights.values())
        weighted_sum = sum(signals[k] * weights[k] for k in signals.keys())

        return weighted_sum / total_weight if total_weight > 0 else 0.0


def calculate_support_resistance(prices: pd.DataFrame, window: int = 20
                                 ) -> Tuple[float, float]:
    """
    Calculate support and resistance levels using recent price action.

    Args:
        prices: DataFrame with HIGH, LOW columns
        window: Lookback window

    Returns:
        Tuple of (support_level, resistance_level)
    """
    recent_high = prices['HIGH'].tail(window)
    recent_low = prices['LOW'].tail(window)

    resistance = recent_high.max()
    support = recent_low.min()

    return support, resistance


def detect_trend(prices: pd.DataFrame, short_period: int = 20, long_period: int = 50
                ) -> str:
    """
    Detect market trend using moving average crossover.

    Args:
        prices: DataFrame with CLOSE column
        short_period: Short-term MA period
        long_period: Long-term MA period

    Returns:
        'uptrend', 'downtrend', or 'sideways'
    """
    close = prices['CLOSE'].values

    if len(close) < long_period:
        return 'sideways'

    sma_short = talib.SMA(close, timeperiod=short_period)
    sma_long = talib.SMA(close, timeperiod=long_period)

    if np.isnan(sma_short[-1]) or np.isnan(sma_long[-1]):
        return 'sideways'

    if sma_short[-1] > sma_long[-1] * 1.02:  # 2% buffer
        return 'uptrend'
    elif sma_short[-1] < sma_long[-1] * 0.98:
        return 'downtrend'
    else:
        return 'sideways'
