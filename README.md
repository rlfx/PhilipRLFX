# Advanced Trading Strategy Framework

A comprehensive trading strategy framework combining traditional technical analysis, machine learning, and reinforcement learning approaches.

## Overview

This framework provides:

- **Multiple Trading Strategies**: Bollinger Bands, MFE/MAE, Turtle Trading, Q-Learning
- **Advanced Technical Indicators**: RSI, MACD, Stochastic, ATR, Bollinger Bands, and more
- **Ensemble Methods**: Combine multiple signals using voting, weighted averaging, or meta-learning
- **Risk Management**: Position sizing, stop-loss, take-profit, drawdown limits
- **Performance Analysis**: Sharpe ratio, Sortino ratio, max drawdown, profit factor, and more
- **Machine Learning**: LeNet and AlexNet for time series classification
- **Time Series Transformations**: GAF (Gramian Angular Field) and MTF (Markov Transition Field)

## Project Structure

```
PhilipRLFX/
├── quantopians/              # Quantopian-based strategies
│   ├── quantopian_strat_bb.py      # Bollinger Bands strategy
│   ├── quantopian_strat_mfae.py    # MFE/MAE strategy
│   └── quantopian_strat_turtle.py  # Turtle trading strategy
├── classifications/          # Deep learning models
│   ├── lenet.py             # LeNet architecture
│   ├── alexnet.py           # AlexNet architecture
│   ├── use_lenet.py         # LeNet training script
│   └── use_alexnet.py       # AlexNet training script
├── transformations/          # Time series transformations
│   ├── gaf.py               # Gramian Angular Field
│   └── mtf.py               # Markov Transition Field
├── rl.py                    # Q-Learning reinforcement learning
├── indicators.py            # Advanced technical indicators
├── performance_metrics.py   # Performance analysis tools
├── ensemble_strategy.py     # Ensemble methods
├── risk_management.py       # Risk management utilities
└── tests/                   # Comprehensive test suite
```

## Installation

### Requirements

```bash
pip install numpy pandas talib scikit-learn keras tensorflow tflearn
```

### Data Format

Price data should be in CSV format with columns:
- `DATE`: Date/timestamp
- `OPEN`: Opening price
- `HIGH`: Highest price
- `LOW`: Lowest price
- `CLOSE`: Closing price
- `VOLUME`: Trading volume (optional)

## Usage Examples

### 1. Technical Indicators

```python
from indicators import TechnicalIndicators
import pandas as pd

# Load data
prices = pd.read_csv('EURUSD_1D.csv')

# Initialize indicators
ti = TechnicalIndicators(prices)

# Get individual indicators
rsi = ti.calculate_rsi(period=14)
macd, signal, hist = ti.calculate_macd()
bb_upper, bb_mid, bb_lower = ti.calculate_bollinger_bands()

# Get trading signals
signals = ti.get_all_signals()
print(signals)  # {'rsi': 1, 'macd': 0, 'stochastic': 1, 'bollinger': 0}

# Get composite signal
composite = ti.get_composite_signal()
```

### 2. Ensemble Strategy

```python
from ensemble_strategy import EnsembleStrategy
import pandas as pd

# Load data
prices = pd.read_csv('EURUSD_1D.csv')

# Create ensemble with weighted voting
ensemble = EnsembleStrategy(
    method='weighted',
    weights={'rsi': 1.0, 'macd': 1.5, 'bollinger': 1.2, 'trend': 2.0}
)

# Get ensemble signal
signal = ensemble.get_ensemble_signal(prices)
# Returns: 1 (buy), -1 (sell), or 0 (neutral)
```

### 3. Risk Management

```python
from risk_management import RiskManager

# Initialize risk manager
rm = RiskManager(
    account_balance=100000,
    max_risk_per_trade=0.02,      # 2% per trade
    max_portfolio_risk=0.06,       # 6% total
    max_drawdown=0.20              # 20% max drawdown
)

# Calculate position size
entry_price = 1.2000
stop_loss = 1.1950
position_size = rm.calculate_position_size_fixed_risk(entry_price, stop_loss)
print(f"Position size: {position_size:.2f} units")

# Calculate stop loss and take profit
from indicators import TechnicalIndicators
ti = TechnicalIndicators(prices)
atr = ti.calculate_atr()[-1]

stop_loss = rm.calculate_stop_loss_atr(entry_price, atr, multiplier=2.0, direction='long')
take_profit = rm.calculate_take_profit_atr(entry_price, atr, multiplier=3.0, direction='long')
```

### 4. Performance Analysis

```python
from performance_metrics import PerformanceAnalyzer
import numpy as np

# Calculate returns from your strategy
returns = np.array([0.01, -0.005, 0.02, 0.003, -0.01, ...])

# Initialize analyzer
analyzer = PerformanceAnalyzer(returns, risk_free_rate=0.02)

# Get all metrics
metrics = analyzer.calculate_all_metrics()

# Print formatted report
analyzer.print_report()
```

### 5. Q-Learning Reinforcement Learning

```python
from rl import QLearningTrader
import pandas as pd

# Load data
prices = pd.read_csv('EURUSD_1D.csv')

# Initialize Q-Learning trader
trader = QLearningTrader(
    train_ratio=0.7,
    cci_period=14,
    alpha=0.4,              # Learning rate
    gamma=0.9,              # Discount factor
    transaction_cost=0.001  # 0.1% transaction cost
)

# Train the model
train_results = trader.train(prices)
print(f"Training reward: {train_results['final_reward']:.4f}")

# Test the model
test_results = trader.test(prices)
print(f"Test reward: {test_results['cumulative_reward']:.4f}")
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_indicators.py

# Or run individual test files
python tests/test_indicators.py
python tests/test_performance_metrics.py
python tests/test_risk_management.py
python tests/test_rl.py
```

## Key Features

### Technical Indicators Module (`indicators.py`)

- **Trend Indicators**: EMA, SMA, MACD, ADX
- **Momentum Indicators**: RSI, Stochastic, CCI, Williams %R
- **Volatility Indicators**: Bollinger Bands, ATR, Keltner Channels
- **Volume Indicators**: OBV, MFI
- **Support/Resistance Detection**
- **Trend Detection**

### Ensemble Strategy Module (`ensemble_strategy.py`)

- **Voting Ensemble**: Majority vote across indicators
- **Weighted Average**: Custom weights for each indicator
- **Adaptive Weighting**: Weights adjust based on recent performance
- **Multi-Timeframe Ensemble**: Combine signals across timeframes

### Risk Management Module (`risk_management.py`)

- **Position Sizing**: Fixed risk, Kelly Criterion, Volatility targeting
- **Stop Loss Calculation**: ATR-based, Percentage-based, Trailing stops
- **Risk Metrics Tracking**: Drawdown, Portfolio risk, Position risk

### Performance Metrics Module (`performance_metrics.py`)

- **Return Analysis**: Total, annualized, risk-adjusted returns
- **Risk Metrics**: Volatility, drawdown, Ulcer Index
- **Ratio Metrics**: Sharpe, Sortino, Calmar ratios
- **Trade Statistics**: Win rate, profit factor, expectancy

## Enhancements Summary

### What Was Improved:

1. **Fixed Critical Bugs**:
   - Fixed syntax errors in `rl.py` (missing colons, MATLAB syntax)
   - Completed Q-Learning implementation with proper value functions
   - Fixed logic error in Bollinger Bands strategy
   - Fixed indentation issues

2. **Added New Capabilities**:
   - Comprehensive technical indicators module (9 indicators)
   - Ensemble strategy with 5 different methods
   - Advanced risk management with position sizing
   - Performance metrics module with 15+ metrics

3. **Testing & Documentation**:
   - Complete unit test suite (100+ tests)
   - Comprehensive README with usage examples
   - Detailed docstrings for all classes and methods

## License

MIT License