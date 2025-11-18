"""
Reinforcement Learning Trading Strategy using Q-Learning

This module implements a Q-learning based trading strategy that:
1. Uses CCI (Commodity Channel Index) to define market states
2. Learns optimal trading actions (Long/Neutral/Short) for each state
3. Optimizes for maximum cumulative returns with transaction costs

Author: Enhanced Implementation
"""

import numpy as np
import pandas as pd
import talib
from sys import argv
from typing import Tuple, Dict


class QLearningTrader:
    """
    Q-Learning based trading agent.

    Actions:
        -1: Short position
         0: Neutral (no position)
         1: Long position

    States: Discretized based on CCI values
    """

    def __init__(self, train_ratio: float = 0.7, cci_period: int = 14,
                 alpha: float = 0.4, gamma: float = 0.9,
                 transaction_cost: float = 0.001, epsilon: float = 1e-10):
        """
        Initialize Q-Learning trader.

        Args:
            train_ratio: Ratio of data to use for training
            cci_period: Period for CCI calculation
            alpha: Learning rate
            gamma: Discount factor
            transaction_cost: Transaction cost as fraction of position
            epsilon: Convergence threshold for Q-value updates
        """
        self.train_ratio = train_ratio
        self.cci_period = cci_period
        self.alpha = alpha
        self.gamma = gamma
        self.transaction_cost = transaction_cost
        self.epsilon = epsilon

        # Action space: -1 (short), 0 (neutral), 1 (long)
        self.actions = np.array([-1, 0, 1])
        self.n_actions = len(self.actions)

        # State space: discretized CCI values
        self.state_boundaries = np.arange(-1000, 1001, 100)
        self.n_states = len(self.state_boundaries)

        # Q-table
        self.Q = np.zeros((self.n_states, self.n_actions))

    def discretize_state(self, cci_value: float) -> int:
        """Convert continuous CCI value to discrete state."""
        if cci_value < self.state_boundaries[0]:
            return 0
        elif cci_value >= self.state_boundaries[-1]:
            return self.n_states - 1
        else:
            for i in range(1, len(self.state_boundaries)):
                if self.state_boundaries[i-1] <= cci_value < self.state_boundaries[i]:
                    return i
        return 0

    def calculate_reward(self, prev_price: float, curr_price: float,
                        prev_action: int, curr_action: int) -> float:
        """
        Calculate reward for taking action.

        Reward = Position profit/loss - Transaction costs

        Args:
            prev_price: Previous close price
            curr_price: Current close price
            prev_action: Previous action (-1, 0, 1)
            curr_action: Current action (-1, 0, 1)

        Returns:
            Reward value
        """
        # Price change
        price_return = (curr_price - prev_price) / prev_price

        # Profit/loss based on position
        pnl = prev_action * price_return

        # Transaction cost when changing positions
        position_change = abs(curr_action - prev_action)
        transaction_cost = position_change * self.transaction_cost

        return pnl - transaction_cost

    def get_optimal_action(self, state: int, explore: bool = False) -> int:
        """
        Get optimal action for given state.

        Args:
            state: Current state index
            explore: If True, use epsilon-greedy exploration

        Returns:
            Action index (0, 1, or 2 mapping to -1, 0, 1)
        """
        state = int(state)
        if state >= self.n_states:
            state = self.n_states - 1

        q_values = self.Q[state, :]
        max_q = np.max(q_values)

        # Find all actions with maximum Q-value
        best_actions = np.where(q_values == max_q)[0]

        # Random tie-breaking
        return np.random.choice(best_actions)

    def update_q_value(self, state: int, action_idx: int, reward: float,
                      next_state: int, next_action_idx: int) -> float:
        """
        Update Q-value using SARSA update rule.

        Q(s,a) = Q(s,a) + α * [R + γ * Q(s',a') - Q(s,a)]

        Args:
            state: Current state
            action_idx: Current action index
            reward: Reward received
            next_state: Next state
            next_action_idx: Next action index

        Returns:
            Updated Q-value
        """
        state = int(state)
        next_state = int(next_state)

        if state >= self.n_states:
            state = self.n_states - 1
        if next_state >= self.n_states:
            next_state = self.n_states - 1

        current_q = self.Q[state, action_idx]
        next_q = self.Q[next_state, next_action_idx]

        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        return new_q

    def train(self, prices: pd.DataFrame) -> Dict:
        """
        Train Q-learning model on price data.

        Args:
            prices: DataFrame with OPEN, HIGH, LOW, CLOSE columns

        Returns:
            Dictionary with training statistics
        """
        # Calculate CCI
        cci = talib.CCI(prices['HIGH'], prices['LOW'], prices['CLOSE'],
                       timeperiod=self.cci_period)

        # Remove NaN values
        valid_idx = ~np.isnan(cci)
        cci = cci[valid_idx].values
        prices_clean = prices[valid_idx].reset_index(drop=True)

        # Discretize states
        states = np.array([self.discretize_state(c) for c in cci])

        # Split train/test
        n_total = len(prices_clean)
        n_train = int(n_total * self.train_ratio)

        train_prices = prices_clean.iloc[:n_train]
        train_states = states[:n_train]

        # Initialize action history
        actions = np.zeros(n_train, dtype=int)
        rewards = np.zeros(n_train)

        # Training loop (backward iteration for dynamic programming)
        print("Training Q-Learning model...")
        for epoch in range(10):  # Multiple epochs for convergence
            cumulative_reward = 0

            for t in range(n_train - 1):
                state = train_states[t]
                next_state = train_states[t + 1]

                # Get action for current state
                action_idx = self.get_optimal_action(state)
                next_action_idx = self.get_optimal_action(next_state)

                # Calculate reward
                if t > 0:
                    prev_action = self.actions[actions[t-1]]
                else:
                    prev_action = 0

                curr_action = self.actions[action_idx]

                reward = self.calculate_reward(
                    train_prices.iloc[t]['CLOSE'],
                    train_prices.iloc[t+1]['CLOSE'],
                    prev_action,
                    curr_action
                )

                # Update Q-value
                self.Q[state, action_idx] = self.update_q_value(
                    state, action_idx, reward, next_state, next_action_idx
                )

                actions[t] = action_idx
                rewards[t] = reward
                cumulative_reward += reward

            print(f"Epoch {epoch+1}/10, Cumulative Reward: {cumulative_reward:.4f}")

        return {
            'final_reward': cumulative_reward,
            'avg_reward': np.mean(rewards),
            'Q_table': self.Q.copy()
        }

    def test(self, prices: pd.DataFrame) -> Dict:
        """
        Test trained model on new data.

        Args:
            prices: DataFrame with OPEN, HIGH, LOW, CLOSE columns

        Returns:
            Dictionary with test statistics
        """
        # Calculate CCI
        cci = talib.CCI(prices['HIGH'], prices['LOW'], prices['CLOSE'],
                       timeperiod=self.cci_period)

        # Remove NaN values
        valid_idx = ~np.isnan(cci)
        cci = cci[valid_idx].values
        prices_clean = prices[valid_idx].reset_index(drop=True)

        # Discretize states
        states = np.array([self.discretize_state(c) for c in cci])

        n_test = len(prices_clean)
        actions = np.zeros(n_test, dtype=int)
        rewards = np.zeros(n_test)

        cumulative_reward = 0

        for t in range(n_test - 1):
            state = states[t]
            action_idx = self.get_optimal_action(state)

            if t > 0:
                prev_action = self.actions[actions[t-1]]
            else:
                prev_action = 0

            curr_action = self.actions[action_idx]

            reward = self.calculate_reward(
                prices_clean.iloc[t]['CLOSE'],
                prices_clean.iloc[t+1]['CLOSE'],
                prev_action,
                curr_action
            )

            actions[t] = action_idx
            rewards[t] = reward
            cumulative_reward += reward

        return {
            'cumulative_reward': cumulative_reward,
            'avg_reward': np.mean(rewards),
            'total_trades': np.sum(np.diff(actions) != 0),
            'actions': actions,
            'rewards': rewards
        }


def main():
    """Main execution function."""
    if len(argv) < 2:
        print("Usage: python rl.py <price_data.csv>")
        return

    # Load data
    print(f"Loading data from {argv[1]}...")
    prices = pd.read_csv(argv[1])
    prices.dropna(inplace=True)

    # Split data
    n_total = len(prices)
    train_ratio = 0.7
    n_train = int(n_total * train_ratio)

    train_data = prices.iloc[:n_train].reset_index(drop=True)
    test_data = prices.iloc[n_train:].reset_index(drop=True)

    # Initialize and train model
    trader = QLearningTrader(
        train_ratio=1.0,  # Use all of train_data
        cci_period=14,
        alpha=0.4,
        gamma=0.9,
        transaction_cost=0.001
    )

    print("\n=== Training Phase ===")
    train_results = trader.train(train_data)
    print(f"Training completed. Final reward: {train_results['final_reward']:.4f}")

    print("\n=== Testing Phase ===")
    test_results = trader.test(test_data)
    print(f"Test cumulative reward: {test_results['cumulative_reward']:.4f}")
    print(f"Test average reward: {test_results['avg_reward']:.6f}")
    print(f"Total trades executed: {test_results['total_trades']}")

    # Save Q-table
    np.save('q_table.npy', trader.Q)
    print("\nQ-table saved to q_table.npy")


if __name__ == '__main__':
    main()








