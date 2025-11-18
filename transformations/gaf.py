"""Gramian Angular Field (GAF) for time series transformation.

This module implements GAF, a technique to convert time series data into
2D images using Gramian angular matrices, suitable for deep learning.

Functions:
    - GramianAngularField: Main transformation function
    - output_miscellaneous: Save generated GAF images
"""

import errno
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange


def output_miscellaneous(features: np.ndarray, output_dir: str = "gaf_misc") -> None:
    """Generate and save GAF images.

    Args:
        features: Feature array from GramianAngularField
        output_dir: Directory to save images
    """
    # Create output directory
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Save images
    for index in trange(features.shape[0], desc="Drawing..."):
        # Normalize to 0-255 range for image
        img_array = features[index]
        img_normalized = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        img = Image.fromarray(img_normalized, mode='L')
        img.save(f'{output_dir}/{index:04d}.png')


def gramian_angular_field(
    all_ts: List[float],
    window_size: int,
    rolling_length: int,
    method: str = 'summation',
    scale: str = '[0,1]'
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert time series to Gramian Angular Field.

    Args:
        all_ts: Time series data (1D array)
        window_size: Window size for feature extraction
        rolling_length: Step size for rolling window
        method: 'summation' for GASF or 'difference' for GADF
        scale: '[0,1]' or '[-1,1]' for scaling

    Returns:
        Tuple of (gramian_field, prices)
    """
    # Get time series length
    n = len(all_ts)

    # Moving window size (2x the window size to avoid micro-magnification)
    moving_window_size = window_size * 2

    # Calculate number of rolling windows
    n_rolling_data = int(np.floor((n - moving_window_size) / rolling_length))

    # Initialize lists for features and prices
    gramian_field = []
    prices = []

    # Extract features from rolling windows
    for i_rolling_data in trange(n_rolling_data, desc="Extracting..."):
        # Starting position
        start_flag = i_rolling_data * rolling_length

        # Get window data
        full_window_data = list(all_ts[start_flag : start_flag + moving_window_size])

        # Store prices for visualization
        prices.append(full_window_data[-window_size:])

        # Normalize time series for cos/sin operations
        min_ts = np.min(full_window_data)
        max_ts = np.max(full_window_data)

        if scale == '[0,1]':
            rescaled_ts = (np.array(full_window_data) - min_ts) / (max_ts - min_ts)
        elif scale == '[-1,1]':
            rescaled_ts = (2 * np.array(full_window_data) - max_ts - min_ts) / (max_ts - min_ts)
        else:
            rescaled_ts = np.array(full_window_data)

        # Keep only the last window_size values
        rescaled_ts = rescaled_ts[-window_size:]

        # Calculate Gramian Angular Matrix
        this_gam = np.zeros((window_size, window_size), dtype=float)
        sin_ts = np.sqrt(np.clip(1 - rescaled_ts ** 2, 0, 1))

        if method == 'summation':
            # GASF: cos(x1 + x2) = cos(x1)cos(x2) - sin(x1)sin(x2)
            this_gam = np.outer(rescaled_ts, rescaled_ts) - np.outer(sin_ts, sin_ts)
        elif method == 'difference':
            # GADF: sin(x1 - x2) = sin(x1)cos(x2) - cos(x1)sin(x2)
            this_gam = np.outer(sin_ts, rescaled_ts) - np.outer(rescaled_ts, sin_ts)

        gramian_field.append(this_gam)

    return np.array(gramian_field), np.array(prices)


def main():
    """Main entry point for GAF feature extraction."""
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python gaf.py <csv_file>")
        sys.exit(1)

    # Load data
    data = pd.read_csv(sys.argv[1])
    data.dropna(inplace=True)

    # Configuration for GAF extraction
    window_size = 100
    rolling_length = 10

    # Extract Gramian Angular Field features
    features, prices = gramian_angular_field(
        all_ts=data['CLOSE'],
        window_size=window_size,
        rolling_length=rolling_length
    )

    # Save features to pickle files
    features.dump('gaf_Features.pkl')
    prices.dump('gaf_Prices.pkl')

    # Print shapes
    print(f'Features shape: {features.shape}')
    print(f'Prices shape: {prices.shape}')

    # Generate visualization images
    output_miscellaneous(features)


if __name__ == "__main__":
    main()
