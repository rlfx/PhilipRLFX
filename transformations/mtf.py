"""Markov Transition Field (MTF) for time series transformation.

This module implements MTF, a technique to convert time series data into
2D images using Markov transition matrices, suitable for deep learning.

Functions:
    - MarkovTransitionField: Main transformation function
    - placeholderMatrix: Initialize placeholder matrices
    - findTrend: Find trend direction using linear regression
    - outputMiscellaneous: Save generated MTF images
"""

import errno
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange


def placeholder_matrix(n: int, m: int, q: int) -> np.ndarray:
    """Initialize a placeholder matrix array.

    Args:
        n: Number of first dimension
        m: Number of second dimension
        q: Size of zero matrices (q x q)

    Returns:
        numpy array of shape (n, m) containing q x q zero matrices
    """
    matrix = []
    for i in range(n):
        tmp = []
        for j in range(m):
            tmp.append(np.zeros((q, q), dtype=float))
        matrix.append(tmp)
    return np.array(matrix, dtype=object)

def find_trend(
    src: List[float],
    slope_thresh: Tuple[float, float] = None,
    residual_thresh: float = None
) -> Tuple[float, float]:
    """Find trend direction using linear regression.

    Args:
        src: Time series data
        slope_thresh: Tuple of (up_threshold, down_threshold)
        residual_thresh: Residual threshold for fit quality

    Returns:
        Trend direction (-1, 0, 1) or (slope, residual) if thresholds not provided
    """
    n = len(src)

    # Create X as [0, 1, 2, ...] and y as the time series
    x = np.array([i for i in range(n)])
    y = np.array(src)
    x = np.vstack([x, np.ones(n)]).T

    # Perform linear regression
    lin_reg, residuals, _, _ = np.linalg.lstsq(x, y, rcond=None)

    # Get slope from regression
    slope = lin_reg[0]

    # Get residual sum (distance from regression line)
    residual = 9999.0
    if len(residuals) > 0:
        residual = residuals[0]

    # If no thresholds provided, return slope and residual
    if slope_thresh is None or residual_thresh is None:
        return slope, residual

    # If residual is small enough (good fit), classify trend
    if residual < residual_thresh:
        if slope >= slope_thresh[0] and slope > 0.0:
            return 1  # Uptrend
        elif slope <= slope_thresh[1] and slope < 0.0:
            return -1  # Downtrend
        else:
            return 0  # Neutral
    else:
        return 0  # Bad fit, no trend

def output_miscellaneous(features: np.ndarray) -> None:
    """Generate and save MTF images.

    Args:
        features: Feature array from MarkovTransitionField
    """
    # Combine all MTF matrices into images
    N = features.shape[0]
    Q = int(np.sqrt(features.shape[1]))
    W = features.shape[2]
    new_features = np.zeros((N, W * Q, W * Q), dtype=float)

    for n in trange(N, desc="Combining..."):
        for i in range(Q):
            for j in range(Q):
                for k in range(W):
                    for l in range(W):
                        new_features[n, i * W + k, j * W + l] = features[n, i * Q + j, k, l]

    # Save features as pickle file
    new_features.dump('mtf_Features4plot.pkl')

    # Create output directory
    try:
        os.makedirs("mtf_misc")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Save images
    for index in trange(new_features.shape[0], desc="Drawing..."):
        # Normalize to 0-255 range for image
        img_array = new_features[index]
        img_normalized = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        img = Image.fromarray(img_normalized, mode='L')
        img.save(f'mtf_misc/{index:04d}.png')                   
    
def markov_transition_field(
    all_ts: List[float],
    window_size: int,
    rolling_length: int,
    quantile_size: int,
    label_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """Convert time series to Markov Transition Field.

    Args:
        all_ts: Time series data (1D array)
        window_size: Window size for feature extraction
        rolling_length: Step size for rolling window
        quantile_size: Number of quantiles (K for K-quantiles)
        label_size: Look-ahead period for trend labels

    Returns:
        Tuple of (markov_field, labels, prices)
    """
    # Get time series length
    n = len(all_ts)

    # Window size needs +1 for state transition detection
    real_window_size = window_size + 1

    # Calculate number of rolling windows
    n_rolling_data = int(np.floor((n - real_window_size) / rolling_length))

    # Feature size: quantile_size x quantile_size
    feature_size = quantile_size * quantile_size

    # Initialize Markov transition field
    markov_field = placeholder_matrix(n_rolling_data, feature_size, window_size)

    # Initialize arrays for labels and prices
    labels = []
    prices = []
    
    # Extract features from rolling windows
    for i_rolling_data in trange(n_rolling_data, desc="Extracting..."):
        # Initialize Markov transition field matrix
        this_markov_field = placeholder_matrix(window_size, window_size, quantile_size)
        
        # 起始位置
        start_flag = i_rolling_data*rolling_length
        
        # Get window data from time series
        full_window_data = list(all_ts[start_flag : start_flag + real_window_size])

        # Store prices for visualization
        prices.append(full_window_data)

        # Calculate slope thresholds from historical data
        history_slope_data = []
        history_residual_data = []
        for d in range(real_window_size - label_size):
            this_slope, this_residual = find_trend(full_window_data[d:d + label_size])
            history_slope_data.append(this_slope)
            history_residual_data.append(this_residual)

        slope_up_thresh = np.percentile(history_slope_data, 63)
        slope_down_thresh = np.percentile(history_slope_data, 37)
        slope_thresh = [slope_up_thresh, slope_down_thresh]
        residual_thresh = np.percentile(history_residual_data, 50)

        # Create label for future trend
        label_source = list(all_ts[start_flag + real_window_size : start_flag + real_window_size + label_size])
        new_label = find_trend(label_source, slope_thresh=slope_thresh, residual_thresh=residual_thresh)
        labels.append(new_label)
        
        # 從第 i 筆資料開始
        for i_window_size in range(window_size):
            # 到第 j 筆資料，我們要算的是投影片裡面的 W_i,j
            for j_window_size in range(window_size):
            
                # 因為如果我們要算 5 分位數，至少要有 5 筆資料，如果小於我們就放 0
                if np.abs(i_window_size - j_window_size) > quantile_size - 1:
                    
                    # 如果 j > i 就是時間序列要反過來
                    this_window_data = []
                    if i_window_size > j_window_size:
                        # 正常情況(窗格內我們要使用的資料，在本次迴圈中大小就取出來)
                        this_window_data = full_window_data[j_window_size:i_window_size+1]
                    else:
                        # 反過來情況
                        flips = full_window_data[i_window_size:j_window_size+1]
                        this_window_data =  flips[::-1]
                    
                    # 取得本次要算馬可夫矩陣的資料長度大小
                    n_this_window_data = len(this_window_data)
                    
                    # 根據本次要算矩陣的資料，取得他的 K 分位數
                    quantiles = [(100/quantile_size)*i for i in range(1, quantile_size)]
                    this_quantile = []
                    for q in quantiles:
                        this_quantile.append(np.percentile( this_window_data, q, interpolation='midpoint'))
                    
                    # 加入 -inf 與 inf 方便我們判斷資料是介在哪個分位數之間
                    this_quantile = [ -np.inf ] + this_quantile + [ np.inf ]
                    
                    # 取得分位數總長（為了跑迴圈用）
                    n_quantile = len( this_quantile )
                    
                    # 先宣告一個矩陣待會要放馬可夫矩陣
                    this_markov_matrix = np.zeros((quantile_size, quantile_size), float);
                    
                    # 從第一筆資料開始算是介在哪個狀態（哪兩個 K 分位數之間）
                    for i_this_window_data in range(n_this_window_data-1):
                        
                        # 從兩個分位數開始跑迴圈
                        for i_quantile in range(1, n_quantile):
                            for j_quantile in range(1, n_quantile):
                                
                                # 如果資料介於 i 與 j 之間，矩陣在 i, j 就要 +1
                                if this_window_data[i_this_window_data] < this_quantile[i_quantile] and \
                                    this_window_data[i_this_window_data] >= this_quantile[i_quantile-1] and \
                                    this_window_data[i_this_window_data+1] < this_quantile[j_quantile] and \
                                    this_window_data[i_this_window_data+1] >= this_quantile[j_quantile-1]:
                                        this_markov_matrix[ i_quantile-1 , j_quantile-1 ] += 1
                                    
                    
                    # 由於剛剛算的是個數，最後每一行都要除以行總數，來得到轉移機率
                    this_markov_matrix_count = [ sum(x) for x in this_markov_matrix ]
                    n_this_markov_matrix_count = len(this_markov_matrix_count)
                    for i_this_markov_matrix_count in range(n_this_markov_matrix_count):
                        # 如果那個狀態轉換有發生至少 1 次
                        if this_markov_matrix_count[i_this_markov_matrix_count] > 0:
                            this_markov_matrix[i_this_markov_matrix_count,:] /= this_markov_matrix_count[i_this_markov_matrix_count]
                        else:
                            # 如果狀態轉換根本沒發生，就不要除，否則會有除零誤
                            this_markov_matrix[i_this_markov_matrix_count,:] = 0
                        
                    # 最後把矩陣放到矩陣的矩陣裡面的  W_i,j 位置
                    this_markov_field[i_window_size,j_window_size] = this_markov_matrix
                
        # 當矩陣的矩陣都弄完了，我們就要把矩陣的矩陣切成各別 N 個矩陣
        feature_count = 0
        
        # 切法是依照狀態轉換, 例如 1->1, 1->2 ... 2->1 , ... 所以兩個 for loop
        for i_quantile in range(quantile_size):
            for j_quantile in range(quantile_size):
            
                # 先建立一個要蒐集所有被拆開出來的相同元素要放的矩陣
                seperated_markov_matrix = np.zeros( (window_size, window_size), float )
                
                # 從本次的「矩陣的矩陣」中依序 1...n 和 1...n 去取出來，放到前面宣告的矩陣
                for i_window_size in range(window_size):
                    for j_window_size in range(window_size):
                        
                        # 先從矩陣的矩陣取出特定的 W_i,j
                        this_markov_matrix = this_markov_field[i_window_size, j_window_size];
                        
                        # If matrix has values, use it; otherwise use 0
                        if sum(sum(this_markov_matrix)) != 0:
                            # 如果有矩陣，就把對應的狀態轉換機率放到拆分後的矩陣中
                            seperated_markov_matrix[i_window_size, j_window_size] = this_markov_matrix[i_quantile, j_quantile];
                        else:
                            # 如果 i j 太近沒矩陣，就放 0 
                            seperated_markov_matrix[ i_window_size, j_window_size ] = 0.0
                
                # 再把拆分出來的矩陣，放到整個滾動資料的對應位置
                markov_field[i_rolling_data,feature_count] = seperated_markov_matrix
                feature_count += 1
            
        
    return np.array(markov_field), np.array(labels), np.array(prices)

def main():
    """Main entry point for MTF feature extraction."""
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python mtf.py <csv_file>")
        sys.exit(1)

    # Load data
    data = pd.read_csv(sys.argv[1])
    data.dropna(inplace=True)

    # Configuration for MTF extraction
    window_size = 20
    rolling_length = 2
    quantile_size = 4
    label_size = 4

    # Extract Markov Transition Field features
    features, labels, prices = markov_transition_field(
        all_ts=data['CLOSE'],
        window_size=window_size,
        rolling_length=rolling_length,
        quantile_size=quantile_size,
        label_size=label_size
    )

    # Save features to pickle files
    features.dump('mtf_Features.pkl')
    labels.dump('mtf_Labels.pkl')
    prices.dump('mtf_Prices.pkl')

    # Check label distribution
    unique, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique, counts))
    print(f'Labels distribution: {label_dist}')

    # Print shapes
    print(f'Features shape: {np.array(features).shape}')
    print(f'Labels shape: {np.array(labels).shape}')
    print(f'Prices shape: {np.array(prices).shape}')

    # Generate visualization images
    output_miscellaneous(features)


if __name__ == "__main__":
    main()

    
    
    
