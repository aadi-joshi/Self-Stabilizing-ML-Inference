"""
Stability metrics for controller evaluation.

Definitions:
- Stability Horizon: Longest consecutive period where the system remains above the reliability threshold.
- Oscillation Bound per Window: Maximum number of model switches in a fixed window (e.g., 50 steps).
- Recovery Time Distribution: Distribution of steps required to recover above threshold after a drop.

All metrics are environment-agnostic and operate on controller logs (DataFrame).
"""
import numpy as np
import pandas as pd

def compute_stability_horizon(df, threshold):
    """
    Returns the longest consecutive period where smoothed reliability > threshold.
    """
    stable = df['smoothed_reliability'] > threshold
    max_stable = (stable.groupby((stable != stable.shift()).cumsum()).cumsum() * stable).max()
    return int(max_stable) if not np.isnan(max_stable) else 0

def compute_oscillation_bound(df, window=50):
    """
    Returns the maximum number of model switches in any window of given size.
    """
    switches = (df['active_model'] != df['active_model'].shift()).astype(int)
    rolling = switches.rolling(window=window, min_periods=1).sum()
    return int(rolling.max())

def compute_recovery_time_distribution(df, threshold):
    """
    Returns a list of recovery times (steps to recover above threshold after a drop).
    """
    below = df['smoothed_reliability'] < threshold
    recovery_times = []
    i = 0
    while i < len(df):
        if below.iloc[i]:
            drop_idx = i
            # Find next recovery
            recovered = (df['smoothed_reliability'].iloc[drop_idx:] > threshold)
            if recovered.any():
                rec_idx = recovered.idxmax() + drop_idx
                recovery_times.append(rec_idx - drop_idx)
                i = rec_idx
            else:
                break
        i += 1
    return recovery_times

def compute_stability_metrics(df, threshold, osc_window=50):
    """
    Computes all formal stability metrics and returns as a dict.
    """
    horizon = compute_stability_horizon(df, threshold)
    osc_bound = compute_oscillation_bound(df, window=osc_window)
    rec_times = compute_recovery_time_distribution(df, threshold)
    return {
        'stability_horizon': horizon,
        'oscillation_bound': osc_bound,
        'recovery_time_mean': np.mean(rec_times) if rec_times else np.nan,
        'recovery_time_std': np.std(rec_times) if rec_times else np.nan,
        'recovery_time_median': np.median(rec_times) if rec_times else np.nan,
        'recovery_time_min': np.min(rec_times) if rec_times else np.nan,
        'recovery_time_max': np.max(rec_times) if rec_times else np.nan,
        'recovery_time_count': len(rec_times)
    }
