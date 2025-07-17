import numpy as np
from scipy.signal import butter, filtfilt
from typing import Callable, Iterable
import warnings

def _apply_filter_with_nan_handling(data: Iterable, filter_func: Callable) -> np.ndarray:
    """
    Apply a filter function while properly handling NaN values.
    Preserves original array structure with NaNs in appropriate positions.
    """
    data = np.asarray(data, dtype=np.float64)
    
    if data.ndim == 1:
        # Handle 1D case
        if np.isnan(data).all():
            return data
        
        # Find valid data range
        valid_mask = ~np.isnan(data)
        if not valid_mask.any():
            return data
            
        start = np.argmax(valid_mask)
        end = len(data) - np.argmax(valid_mask[::-1])
        
        # Extract and interpolate valid section
        valid_data = data[start:end].copy()
        nans = np.isnan(valid_data)
        if nans.any():
            valid_data[nans] = np.interp(
                np.flatnonzero(nans),
                np.flatnonzero(~nans),
                valid_data[~nans]
            )
        
        # Apply filter
        filtered_section = filter_func(valid_data)
        
        # Reconstruct full array
        result = np.full_like(data, np.nan)
        result[start:end] = filtered_section
        return result
    
    else:
        # Handle multi-dimensional case (apply to each column/axis)
        result = np.full_like(data, np.nan)
        for i in range(data.shape[1] if data.ndim == 2 else data.shape[0]):
            if data.ndim == 2:
                column_data = data[:, i]
            else:
                column_data = data[i]
            result[:, i] = _apply_filter_with_nan_handling(column_data, filter_func)
        return result

def force_plate_filter(rate: float) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    50 Hz low-pass filter with 58-62 Hz notch filter.
    """
    nyquist = 0.5 * rate
    low_cutoff = 50.0 / nyquist
    low = butter(2, low_cutoff, btype='low')
    if low is None:
        raise ValueError("Low-pass filter coefficients are None, check filter design.")
    notch_freq = [freq / nyquist for freq in [58.0, 62.0]]
    notch = butter(2, notch_freq, btype='bandstop')
    if notch is None:
        raise ValueError("Notch filter coefficients are None, check filter design.")
    
    def apply_filters(x):
        x = np.asarray(x)
        # Apply notch filter first, then low-pass
        notched = filtfilt(notch[0], notch[1], x, axis=0)
        return filtfilt(low[0], low[1], notched, axis=0)
    
    return lambda x: _apply_filter_with_nan_handling(x, apply_filters)

def marker_filter(rate: float) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    15 Hz low-pass filter.
    """
    nyquist = 0.5 * rate
    low_cutoff = 15.0 / nyquist
    low = butter(2, low_cutoff, btype='low')
    if low is None:
        raise ValueError("Low-pass filter coefficients are None, check filter design.")
    def apply_filter(x):
        return filtfilt(low[0], low[1], x, axis=0)
    
    return lambda x: _apply_filter_with_nan_handling(x, apply_filter)