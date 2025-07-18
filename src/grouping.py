import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline

def trim_and_spline(df: pl.DataFrame, start: float, end: float, time_col: str = 'time', num_points: int = 100) -> pl.DataFrame:
    # Trim the DataFrame to the specified time range and clean it using polars operations
    trimmed = (df
               .filter(pl.col(time_col).is_between(start, end))  # Trim to time range
               .filter(pl.col(time_col).is_finite())             # Remove non-finite times
               .sort(time_col)                                   # Sort by time
               .unique(subset=[time_col], keep='first'))         # Remove duplicate times
    
    if len(trimmed) < 2:
        raise ValueError(f"Not enough data points in time range [{start:.3f}, {end:.3f}] after cleaning. Only {len(trimmed)} points found.")
    
    # Get time bounds for interpolation - convert to numpy for safety
    time_array = trimmed[time_col].to_numpy()
    time_min = float(time_array.min())
    time_max = float(time_array.max())
    splined_times = np.linspace(time_min, time_max, num_points)
    
    # Create the result DataFrame with the new time points
    result_df = pl.DataFrame({time_col: splined_times})
    data_df = trimmed.drop(time_col)

    # Identify columns with and without NaN values (check both null and NaN)
    clean_columns = []
    nan_columns = []
    
    for col in data_df.columns:
        col_data = data_df[col]
        has_nulls = col_data.null_count() > 0
        has_nans = col_data.is_nan().sum() > 0 if col_data.dtype.is_numeric() else False
        
        if has_nulls or has_nans:
            nan_columns.append(col)
        else:
            clean_columns.append(col)
    
    if nan_columns:
        print(f"Warning: Skipping columns with NaN values: {nan_columns}")
    
    if clean_columns:
        # Process clean columns with vectorized approach
        clean_data_df = data_df.select(clean_columns)
        cs = CubicSpline(time_array, clean_data_df.to_numpy(), axis=0)
        splined_data = cs(splined_times)
        splined_df = pl.DataFrame(splined_data, schema=clean_columns)
        result_df = result_df.with_columns(splined_df)
    return result_df

def spline_to_stance_swing(df: pl.DataFrame, stance_start: float, stance_end: float, swing_start: float, swing_end: float, time_col: str = 'time') -> pl.DataFrame:
    """
    Splits the DataFrame into stance and swing phases based on the provided time intervals.
    """
    stance_df = trim_and_spline(df, stance_start, stance_end, time_col)
    swing_df = trim_and_spline(df, swing_start, swing_end, time_col)
    return pl.concat([stance_df, swing_df]).sort(time_col)