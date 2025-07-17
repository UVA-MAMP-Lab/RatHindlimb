import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline

def trim_and_spline(df: pl.DataFrame, start: float, end: float, time_col: str = 'time', num_points: int = 100) -> pl.DataFrame:
    # Trim the DataFrame to the specified time range
    trimmed = df.filter(pl.col(time_col).is_between(start, end))
    time_np = trimmed[time_col].to_numpy()
    splined_times = np.linspace(time_np.min(), time_np.max(), num_points)
    
    # Apply cubic spline interpolation
    cs = CubicSpline(time_np, trimmed.drop(time_col).to_numpy(), axis=0)
    splined_data = cs(splined_times)
    splined_df = pl.DataFrame(splined_data, schema=trimmed.columns[1:])
    splined_df = splined_df.with_columns(pl.Series(time_col, splined_times))
    return splined_df

def spline_to_stance_swing(df: pl.DataFrame, stance_start: float, stance_end: float, swing_start: float, swing_end: float, time_col: str = 'time') -> pl.DataFrame:
    """
    Splits the DataFrame into stance and swing phases based on the provided time intervals.
    """
    stance_df = trim_and_spline(df, stance_start, stance_end, time_col)
    swing_df = trim_and_spline(df, swing_start, swing_end, time_col)
    return pl.concat([stance_df, swing_df]).sort(time_col)