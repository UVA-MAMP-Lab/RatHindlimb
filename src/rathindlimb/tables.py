from IPython.display import Markdown as md
from tabulate import tabulate
import pandas as pd
import polars as pl

def display_pd(df: pd.DataFrame, *args, **kwargs):
    md(tabulate(df, headers='keys', tablefmt='pipe', *args, **kwargs))

def display_pl(df: pl.DataFrame, *args, **kwargs):
    display_pd(df.to_pandas(), *args, **kwargs)