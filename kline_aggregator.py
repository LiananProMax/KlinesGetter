# kline_aggregator.py
import pandas as pd
from kline_utils import get_pandas_resample_interval # 从本地模块导入

def aggregate_klines_df(df_source, agg_interval_str):
    """将K线DataFrame聚合到指定的时间间隔。"""
    if df_source.empty:
        return pd.DataFrame()

    df_resample_source = df_source.set_index('timestamp').copy()
    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum'
    }
    
    pandas_agg_interval = get_pandas_resample_interval(agg_interval_str)
    df_agg = df_resample_source.resample(pandas_agg_interval, label='left', closed='left').agg(agg_rules)
    df_agg = df_agg.dropna().reset_index()
    return df_agg
