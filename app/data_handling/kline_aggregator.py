#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from app.utils.kline_utils import get_pandas_resample_interval # 更新了导入路径

def aggregate_klines_df(df_source: pd.DataFrame, agg_interval_str: str) -> pd.DataFrame:
    """将K线DataFrame聚合到指定的时间间隔。"""
    if df_source.empty:
        return pd.DataFrame()

    # 确保timestamp是重采样的索引
    if 'timestamp' in df_source.columns:
        df_resample_source = df_source.set_index('timestamp').copy()
    else: # 如果timestamp已经是索引
        df_resample_source = df_source.copy()

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
