#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

def format_kline_from_api(kline_data):
    """将币安API的单个K线列表转换为字典。"""
    return {
        'timestamp': pd.to_datetime(kline_data[0], unit='ms', utc=True),
        'open': float(kline_data[1]),
        'high': float(kline_data[2]),
        'low': float(kline_data[3]),
        'close': float(kline_data[4]),
        'volume': float(kline_data[5]),
        'quote_volume': float(kline_data[7]),
    }

BINANCE_TO_PANDAS = {
    'm': 'min',
    'h': 'h',
    'd': 'D',
    'w': 'W'
}

def get_pandas_resample_interval(binance_interval_str):
    """将币安间隔字符串转换为Pandas重采样兼容字符串。"""
    suffix = binance_interval_str[-1]
    mapping = BINANCE_TO_PANDAS.get(suffix)
    if not mapping:
        raise ValueError(f"不支持Binance间隔后缀: {suffix}")
    return binance_interval_str[:-1] + mapping
