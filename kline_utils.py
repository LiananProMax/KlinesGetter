# kline_utils.py
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

def get_pandas_resample_interval(binance_interval_str):
    """将币安间隔字符串转换为Pandas重采样兼容字符串。"""
    if binance_interval_str.endswith('m'): # 例如，1m, 3m, 5m
        return binance_interval_str[:-1] + 'min'
    elif binance_interval_str.endswith('h'): # 例如，1h, 3h, 4h
        return binance_interval_str[:-1] + 'H'
    elif binance_interval_str.endswith('d'): # 例如，1d
        return binance_interval_str[:-1] + 'D'
    elif binance_interval_str.endswith('w'): # 例如，1w
        return binance_interval_str[:-1] + 'W'
    return binance_interval_str
