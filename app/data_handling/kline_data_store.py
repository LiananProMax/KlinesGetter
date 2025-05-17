#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import logging # 导入logging
from typing import List, Dict, Any
from .data_interfaces import KlinePersistenceInterface # 使用相对导入

class KlineDataStore(KlinePersistenceInterface):
    def __init__(self, base_interval_str: str, agg_interval_str: str, historical_candles_to_display_count: int):
        self.df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
        self._base_interval_str = base_interval_str
        self._agg_interval_str = agg_interval_str
        self._historical_candles_to_display_count = historical_candles_to_display_count
        logging.info(f"KlineDataStore初始化完成，用于{base_interval_str} -> {agg_interval_str}。")

    def add_klines(self, klines_list_of_dicts: List[Dict[str, Any]]):
        """向存储中添加新的K线字典列表。"""
        if not klines_list_of_dicts:
            return

        new_klines_df = pd.DataFrame(klines_list_of_dicts)
        new_klines_df['timestamp'] = pd.to_datetime(new_klines_df['timestamp'], utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            if col in new_klines_df.columns:
                new_klines_df[col] = pd.to_numeric(new_klines_df[col])

        if self.df.empty:
            self.df = new_klines_df
        else:
            self.df = pd.concat([self.df, new_klines_df], ignore_index=True)
        
        self.df = self.df.drop_duplicates(subset=['timestamp'], keep='last')
        self.df = self.df.sort_values(by='timestamp').reset_index(drop=True)
        
        self._manage_memory()

    def add_single_kline(self, kline_dict: Dict[str, Any]):
        """添加单个K线字典，通常来自WebSocket。"""
        self.add_klines([kline_dict])

    def get_klines_df(self) -> pd.DataFrame:
        """返回当前基础K线的DataFrame副本。"""
        return self.df.copy()

    def get_latest_base_kline_timestamp(self) -> pd.Timestamp | None:
        """获取内存中最新基础K线的timestamp（开盘时间）。"""
        if self.df.empty:
            return None
        return self.df['timestamp'].iloc[-1]

    def get_base_interval_str(self) -> str:
        return self._base_interval_str

    def get_agg_interval_str(self) -> str:
        return self._agg_interval_str
        
    def get_historical_candles_to_display_count(self) -> int:
        return self._historical_candles_to_display_count

    def _manage_memory(self):
        """通过保持基础K线数据的滚动窗口来管理内存。"""
        try:
            base_td = pd.Timedelta(self._base_interval_str)
            agg_td = pd.Timedelta(self._agg_interval_str)

            if base_td.total_seconds() == 0:
                logging.error(f"基础间隔'{self._base_interval_str}'在_manage_memory中解析为零持续时间。")
                # 备用倍数，例如，如果'1m'无法解析，则为3小时的1分钟数据
                base_intervals_per_agg = 180 
            else:
                base_intervals_per_agg = agg_td.total_seconds() / base_td.total_seconds()

            max_base_rows_to_keep = int((self._historical_candles_to_display_count + 20) * base_intervals_per_agg)
            
            if len(self.df) > max_base_rows_to_keep and max_base_rows_to_keep > 0:
                logging.debug(f"内存管理：将基础K线从{len(self.df)}行修剪至{max_base_rows_to_keep}行。")
                self.df = self.df.iloc[-max_base_rows_to_keep:]
        except Exception as e_mem:
            logging.error(f"_manage_memory中出错：{e_mem}。"
                          f"基础间隔：{self._base_interval_str}，聚合间隔：{self._agg_interval_str}")
            default_max_rows = (self._historical_candles_to_display_count + 20) * 180 
            if len(self.df) > default_max_rows:
                 self.df = self.df.iloc[-default_max_rows:]
