#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any

class KlinePersistenceInterface(ABC):
    """
    K线数据持久化接口。
    允许使用不同的存储后端（例如，内存、数据库）。
    """

    @abstractmethod
    def add_klines(self, klines_list_of_dicts: List[Dict[str, Any]]):
        """添加K线数据列表到存储中。"""
        pass

    @abstractmethod
    def add_single_kline(self, kline_dict: Dict[str, Any]):
        """添加单个K线到存储中。"""
        pass

    @abstractmethod
    def get_klines_df(self) -> pd.DataFrame:
        """
        返回所有当前基础K线的DataFrame。
        """
        pass

    @abstractmethod
    def get_base_interval_str(self) -> str:
        """返回基础间隔字符串（例如，'1m'）。"""
        pass

    @abstractmethod
    def get_agg_interval_str(self) -> str:
        """返回聚合间隔字符串（例如，'3m'）。"""
        pass

    @abstractmethod
    def get_historical_candles_to_display_count(self) -> int:
        """返回要显示的历史聚合蜡烛数量。"""
        pass

    # TODO: 未来可能添加更多特定方法，例如，按时间范围获取数据，
    # 这对于数据库实现非常相关。
