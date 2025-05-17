#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any

from typing import Union

class KlinePersistenceInterface(ABC):
    """
    K线数据持久化接口。
    允许使用不同的存储后端（例如，内存、数据库）。
    """

    @abstractmethod
    def add(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """添加单个K线字典或K线字典列表到存储中。"""
        pass
        
    @abstractmethod
    def get_aggregated(self, agg_interval: str) -> pd.DataFrame:
        """
        返回按指定间隔聚合的K线数据。
        """
        pass

    @abstractmethod
    def get_klines_df(self) -> pd.DataFrame:
        """
        返回原始基础K线数据。
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
