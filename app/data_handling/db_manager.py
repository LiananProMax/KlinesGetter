#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import structlog
import pandas as pd
import psycopg2
from psycopg2 import extras
from typing import List, Dict, Any, Optional, Union
from datetime import timezone

from app.core.config import config # 导入数据库连接详情和间隔配置
from .data_interfaces import KlinePersistenceInterface
from .kline_aggregator import aggregate_klines_df

class DBManager(KlinePersistenceInterface):
    def __init__(self, symbol: str, base_interval_str: str, agg_interval_str: str, historical_candles_to_display_count: int):
        self.symbol = symbol.lower()
        self._base_interval_str = base_interval_str
        self._agg_interval_str = agg_interval_str
        self._historical_candles_to_display_count = historical_candles_to_display_count

        # 将表名部分进行安全处理
        safe_symbol = ''.join(filter(str.isalnum, self.symbol))
        safe_interval = ''.join(filter(str.isalnum, self._base_interval_str))
        self.table_name = f"klines_{safe_symbol}_{safe_interval}"

        self.conn_params = {
            "host": config.DB_HOST,
            "port": config.DB_PORT,
            "dbname": config.DB_NAME,
            "user": config.DB_USER,
            "password": config.DB_PASSWORD
        }
        self._initialize_db()
        log = structlog.get_logger()
        log.info("数据库管理器已初始化", table_name=self.table_name)

    def _get_connection(self):
        """建立新的数据库连接。"""
        try:
            conn = psycopg2.connect(**self.conn_params)
            return conn
        except psycopg2.Error as e:
            log = structlog.get_logger()
            log.error("连接PostgreSQL数据库时出错", error=str(e))
            raise

    def _execute_with_connection(self, query, params=None):
        """封装数据库操作，管理连接生命周期"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:  # 使用自动提交的cursor
                cur.execute(query, params)
                return cur.fetchall()

    def _release_connection(self, conn):
        """关闭数据库连接。"""
        if conn:
            conn.close()

    def _initialize_db(self):
        """如果K线表不存在，则创建它。"""
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            timestamp TIMESTAMPTZ PRIMARY KEY,
            open NUMERIC,
            high NUMERIC,
            low NUMERIC,
            close NUMERIC,
            volume NUMERIC,
            quote_volume NUMERIC
        );
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_table_query)
                # 自动提交事务 (with块结束时)
            log = structlog.get_logger()
            log.info("表已确保存在", table_name=self.table_name)
        except psycopg2.Error as e:
            log = structlog.get_logger()
            log.error("初始化数据库表时出错", table_name=self.table_name, error=str(e))
            # 如果表创建失败，应用程序可能无法正常运行。
            raise

    def add(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """实现抽象方法：添加单个K线字典或K线字典列表到存储中。"""
        if isinstance(data, list):
            self.add_klines(data)
        else:
            self.add_single_kline(data)

    def add_klines(self, klines_list_of_dicts: List[Dict[str, Any]]):
        """将K线字典列表添加到数据库。"""
        if not klines_list_of_dicts:
            return

        insert_query = f"""
        INSERT INTO {self.table_name} (timestamp, open, high, low, close, volume, quote_volume)
        VALUES %s
        ON CONFLICT (timestamp) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            quote_volume = EXCLUDED.quote_volume;
        """

        # 将字典列表转换为元组列表，用于execute_values
        # 确保'timestamp'是datetime对象，其他是数值类型，并显式设置为UTC
        data_to_insert = []
        for kline in klines_list_of_dicts:
            dt_object = pd.to_datetime(kline['timestamp'], utc=True).tz_convert('UTC')  # 显式转换为UTC
            data_to_insert.append((
                dt_object,  # UTC-aware datetime
                float(kline['open']), float(kline['high']), float(kline['low']), float(kline['close']),
                float(kline['volume']), float(kline['quote_volume'])
            ))
            
            # 添加调试日志以验证timestamp
            log = structlog.get_logger()
            log.debug("准备插入的K线timestamp", timestamp=dt_object.isoformat(), tzinfo=str(dt_object.tzinfo))

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(cur, insert_query, data_to_insert, page_size=100)
                # 自动提交事务
            log = structlog.get_logger()
            log.debug("成功插入/更新K线数据", count=len(data_to_insert), table_name=self.table_name)
        except psycopg2.Error as e:
            log = structlog.get_logger()
            log.error("向表插入K线数据时出错", table_name=self.table_name, error=str(e))
        except Exception as ex: # 捕获其他潜在错误，如数据准备期间的错误
            log = structlog.get_logger()
            log.error("add_klines过程中发生意外错误", error=str(ex))

    def add_single_kline(self, kline_dict: Dict[str, Any]):
        """将单个K线字典添加到数据库。"""
        self.add_klines([kline_dict])

    def get_aggregated(self, agg_interval: str) -> pd.DataFrame:
        """实现抽象方法：返回按指定间隔聚合的K线数据。"""
        base_df = self.get_klines_df()
        return aggregate_klines_df(base_df, agg_interval)

    def get_klines_df(self) -> pd.DataFrame:
        """
        从数据库获取K线数据框。
        获取用于聚合和显示所需的最近K线数据。
        """
        num_base_rows_to_fetch = 0
        try:
            base_td = pd.Timedelta(self._base_interval_str)
            agg_td = pd.Timedelta(self._agg_interval_str)

            if base_td.total_seconds() == 0 or agg_td.total_seconds() == 0 : # 有效间隔不应发生这种情况
                log = structlog.get_logger()
                log.warning("间隔解析为零持续时间", base_interval=self._base_interval_str, agg_interval=self._agg_interval_str)
                # 如果间隔解析失败，设置一个较大的默认值，假设1分钟基础K线用于3小时聚合。
                base_intervals_per_agg = 180
            else:
                base_intervals_per_agg = agg_td.total_seconds() / base_td.total_seconds()

            # 添加缓冲区（例如，再加20个聚合间隔）以确保有足够的数据进行重采样边缘情况
            num_base_rows_to_fetch = int((self._historical_candles_to_display_count + 20) * base_intervals_per_agg)

            if num_base_rows_to_fetch <= 0: # 安全检查
                log = structlog.get_logger()
                log.warning("计算出的行数无效，使用默认值", num_base_rows_to_fetch=num_base_rows_to_fetch, default_value=5000)
                num_base_rows_to_fetch = 5000 # 一个合理的回退值

        except Exception as e_calc:
            log = structlog.get_logger()
            log.error("计算要获取的行数时出错", error=str(e_calc))
            # 基于典型TEST模式的默认值（例如，从1分钟基础聚合为3分钟）显示50个K线+缓冲区。
            # (50个要显示的K线 + 20个缓冲区K线) * (3分钟聚合间隔 / 1分钟基础间隔) = 70 * 3 = 210
            # 如果是PRODUCTION（例如，从1小时基础聚合为3小时）-> 70 * 3 = 210
            # 如果间隔字符串不可预测，更高的通用默认值可能更安全。使用5000。
            num_base_rows_to_fetch = 5000


        # 此查询获取最新的'num_base_rows_to_fetch'行，按时间戳降序排序，并显式指定时区为UTC
        query = f"""
            SELECT timestamp AT TIME ZONE 'UTC' AS timestamp_utc, open, high, low, close, volume, quote_volume 
            FROM {self.table_name}
            ORDER BY timestamp DESC
            LIMIT %s
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (num_base_rows_to_fetch,))
                    rows = cur.fetchall()
                    columns = [desc[0] for desc in cur.description]

            df = pd.DataFrame(rows, columns=columns)
            if not df.empty and 'timestamp_utc' in df.columns:  # 使用别名timestamp_utc
                df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)  # 确保UTC
                df.rename(columns={'timestamp_utc': 'timestamp'}, inplace=True)  # 重命名回'timestamp'
            else: # 即使为空也确保模式匹配
                df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])

            log = structlog.get_logger()
            log.debug("从表获取数据", table_name=self.table_name, rows_count=len(df), requested_max=num_base_rows_to_fetch)
            return df
        except psycopg2.Error as e:
            log = structlog.get_logger()
            log.error("从表获取K线数据时出错", table_name=self.table_name, error=str(e))
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])

    def get_base_interval_str(self) -> str:
        return self._base_interval_str

    def get_agg_interval_str(self) -> str:
        return self._agg_interval_str

    def get_historical_candles_to_display_count(self) -> int:
        return self._historical_candles_to_display_count

    def close(self):
        """如果需要显式资源清理的占位符，尽管连接是按方法管理的。"""
        log = structlog.get_logger()
        log.info("数据库管理器正在关闭", table_name=self.table_name, note="连接按操作管理")
