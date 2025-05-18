#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import structlog
import pandas as pd
import psycopg2
from psycopg2 import extras, sql
from typing import List, Dict, Any, Optional, Union
from datetime import timezone
from app.core.config import config
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
        """建立新的数据库连接，并设置会话时区为UTC。"""
        try:
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor() as cur:
                cur.execute("SET TIME ZONE 'UTC';")  # 确保会话时区为UTC
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
        """确保K线表存在，并检查timestamp列是否为TIMESTAMPTZ。如果不是，删除并重建表。"""
        log = structlog.get_logger()
      
        # 检查表是否存在，并验证timestamp列的类型
        check_query = """
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.columns 
            WHERE table_name = %s 
              AND column_name = 'timestamp' 
              AND data_type = 'timestamp with time zone'
        );
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(check_query, (self.table_name,))
                    result = cur.fetchone()[0]  # 返回布尔值：True如果列存在且类型正确，False否则
                  
                    if not result:
                        # timestamp列类型不正确或表不存在，删除表（如果存在）
                        drop_query = sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(self.table_name))
                        cur.execute(drop_query)
                        log.warning(f"删除了表 '{self.table_name}' 因为 timestamp 列类型不正确或表不存在。之前的数据已删除。")
                  
                    # 创建或确保表存在，使用TIMESTAMPTZ
                    create_table_query = sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {} (
                        timestamp TIMESTAMPTZ PRIMARY KEY,
                        open NUMERIC,
                        high NUMERIC,
                        low NUMERIC,
                        close NUMERIC,
                        volume NUMERIC,
                        quote_volume NUMERIC
                    );
                    """).format(sql.Identifier(self.table_name))
                    cur.execute(create_table_query)
                    log.info(f"确保了表 '{self.table_name}' 的存在，timestamp 列为 TIMESTAMPTZ（UTC存储）。")
        except psycopg2.Error as e:
            log.error("初始化数据库表时出错", table_name=self.table_name, error=str(e))
            raise

    def add(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """实现抽象方法：添加单个K线字典或K线字典列表到存储中。"""
        if isinstance(data, list):
            self.add_klines(data)
        else:
            self.add_single_kline(data)

    def add_klines(self, klines_list_of_dicts: List[Dict[str, Any]]):
        """将K线字典列表添加到数据库，并验证时间戳是否为UTC。"""
        if not klines_list_of_dicts:
            return

        insert_query = sql.SQL("""
        INSERT INTO {} (timestamp, open, high, low, close, volume, quote_volume)
        VALUES %s
        ON CONFLICT (timestamp) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            quote_volume = EXCLUDED.quote_volume;
        """).format(sql.Identifier(self.table_name))

        # 准备数据，转换timestamp为UTC并验证
        data_to_insert = []
        log = structlog.get_logger()
        for kline in klines_list_of_dicts:
            dt_object = pd.to_datetime(kline['timestamp'], utc=True).tz_convert('UTC')  # 强制转换为UTC
            # 验证时间戳是否为UTC-aware
            if dt_object.tzinfo != timezone.utc:
                log.warning(f"插入数据时检测到非UTC时间戳，已强制转换为UTC。时间戳：{dt_object.isoformat()}")
            data_to_insert.append((
                dt_object,  # UTC-aware datetime
                float(kline['open']), float(kline['high']), float(kline['low']), float(kline['close']),
                float(kline['volume']), float(kline['quote_volume'])
            ))
            log.debug("准备插入的K线timestamp", timestamp=dt_object.isoformat(), tzinfo=str(dt_object.tzinfo))

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(cur, insert_query.as_string(conn), data_to_insert, page_size=100)
                # 自动提交事务（由于使用with语句）
            log.debug("成功插入/更新K线数据", count=len(data_to_insert), table_name=self.table_name)
        except psycopg2.Error as e:
            log.error("向表插入K线数据时出错", table_name=self.table_name, error=str(e))
        except Exception as ex:
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
        从数据库获取K线数据框，并验证timestamp是否为UTC。
        """
        num_base_rows_to_fetch = 0
        try:
            base_td = pd.Timedelta(self._base_interval_str)
            agg_td = pd.Timedelta(self._agg_interval_str)

            if base_td.total_seconds() == 0 or agg_td.total_seconds() == 0:
                log = structlog.get_logger()
                log.warning("间隔解析为零持续时间", base_interval=self._base_interval_str, agg_interval=self._agg_interval_str)
                num_base_rows_to_fetch = 5000  # 默认回退值
            else:
                base_intervals_per_agg = agg_td.total_seconds() / base_td.total_seconds()
                num_base_rows_to_fetch = int((self._historical_candles_to_display_count + 20) * base_intervals_per_agg)

            if num_base_rows_to_fetch <= 0:
                log = structlog.get_logger()
                log.warning("计算出的行数无效，使用默认值", num_base_rows_to_fetch=num_base_rows_to_fetch, default_value=5000)
                num_base_rows_to_fetch = 5000
        except Exception as e_calc:
            log = structlog.get_logger()
            log.error("计算要获取的行数时出错", error=str(e_calc))
            num_base_rows_to_fetch = 5000

        # 查询获取最新的行，按时间戳降序排序
        query = sql.SQL("""
            SELECT timestamp, open, high, low, close, volume, quote_volume 
            FROM {} 
            ORDER BY timestamp DESC 
            LIMIT %s
        """).format(sql.Identifier(self.table_name))

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (num_base_rows_to_fetch,))
                    rows = cur.fetchall()
                    columns = [desc[0] for desc in cur.description]

            df = pd.DataFrame(rows, columns=columns)
            if not df.empty:
                # 确保timestamp是UTC-aware datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)  # 数据库返回的TIMESTAMPTZ应该已经是UTC，但强制确保
                # 验证timestamp是否为UTC
                if df['timestamp'].dt.tz is not timezone.utc:
                    log.warning("检索到的timestamp中有些不是UTC-aware，已强制转换为UTC。")
                    df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')  # 如果需要转换，但TIMESTAMPTZ通常已经是UTC
            else:
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
