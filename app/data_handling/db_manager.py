#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import psycopg2 # 导入psycopg2
from psycopg2 import extras
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.core import config # 导入数据库连接详情和间隔配置
from .data_interfaces import KlinePersistenceInterface

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
        logging.info(f"数据库管理器已初始化，表名：'{self.table_name}'。")

    def _get_connection(self):
        """建立新的数据库连接。"""
        try:
            conn = psycopg2.connect(**self.conn_params)
            return conn
        except psycopg2.Error as e:
            logging.error(f"连接PostgreSQL数据库时出错：{e}")
            raise

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
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(create_table_query)
            conn.commit()
            logging.info(f"表 '{self.table_name}' 已确保存在。")
        except psycopg2.Error as e:
            logging.error(f"初始化数据库表 {self.table_name} 时出错：{e}")
            # 如果表创建失败，应用程序可能无法正常运行。
            # 考虑是抛出错误还是更优雅地处理它。
            if conn: conn.rollback()
            raise
        finally:
            self._release_connection(conn)

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
        # 确保'timestamp'是datetime对象，其他是数值类型
        data_to_insert = []
        for kline in klines_list_of_dicts:
            dt_object = pd.to_datetime(kline['timestamp'], utc=True)
            data_to_insert.append((
                dt_object,
                kline['open'], kline['high'], kline['low'], kline['close'],
                kline['volume'], kline['quote_volume']
            ))

        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, insert_query, data_to_insert, page_size=100)
            conn.commit()
            logging.debug(f"成功插入/更新 {len(data_to_insert)} 条K线数据到 {self.table_name}。")
        except psycopg2.Error as e:
            logging.error(f"向 {self.table_name} 插入K线数据时出错：{e}")
            if conn: conn.rollback()
        except Exception as ex: # 捕获其他潜在错误，如数据准备期间的错误
            logging.error(f"add_klines过程中发生意外错误：{ex}")
            if conn: conn.rollback()
        finally:
            self._release_connection(conn)

    def add_single_kline(self, kline_dict: Dict[str, Any]):
        """将单个K线字典添加到数据库。"""
        self.add_klines([kline_dict])

    def get_latest_base_kline_timestamp(self) -> datetime | None:
        """从数据库获取最新基础K线的timestamp（开盘时间）。"""
        query = f"""
            SELECT timestamp
            FROM {self.table_name}
            ORDER BY timestamp DESC
            LIMIT 1;
        """
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(query)
                row = cur.fetchone()
            if row:
                # fetchone returns a tuple, row[0] is the datetime object
                return row[0]
            return None # Return None if no rows found
        except psycopg2.Error as e:
            logging.error(f"从 {self.table_name} 获取最新时间戳时出错：{e}")
            return None
        finally:
            self._release_connection(conn)

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
                logging.warning(f"基础间隔'{self._base_interval_str}'或聚合间隔'{self._agg_interval_str}'解析为零持续时间。")
                # 如果间隔解析失败，设置一个较大的默认值，假设1分钟基础K线用于3小时聚合。
                base_intervals_per_agg = 180
            else:
                base_intervals_per_agg = agg_td.total_seconds() / base_td.total_seconds()
            
            # 添加缓冲区（例如，再加20个聚合间隔）以确保有足够的数据进行重采样边缘情况
            num_base_rows_to_fetch = int((self._historical_candles_to_display_count + 20) * base_intervals_per_agg)
            
            if num_base_rows_to_fetch <= 0: # 安全检查
                logging.warning(f"计算出的num_base_rows_to_fetch为{num_base_rows_to_fetch}。使用默认值5000。")
                num_base_rows_to_fetch = 5000 # 一个合理的回退值

        except Exception as e_calc:
            logging.error(f"在DBManager.get_klines_df中计算要获取的行数时出错：{e_calc}。使用默认值。")
            # 基于典型TEST模式的默认值（例如，从1分钟基础聚合为3分钟）显示50个K线+缓冲区。
            # (50个要显示的K线 + 20个缓冲区K线) * (3分钟聚合间隔 / 1分钟基础间隔) = 70 * 3 = 210
            # 如果是PRODUCTION（例如，从1小时基础聚合为3小时）-> 70 * 3 = 210
            # 如果间隔字符串不可预测，更高的通用默认值可能更安全。使用5000。
            num_base_rows_to_fetch = 5000


        # 此查询获取最新的'num_base_rows_to_fetch'行，按时间戳升序排序
        query = f"""
            SELECT * FROM (
                SELECT timestamp, open, high, low, close, volume, quote_volume
                FROM {self.table_name}
                ORDER BY timestamp DESC
                LIMIT %s
            ) AS recent_klines
            ORDER BY timestamp ASC;
        """
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(query, (num_base_rows_to_fetch,))
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
            
            df = pd.DataFrame(rows, columns=columns)
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True) # 确保时区
            else: # 即使为空也确保模式匹配
                 df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])


            logging.debug(f"从 {self.table_name} 获取了 {len(df)} 行用于处理。请求了最多 {num_base_rows_to_fetch} 行。")
            return df
        except psycopg2.Error as e:
            logging.error(f"从 {self.table_name} 获取K线数据时出错：{e}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
        finally:
            self._release_connection(conn)

    def get_base_interval_str(self) -> str:
        return self._base_interval_str

    def get_agg_interval_str(self) -> str:
        return self._agg_interval_str

    def get_historical_candles_to_display_count(self) -> int:
        return self._historical_candles_to_display_count

    def close(self):
        """如果需要显式资源清理的占位符，尽管连接是按方法管理的。"""
        logging.info(f"{self.table_name} 的数据库管理器正在'关闭'（连接按操作管理）。")
