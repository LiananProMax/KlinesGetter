#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import structlog
import pandas as pd
import psycopg2
from psycopg2 import extras, sql
from typing import List, Dict, Any, Optional, Union
from datetime import timezone, datetime #确保导入datetime
from app.core.config import config
from .data_interfaces import KlinePersistenceInterface
from .kline_aggregator import aggregate_klines_df

class DBManager(KlinePersistenceInterface):
    def __init__(self, symbol: str, base_interval_str: str, agg_interval_str: str, historical_candles_to_display_count: int):
        self.symbol = symbol.lower()
        self._base_interval_str = base_interval_str
        self._agg_interval_str = agg_interval_str
        self._historical_candles_to_display_count = historical_candles_to_display_count
        self.log = structlog.get_logger(db_manager_for=self.symbol) # 为每个实例创建不同的logger name

        safe_symbol = ''.join(filter(str.isalnum, self.symbol))
        safe_base_interval = ''.join(filter(str.isalnum, self._base_interval_str))
        safe_agg_interval = ''.join(filter(str.isalnum, self._agg_interval_str))

        self.base_table_name = f"klines_base_{safe_symbol}_{safe_base_interval}"
        self.agg_table_name = f"klines_agg_{safe_symbol}_{safe_agg_interval}"


        self.conn_params = {
            "host": config.DB_HOST,
            "port": config.DB_PORT,
            "dbname": config.DB_NAME,
            "user": config.DB_USER,
            "password": config.DB_PASSWORD
        }
        self._initialize_db()
        self.log.debug("数据库管理器已初始化", base_table=self.base_table_name, agg_table=self.agg_table_name)

    def _get_connection(self):
        """建立新的数据库连接，并设置会话时区为UTC。"""
        try:
            conn = psycopg2.connect(**self.conn_params)
            with conn.cursor() as cur:
                cur.execute("SET TIME ZONE 'UTC';")
            return conn
        except psycopg2.Error as e:
            self.log.error("连接PostgreSQL数据库时出错", error=str(e))
            raise

    def _ensure_kline_table(self, table_name: str):
        """确保指定的K线表存在，并检查timestamp列是否为TIMESTAMPTZ。"""
        check_query = sql.SQL("""
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.columns 
            WHERE table_name = %s 
              AND table_schema = current_schema() -- 确保在正确的模式下检查
              AND column_name = 'timestamp' 
              AND data_type = 'timestamp with time zone'
        );
        """)
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(check_query, (table_name,)) # psycopg2 会自动处理SQL注入风险
                    result = cur.fetchone()
                    table_ok = result[0] if result else False

                    if not table_ok:
                        drop_query = sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name))
                        cur.execute(drop_query)
                        self.log.warning(f"已删除表 '{table_name}'，因为 timestamp 列类型不正确或表不存在。之前的数据已删除。")

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
                    """).format(sql.Identifier(table_name))
                    cur.execute(create_table_query)
                    self.log.debug(f"确保了表 '{table_name}' 的存在，timestamp 列为 TIMESTAMPTZ（UTC存储）。")
        except psycopg2.Error as e:
            self.log.error("初始化数据库表时出错", table_name=table_name, error=str(e))
            raise

    def _initialize_db(self):
        """确保基础K线表和聚合K线表都存在。"""
        self._ensure_kline_table(self.base_table_name)
        self._ensure_kline_table(self.agg_table_name)

    def _insert_klines_to_table(self, klines_list_of_dicts: List[Dict[str, Any]], table_name: str):
        """将K线字典列表添加到指定的数据库表中。"""
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
        """).format(sql.Identifier(table_name))

        data_to_insert = []
        for kline in klines_list_of_dicts:
            # 确保 timestamp 是 Python datetime 对象，并且是 UTC 的
            ts_input = kline['timestamp']
            if isinstance(ts_input, pd.Timestamp):
                dt_object = ts_input.to_pydatetime() # 已经是UTC
            elif isinstance(ts_input, datetime):
                if ts_input.tzinfo is None: # 如果是 naive datetime
                    dt_object = ts_input.replace(tzinfo=timezone.utc) # 假设是UTC
                else: # 如果已经是 aware datetime
                    dt_object = ts_input.astimezone(timezone.utc) # 转换为UTC
            else: # 尝试从字符串或其他类型转换
                dt_object = pd.to_datetime(ts_input, utc=True).to_pydatetime()

            if dt_object.tzinfo != timezone.utc: # 再次确认
                 self.log.warning(f"插入数据时检测到非UTC时间戳（已转换后），这不应该发生。原始时间戳：{kline['timestamp']}, 处理后：{dt_object.isoformat()}", table_name=table_name)
                 dt_object = dt_object.astimezone(timezone.utc)


            data_to_insert.append((
                dt_object,
                float(kline['open']), float(kline['high']), float(kline['low']), float(kline['close']),
                float(kline['volume']), float(kline['quote_volume'])
            ))
            # self.log.debug("准备插入的K线timestamp", timestamp=dt_object.isoformat(), tzinfo=str(dt_object.tzinfo), table_name=table_name)


        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(cur, insert_query.as_string(conn), data_to_insert, page_size=100)
            self.log.debug("成功插入/更新K线数据", count=len(data_to_insert), table_name=table_name)
        except psycopg2.Error as e:
            self.log.error("向表插入K线数据时出错", table_name=table_name, error=str(e), first_kline_ts=klines_list_of_dicts[0]['timestamp'] if klines_list_of_dicts else "N/A")
        except Exception as ex:
            self.log.error("插入过程中发生意外错误", table_name=table_name, error=str(ex))

    def add(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """实现抽象方法：添加单个K线字典或K线字典列表到基础K线存储中。"""
        klines_list = data if isinstance(data, list) else [data]
        self._insert_klines_to_table(klines_list, self.base_table_name)

    def store_aggregated_data(self, aggregated_klines_list: List[Dict[str, Any]]):
        """实现抽象方法：将聚合后的K线字典列表存储到聚合K线存储中。"""
        self._insert_klines_to_table(aggregated_klines_list, self.agg_table_name)

    def add_klines(self, klines_list_of_dicts: List[Dict[str, Any]]):
        """将K线字典列表添加到基础K线数据库。"""
        self._insert_klines_to_table(klines_list_of_dicts, self.base_table_name)

    def add_single_kline(self, kline_dict: Dict[str, Any]):
        """将单个K线字典添加到基础K线数据库。"""
        self._insert_klines_to_table([kline_dict], self.base_table_name)

    def get_aggregated(self, agg_interval: str) -> pd.DataFrame:
        """
        实现抽象方法：返回按指定间隔聚合的K线数据。
        当前实现总是从基础表聚合。未来可以优化为如果请求的 agg_interval
        与 self._agg_interval_str 匹配，则从 self.agg_table_name 读取。
        """
        base_df = self.get_klines_df() # 从基础表获取数据
        if base_df.empty:
            self.log.debug("基础K线数据为空，无法进行聚合", for_interval=agg_interval)
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
        return aggregate_klines_df(base_df, agg_interval)


    def get_klines_df(self) -> pd.DataFrame:
        """
        从基础K线数据库表获取K线数据框。
        """
        num_base_rows_to_fetch = 0
        try:
            base_td = pd.Timedelta(self._base_interval_str)
            agg_td = pd.Timedelta(self._agg_interval_str)

            if base_td.total_seconds() == 0 or agg_td.total_seconds() == 0:
                self.log.warning("间隔解析为零持续时间", base_interval=self._base_interval_str, agg_interval=self._agg_interval_str)
                num_base_rows_to_fetch = 5000
            else:
                base_intervals_per_agg = agg_td.total_seconds() / base_td.total_seconds()
                num_base_rows_to_fetch = int((self._historical_candles_to_display_count + 20) * base_intervals_per_agg) # 加20作为缓冲区

            if num_base_rows_to_fetch <= 0:
                self.log.warning("计算出的行数无效，使用默认值", calculated_rows=num_base_rows_to_fetch, default_value=5000)
                num_base_rows_to_fetch = 5000
        except Exception as e_calc:
            self.log.error("计算要获取的行数时出错", error=str(e_calc))
            num_base_rows_to_fetch = 5000

        query = sql.SQL("""
            SELECT timestamp, open, high, low, close, volume, quote_volume 
            FROM {} 
            ORDER BY timestamp DESC 
            LIMIT %s
        """).format(sql.Identifier(self.base_table_name))

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (num_base_rows_to_fetch,))
                    rows = cur.fetchall()
                    columns = [desc[0] for desc in cur.description]

            df = pd.DataFrame(rows, columns=columns)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                if df['timestamp'].dt.tz is None or df['timestamp'].dt.tz.utcoffset(df['timestamp'].iloc[0]) != timezone.utc.utcoffset(None) :
                    self.log.warning("从数据库检索到的timestamp中有些不是UTC-aware，或非UTC，已强制转换为UTC。", table_name=self.base_table_name)
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
            else:
                df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
            
            # 由于我们是DESC获取，所以需要反转顺序才是时间升序
            df = df.sort_values(by='timestamp').reset_index(drop=True)
            self.log.debug("从表获取基础K线数据", table_name=self.base_table_name, rows_count=len(df), requested_max=num_base_rows_to_fetch)
            return df
        except psycopg2.Error as e:
            self.log.error("从表获取K线数据时出错", table_name=self.base_table_name, error=str(e))
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])

    def get_base_interval_str(self) -> str:
        return self._base_interval_str

    def get_agg_interval_str(self) -> str:
        return self._agg_interval_str

    def get_historical_candles_to_display_count(self) -> int:
        return self._historical_candles_to_display_count

    def close(self):
        self.log.debug("数据库管理器正在关闭", base_table=self.base_table_name, agg_table=self.agg_table_name, note="连接按操作管理")