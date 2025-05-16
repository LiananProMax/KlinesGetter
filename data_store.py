# data_store.py
import pandas as pd
import logging

class KlineDataStore:
    def __init__(self, base_interval_str, agg_interval_str, historical_candles_to_display_count):
        self.df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
        self.base_interval_str = base_interval_str
        self.agg_interval_str = agg_interval_str
        self.historical_candles_to_display_count = historical_candles_to_display_count
        logging.info("KlineDataStore初始化完成。")

    def add_klines(self, klines_list_of_dicts):
        """向存储中添加新的K线字典列表。"""
        if not klines_list_of_dicts:
            return

        new_klines_df = pd.DataFrame(klines_list_of_dicts)
        # 确保正确的数据类型，特别是向空DataFrame添加时
        new_klines_df['timestamp'] = pd.to_datetime(new_klines_df['timestamp'], utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            if col in new_klines_df.columns:
                new_klines_df[col] = pd.to_numeric(new_klines_df[col])

        if self.df.empty:
            self.df = new_klines_df
        else:
            # 连接并处理可能来自重新连接或重叠的重复项
            self.df = pd.concat([self.df, new_klines_df], ignore_index=True)
        
        self.df = self.df.drop_duplicates(subset=['timestamp'], keep='last')
        self.df = self.df.sort_values(by='timestamp').reset_index(drop=True)
        
        self._manage_memory()

    def add_single_kline(self, kline_dict):
        """添加单个K线字典，通常来自WebSocket。"""
        self.add_klines([kline_dict]) # 重用列表添加逻辑

    def get_base_klines_df(self):
        """返回当前基础K线的DataFrame。"""
        return self.df.copy() # 返回副本以防止外部修改

    def _manage_memory(self):
        """通过保持基础K线数据的滚动窗口来管理内存。"""
        try:
            base_td = pd.Timedelta(self.base_interval_str)
            agg_td = pd.Timedelta(self.agg_interval_str)

            if base_td.total_seconds() == 0:
                logging.error(f"基础间隔'{self.base_interval_str}'在_manage_memory中解析为零持续时间。")
                base_intervals_per_agg = 180 # 默认使用180作为1分钟数据3小时的备用倍数
            else:
                base_intervals_per_agg = agg_td.total_seconds() / base_td.total_seconds()

            # 保留足够的行以供HISTORICAL_AGG_CANDLES_TO_DISPLAY + 缓冲区（例如，20个聚合周期）
            max_base_rows_to_keep = int((self.historical_candles_to_display_count + 20) * base_intervals_per_agg)
            
            if len(self.df) > max_base_rows_to_keep and max_base_rows_to_keep > 0:
                logging.debug(f"内存管理：将基础K线从{len(self.df)}行修剪至{max_base_rows_to_keep}行。")
                self.df = self.df.iloc[-max_base_rows_to_keep:]
        except Exception as e_mem:
            logging.error(f"_manage_memory中出错：{e_mem}。"
                          f"基础间隔：{self.base_interval_str}，聚合间隔：{self.agg_interval_str}")
            # 计算失败时的备用内存限制
            default_max_rows = (self.historical_candles_to_display_count + 20) * 180 # 假设最多3小时的1分钟数据
            if len(self.df) > default_max_rows:
                 self.df = self.df.iloc[-default_max_rows:]
