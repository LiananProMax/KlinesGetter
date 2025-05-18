#!/usr/bin/env python
# -*- coding: utf-8 -*-

import structlog
import pandas as pd

# 使用模块级logger
log = structlog.get_logger(__name__)

def display_historical_aggregated_klines(df_aggregated: pd.DataFrame, symbol: str, agg_interval: str, display_count: int):
    """显示初始历史聚合K线集。"""
    if df_aggregated.empty:
        # symbol 和 agg_interval 已经在main_app的context中了，但这里也加上以明确
        log.warning(f"无历史聚合K线数据可显示", symbol_param=symbol, agg_interval_param=agg_interval)
        return

    actual_display_count = min(display_count, len(df_aggregated))
    log.info(
        "显示历史聚合K线 (初始加载)",
        # symbol, agg_interval 已在上下文中
        requested_display=display_count,
        actual_displayed=actual_display_count,
        agg_interval_val=agg_interval # 使用不同键名，避免与上下文中的agg_interval潜在冲突
    )

    if 'timestamp' not in df_aggregated.columns:
        log.error("display_historical: 聚合DataFrame中缺少'timestamp'列", columns=list(df_aggregated.columns))
        return

    # 为了避免过多日志，可以选择只记录部分，或者在DEBUG级别记录全部
    # 此处依旧选择INFO级别，但简化事件名称
    for _, row in df_aggregated.tail(actual_display_count).iterrows():
        log.info(
            "历史K线", # 统一事件名称
            # symbol, agg_interval 已在上下文中
            ts=row['timestamp'].isoformat(), # 使用短键名
            o=row['open'], h=row['high'], l=row['low'], c=row['close'],
            v=row['volume'],
            # qv=row['quote_volume'], # 交易额信息可选，INFO级别下可能过于冗长
            k_status="closed" # 历史数据总是已关闭
        )
    log.info(f"{'='*20} 历史数据显示完毕 ({agg_interval}@{symbol}) {'='*20}")

    if len(df_aggregated) < display_count:
        log.debug(
            "显示的聚合历史K线少于请求数量",
            displayed_count=len(df_aggregated),
            requested_count=display_count
        )


def _determine_kline_status(row_index, df_len, agg_start_time, agg_end_time, latest_base_time=None, base_duration=None):
    """确定K线状态（正在形成或已关闭）"""
    if row_index < df_len - 1:
        return "closed" # 已关闭

    if latest_base_time is not None and base_duration is not None:
        # 如果最新的基础K线的结束时间 >= 这个聚合K线的预期结束时间，那么这个聚合K线也算关闭了
        latest_base_close_time = latest_base_time + base_duration
        if latest_base_close_time >= agg_end_time:
            return "closed"
    return "forming" # 正在形成


def display_realtime_update(df_aggregated: pd.DataFrame, symbol_ws: str, agg_interval_str: str, base_interval_str: str, all_base_klines_df: pd.DataFrame):
    """显示实时更新的最新聚合K线。"""
    if df_aggregated.empty:
        # 此日志可能非常频繁，如果这是正常现象（例如，等待第一个完整的聚合K线），则应为DEBUG
        log.debug("无聚合K线数据可用于实时更新 (可能正在等待数据)",
                  # symbol=symbol_ws, agg_interval=agg_interval_str # 已在上下文
                 )
        return

    # 此初始日志可以是DEBUG，因为它会频繁发生。实际的K线日志是INFO。
    log.debug(
        f"准备显示实时更新 ({agg_interval_str}@{symbol_ws})",
        # symbol=symbol_ws, agg_interval=agg_interval_str # 已在上下文
        base_interval_val=base_interval_str # 使用不同键名
    )
    display_recent_count = 3 # 为简洁起见，控制台显示最近3条，可调整

    if 'timestamp' not in df_aggregated.columns:
        log.error("display_realtime: 聚合DataFrame中缺少'timestamp'列", columns=list(df_aggregated.columns))
        return
    if not all_base_klines_df.empty and 'timestamp' not in all_base_klines_df.columns:
        log.error("display_realtime: 基础K线DataFrame中缺少'timestamp'列", columns=list(all_base_klines_df.columns))
        return

    try:
        base_interval_duration = pd.Timedelta(base_interval_str)
        agg_interval_duration = pd.Timedelta(agg_interval_str)
    except ValueError as e:
        log.error("无法解析间隔字符串 (实时)", base_int=base_interval_str, agg_int=agg_interval_str, error=str(e))
        return

    latest_base_time = None
    if not all_base_klines_df.empty:
        latest_base_time = all_base_klines_df['timestamp'].iloc[-1]

    # log.info(f"--- 当前 {agg_interval_str} OHLCV ({symbol_ws}) ---") # 替代为下面的循环日志
    for i, row_agg in df_aggregated.tail(display_recent_count).iterrows():
        agg_start_time = row_agg['timestamp']
        agg_end_time = agg_start_time + agg_interval_duration

        status = _determine_kline_status(
            df_aggregated.index.get_loc(i), # 获取相对索引
            len(df_aggregated),
            agg_start_time, agg_end_time,
            latest_base_time, base_interval_duration
        )

        log.info(
            "实时K线", # 统一事件名称
            # symbol=symbol_ws, agg_interval=agg_interval_str # 已在上下文
            ts=row_agg['timestamp'].isoformat(),
            o=row_agg['open'], h=row_agg['high'], l=row_agg['low'], c=row_agg['close'],
            # v=row_agg['volume'], # 交易量信息可选
            k_status=status # 关键信息：forming 或 closed
        )
    # 可选：添加一个不太显眼的分隔符或不加，因为实时更新很频繁
    # log.debug(f"--- 实时K线更新显示完毕 ({agg_interval_str}@{symbol_ws}) ---")