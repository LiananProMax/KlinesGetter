import logging
import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone # Keep timezone for clarity
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
# We removed the direct import of WebsocketClient in the previous step, which was correct.
from binance.lib.utils import config_logging

# --- 配置 ---
API_BASE_URL_FUTURES = "https://fapi.binance.com"  # 用于USDⓈ-M期货
SYMBOL = "BTCUSDT"  # 目标交易对

# --- 模式选择 ---
OPERATING_MODE = "TEST"  # "TEST" 或 "PRODUCTION"

if OPERATING_MODE == "TEST":
    BASE_INTERVAL = "1m"
    AGG_INTERVAL = "3m"
    logging.info("运行模式: TEST (1分钟数据 -> 3分钟聚合)")
elif OPERATING_MODE == "PRODUCTION":
    BASE_INTERVAL = "1h"
    AGG_INTERVAL = "3h"
    logging.info("运行模式: PRODUCTION (1小时数据 -> 3小时聚合)")
else:
    raise ValueError("无效的 OPERATING_MODE。请选择 'TEST' 或 'PRODUCTION'。")

HISTORICAL_AGG_CANDLES_TO_DISPLAY = 50
MAX_KLINE_LIMIT_PER_REQUEST = 1000

config_logging(logging, logging.INFO)

all_base_klines_df = pd.DataFrame()
websocket_client_global = None


# --- 辅助函数 --- (Assuming these are correct and working from your previous version)
def format_kline_from_api(kline_data):
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
    if binance_interval_str.endswith('m'):
        return binance_interval_str[:-1] + 'min'
    elif binance_interval_str.endswith('h'):
        return binance_interval_str[:-1] + 'H'
    elif binance_interval_str.endswith('d'):
        return binance_interval_str[:-1] + 'D'
    elif binance_interval_str.endswith('w'):
        return binance_interval_str[:-1] + 'W'
    return binance_interval_str

def aggregate_klines(df_source, agg_interval_str):
    if df_source.empty:
        return pd.DataFrame()
    df_resample_source = df_source.set_index('timestamp').copy()
    agg_rules = {
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum', 'quote_volume': 'sum'
    }
    pandas_agg_interval = get_pandas_resample_interval(agg_interval_str)
    df_agg = df_resample_source.resample(pandas_agg_interval, label='left', closed='left').agg(agg_rules)
    df_agg = df_agg.dropna().reset_index()
    return df_agg

# --- 用于历史数据的REST API --- (Assuming this is correct and working)
def fetch_historical_klines_futures_rest(symbol_rest, interval_rest, num_klines_to_fetch_rest):
    global MAX_KLINE_LIMIT_PER_REQUEST
    klines_fetched_so_far = 0
    all_klines_raw_list = []
    current_end_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000) 
    initial_requested_end_time_iso = pd.to_datetime(current_end_time_ms, unit='ms', utc=True).isoformat()
    logging.info(f"开始获取 {symbol_rest} 的 {num_klines_to_fetch_rest} 个 '{interval_rest}' 历史K线数据。初始请求结束时间 (endTime): {initial_requested_end_time_iso}")

    while klines_fetched_so_far < num_klines_to_fetch_rest:
        remaining_klines = num_klines_to_fetch_rest - klines_fetched_so_far
        current_batch_limit = min(remaining_klines, MAX_KLINE_LIMIT_PER_REQUEST)
        params = {
            'symbol': symbol_rest, 'interval': interval_rest,
            'limit': current_batch_limit, 'endTime': current_end_time_ms 
        }
        current_end_time_iso_for_request = pd.to_datetime(current_end_time_ms, unit='ms', utc=True).isoformat()
        logging.info(f"获取 {symbol_rest} 的一批 {current_batch_limit} 个 '{interval_rest}' K线，请求的endTime: {current_end_time_iso_for_request} (ms: {current_end_time_ms})")
        try:
            response = requests.get(f"{API_BASE_URL_FUTURES}/fapi/v1/klines", params=params)
            response.raise_for_status()
            data_batch = response.json()
            if not data_batch:
                logging.info("在此时间段内未从API找到更多历史数据。")
                break
            first_kline_ts_ms_in_batch = data_batch[0][0]
            last_kline_ts_ms_in_batch = data_batch[-1][0]
            logging.info(f"    批次收到 {len(data_batch)} 条K线。 "
                         f"首条K线时间: {pd.to_datetime(first_kline_ts_ms_in_batch, unit='ms', utc=True).isoformat()}, "
                         f"末条K线时间: {pd.to_datetime(last_kline_ts_ms_in_batch, unit='ms', utc=True).isoformat()}")
            all_klines_raw_list = data_batch + all_klines_raw_list
            klines_fetched_so_far += len(data_batch)
            if len(data_batch) < current_batch_limit:
                logging.info(f"获取了 {len(data_batch)} 个 '{interval_rest}' K线，少于请求的数量 ({current_batch_limit})。假设没有更多旧数据。")
                break
            current_end_time_ms = data_batch[0][0] 
            time.sleep(0.25)
        except requests.exceptions.RequestException as e:
            logging.error(f"获取历史K线时出错: {e}")
            return [] 
        except json.JSONDecodeError as e:
            logging.error(f"从历史K线响应解码JSON时出错: {e}")
            if response: logging.error(f"响应文本: {response.text}")
            return []
    klines_formatted_list = [format_kline_from_api(k) for k in all_klines_raw_list]
    if klines_formatted_list:
        logging.info(f"总共格式化 {len(klines_formatted_list)} 条历史K线。")
    else:
        logging.warning("没有历史K线被格式化。")
    return klines_formatted_list

# --- WebSocket处理程序 --- (Assuming this is correct and working)
def message_handler(_, message):
    global all_base_klines_df, BASE_INTERVAL, AGG_INTERVAL, HISTORICAL_AGG_CANDLES_TO_DISPLAY
    try:
        data = json.loads(message)
        if 'e' in data and data['e'] == 'kline':
            kline_payload = data['k']
            if kline_payload['x']:
                symbol_ws = kline_payload['s']
                base_interval_from_stream = kline_payload['i']
                logging.debug(f"收到 {symbol_ws} 的已关闭 {base_interval_from_stream} K线 @ {pd.to_datetime(kline_payload['t'], unit='ms', utc=True)}")
                new_kline_dict = {
                    'timestamp': pd.to_datetime(kline_payload['t'], unit='ms', utc=True),
                    'open': float(kline_payload['o']), 'high': float(kline_payload['h']),
                    'low': float(kline_payload['l']), 'close': float(kline_payload['c']),
                    'volume': float(kline_payload['v']), 'quote_volume': float(kline_payload['q'])
                }
                new_kline_df_row = pd.DataFrame([new_kline_dict])
                if not all_base_klines_df.empty and new_kline_dict['timestamp'] in all_base_klines_df['timestamp'].values:
                    all_base_klines_df.loc[all_base_klines_df['timestamp'] == new_kline_dict['timestamp']] = new_kline_df_row.values
                else:
                    all_base_klines_df = pd.concat([all_base_klines_df, new_kline_df_row], ignore_index=True)
                all_base_klines_df = all_base_klines_df.drop_duplicates(subset=['timestamp'], keep='last').sort_values(by='timestamp').reset_index(drop=True)
                try:
                    base_td = pd.Timedelta(BASE_INTERVAL)
                    agg_td = pd.Timedelta(AGG_INTERVAL)
                    if base_td.total_seconds() == 0: base_intervals_per_agg = 1
                    else: base_intervals_per_agg = agg_td.total_seconds() / base_td.total_seconds()
                    max_base_rows_to_keep = int((HISTORICAL_AGG_CANDLES_TO_DISPLAY + 20) * base_intervals_per_agg)
                    if len(all_base_klines_df) > max_base_rows_to_keep:
                        all_base_klines_df = all_base_klines_df.iloc[-max_base_rows_to_keep:]
                except Exception as e_mem:
                    logging.error(f"计算max_base_rows_to_keep时出错: {e_mem}.")
                    default_max_rows = (HISTORICAL_AGG_CANDLES_TO_DISPLAY + 20) * 180
                    if len(all_base_klines_df) > default_max_rows:
                         all_base_klines_df = all_base_klines_df.iloc[-default_max_rows:]
                df_aggregated = aggregate_klines(all_base_klines_df, AGG_INTERVAL)
                if not df_aggregated.empty:
                    logging.info(f"\n--- 当前 {AGG_INTERVAL} OHLCV 数据 ({symbol_ws}) ---")
                    display_recent_count = 5
                    output_lines = []
                    base_interval_duration = pd.Timedelta(BASE_INTERVAL)
                    for i, row_agg in df_aggregated.tail(display_recent_count).iterrows():
                        status = "forming"
                        agg_candle_start_time = row_agg['timestamp']
                        agg_candle_end_time = agg_candle_start_time + pd.Timedelta(AGG_INTERVAL)
                        if not all_base_klines_df.empty:
                            latest_base_kline_start_time = all_base_klines_df['timestamp'].iloc[-1]
                            latest_base_kline_close_time = latest_base_kline_start_time + base_interval_duration
                            if latest_base_kline_close_time >= agg_candle_end_time: status = "closed"
                        if i < len(df_aggregated) -1: status = "closed"
                        line = (f"  Start: {row_agg['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z')}, "
                                f"O: {row_agg['open']:.2f}, H: {row_agg['high']:.2f}, L: {row_agg['low']:.2f}, C: {row_agg['close']:.2f}, "
                                f"V: {row_agg['volume']:.2f}, QV: {row_agg['quote_volume']:.2f}, Status: {status}")
                        output_lines.append(line)
                    print("\n".join(output_lines)); print("-" * 70)
                else: logging.info(f"尚未有足够的 {BASE_INTERVAL} 数据形成完整的 {AGG_INTERVAL} K线。")
        elif 'e' in data and data['e'] == 'error': logging.error(f"WebSocket API 错误消息: {data['m']}")
    except Exception as e:
        logging.error(f"WebSocket消息处理程序中出错: {e}", exc_info=True)
        logging.error(f"有问题的原始消息: {message}")

# --- 主应用程序 ---
def main():
    global all_base_klines_df, websocket_client_global, BASE_INTERVAL, AGG_INTERVAL, HISTORICAL_AGG_CANDLES_TO_DISPLAY

    # 1. 获取历史数据 (Assuming this part is now correct and working)
    try:
        base_td = pd.Timedelta(BASE_INTERVAL)
        agg_td = pd.Timedelta(AGG_INTERVAL)
        if base_td.total_seconds() == 0: raise ValueError(f"BASE_INTERVAL '{BASE_INTERVAL}' 解析为零持续时间。")
        if agg_td.total_seconds() == 0: raise ValueError(f"AGG_INTERVAL '{AGG_INTERVAL}' 解析为零持续时间。")
        base_intervals_per_agg = agg_td.total_seconds() / base_td.total_seconds()
        if base_intervals_per_agg <= 0: raise ValueError(f"计算的 base_intervals_per_agg ({base_intervals_per_agg}) 无效。")
    except Exception as e_calc:
        logging.error(f"计算间隔比率时出错: {e_calc}. " f"BASE_INTERVAL: '{BASE_INTERVAL}', AGG_INTERVAL: '{AGG_INTERVAL}'. " "无法确定要获取的历史K线数量。")
        return
    num_base_klines_needed_for_history = int((HISTORICAL_AGG_CANDLES_TO_DISPLAY + 5) * base_intervals_per_agg)
    logging.info(f"计算得到需要获取 {num_base_klines_needed_for_history} 个 '{BASE_INTERVAL}' {SYMBOL} 历史K线，以构建约 {HISTORICAL_AGG_CANDLES_TO_DISPLAY} 个 '{AGG_INTERVAL}' K线...")
    historical_base_klines_list = fetch_historical_klines_futures_rest(SYMBOL, BASE_INTERVAL, num_base_klines_needed_for_history)
    if not historical_base_klines_list:
        logging.error(f"未能获取足够的历史 '{BASE_INTERVAL}' K线。WebSocket将仅使用实时数据继续。")
        all_base_klines_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    else:
        all_base_klines_df = pd.DataFrame(historical_base_klines_list)
        all_base_klines_df['timestamp'] = pd.to_datetime(all_base_klines_df['timestamp'], utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            if col in all_base_klines_df.columns: all_base_klines_df[col] = pd.to_numeric(all_base_klines_df[col])
        all_base_klines_df = all_base_klines_df.drop_duplicates(subset=['timestamp'], keep='last').sort_values(by='timestamp').reset_index(drop=True)
        logging.info(f"成功获取并处理了 {len(all_base_klines_df)} 个历史 '{BASE_INTERVAL}' K线。")
        if not all_base_klines_df.empty:
            min_ts_hist = all_base_klines_df['timestamp'].min().isoformat()
            max_ts_hist = all_base_klines_df['timestamp'].max().isoformat()
            logging.info(f"已加载的历史 '{BASE_INTERVAL}' K线数据范围: 从 {min_ts_hist} 到 {max_ts_hist}")
        else: logging.info("all_base_klines_df 为空，在处理后。")

    # 初始聚合和显示历史数据 (Assuming this part is now correct and working)
    if not all_base_klines_df.empty:
        initial_agg_df = aggregate_klines(all_base_klines_df, AGG_INTERVAL)
        if not initial_agg_df.empty:
            display_count = min(HISTORICAL_AGG_CANDLES_TO_DISPLAY, len(initial_agg_df))
            logging.info(f"\n--- 初始约 {display_count} 个历史 {AGG_INTERVAL} OHLCV 数据 ({SYMBOL}) ---")
            output_lines = []
            for _, row in initial_agg_df.tail(display_count).iterrows():
                 line = (f"  Start: {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z')}, "
                         f"O: {row['open']:.2f}, H: {row['high']:.2f}, L: {row['low']:.2f}, C: {row['close']:.2f}, "
                         f"V: {row['volume']:.2f}, QV: {row['quote_volume']:.2f}, Status: closed")
                 output_lines.append(line)
            print("\n".join(output_lines)); print("-" * 70)
            if len(initial_agg_df) < HISTORICAL_AGG_CANDLES_TO_DISPLAY:
                logging.info(f"注意: 聚合出的历史K线数量 ({len(initial_agg_df)}) 少于期望的 {HISTORICAL_AGG_CANDLES_TO_DISPLAY} 条。")
        else: logging.warning(f"初始历史 '{BASE_INTERVAL}' 数据未产生任何 '{AGG_INTERVAL}' K线。")
    else: logging.warning("未加载历史数据，无法执行初始聚合。")

    logging.info(f"正在连接到币安WebSocket获取 {SYMBOL}@{BASE_INTERVAL}_kline 数据流...")
    websocket_client_global = UMFuturesWebsocketClient(on_message=message_handler)
    
    websocket_client_global.kline(
        symbol=SYMBOL.lower(),
        id=1,
        interval=BASE_INTERVAL
    )
    logging.info(f"已通过WebSocket订阅 {SYMBOL.lower()}@{BASE_INTERVAL}_kline 数据流。")

    try:
        while True:
            time.sleep(30)
            if websocket_client_global:
                try:
                    # The Binance WebsocketClient has an attribute 'ws_connection'
                    # which is an instance of 'websocket.WebSocketApp' from the 'websocket-client' library.
                    # This WebSocketApp instance has a 'keep_running' boolean attribute.
                    if hasattr(websocket_client_global, 'ws_connection') and \
                       websocket_client_global.ws_connection and \
                       hasattr(websocket_client_global.ws_connection, 'keep_running'):
                        
                        if not websocket_client_global.ws_connection.keep_running:
                            logging.warning(
                                "WebSocketApp 'keep_running' 标志为 False。WebSocket 可能已停止。正在终止程序。"
                            )
                            break
                    else:
                        # This case might occur if ws_connection is not yet initialized or is None,
                        # or if keep_running attribute is missing (very unlikely for websocket-client lib)
                        # If the client is not fully connected yet, ws_connection might not be fully set up.
                        # For a more direct status, we might need to reconsider how UMFuturesWebsocketClient
                        # exposes its state if 'status' attribute itself caused issues.
                        # However, if kline() succeeded, ws_connection should exist.
                        # Let's log if we can't find ws_connection.keep_running
                        if not hasattr(websocket_client_global, 'ws_connection') or not websocket_client_global.ws_connection:
                            logging.info("websocket_client_global.ws_connection 不存在或为 None。暂时无法检查 keep_running。")
                        elif not hasattr(websocket_client_global.ws_connection, 'keep_running'):
                             logging.info("websocket_client_global.ws_connection.keep_running 属性不存在。暂时无法检查。")
                        # If we cannot determine the status reliably here and the previous `status` check failed,
                        # we might have to rely on the message_handler to detect prolonged silence
                        # or the stop() in finally to clean up.
                        # For now, if keep_running isn't accessible, we might assume it's okay or log and continue.
                        # Given the prior AttributeError with 'status', being cautious.
                        # If it's simply not yet connected, it might not be an error to continue the loop.
                        pass # Continue loop, assuming it might be connecting or in an intermediate state


                except Exception as e_check_status: # Catch any other exception during status check
                    logging.error(f"检查WebSocket状态时发生意外错误: {e_check_status}. 正在终止程序。")
                    break
            else:
                logging.error("WebSocket客户端未初始化。正在终止程序。")
                break
    except KeyboardInterrupt:
        logging.info("检测到键盘中断。正在关闭...")
    except Exception as e:
        logging.error(f"主循环中发生意外错误: {e}", exc_info=True)
    finally:
        if websocket_client_global:
            if hasattr(websocket_client_global, 'stop') and callable(getattr(websocket_client_global, 'stop')):
                logging.info("正在停止WebSocket客户端...")
                websocket_client_global.stop()
                logging.info("已命令WebSocket客户端停止。")
            else:
                logging.warning("WebSocket客户端没有可调用的 'stop' 方法。")

        logging.info("程序已终止。")

if __name__ == "__main__":
    main()