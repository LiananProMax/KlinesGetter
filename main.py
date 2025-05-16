import requests
import json
import websocket
import socket
import time
import os
import logging
import signal
import threading
from dotenv import load_dotenv

# --- 配置 ---
# 从.env文件加载环境变量
load_dotenv()

# API和WebSocket配置
API_KEY = os.getenv('API_KEY')
REST_API_URL = os.getenv('REST_API_URL', 'https://rest.coinapi.io/v1/symbols')
WS_URL = os.getenv('WS_URL', 'wss://ws.coinapi.io/v1/')

# 重试和超时配置
MAX_REST_RETRIES = int(os.getenv('MAX_REST_RETRIES', 5))
INITIAL_REST_WAIT_TIME = int(os.getenv('INITIAL_REST_WAIT_TIME', 2)) # 增加初始等待时间
MAX_WS_RECONNECT_ATTEMPTS = int(os.getenv('MAX_WS_RECONNECT_ATTEMPTS', 10)) # WebSocket重连最大尝试次数
WS_INITIAL_BACKOFF_SECONDS = int(os.getenv('WS_INITIAL_BACKOFF_SECONDS', 5))
WS_MAX_BACKOFF_SECONDS = int(os.getenv('WS_MAX_BACKOFF_SECONDS', 60))
WS_INACTIVITY_TIMEOUT = int(os.getenv('WS_INACTIVITY_TIMEOUT', 70)) # 秒（CoinAPI每55秒发送一次心跳）

# --- 全局变量 ---
# 用于发出优雅关闭信号的事件
shutdown_event = threading.Event()
# 全局存储选定的symbol_id以供WebSocket使用
selected_symbol_id = None
# WebSocketApp实例
ws_app = None
# 用于检查不活动状态的计时器
inactivity_timer = None
# 用于ws_app线程安全操作的锁（例如，手动关闭）
ws_lock = threading.Lock()


# --- 日志设置 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- DNS解析 ---
def resolve_hostname(host):
    """将主机名解析为IP地址。"""
    try:
        socket.gethostbyname(host)
        logger.info(f"成功解析主机名：{host}")
        return True
    except socket.gaierror as e:
        logger.error(f"无法解析主机名'{host}'：{e}。请检查您的网络连接或DNS设置。")
        return False

# --- REST API函数 ---
def get_symbols_from_rest_api():
    """使用重试逻辑从CoinAPI REST API获取交易对信息。"""
    headers = {'X-CoinAPI-Key': API_KEY}
    current_wait_time = INITIAL_REST_WAIT_TIME

    if not API_KEY:
        logger.error("在环境变量中未找到API_KEY。退出。")
        return None

    # 从REST_API_URL提取主机名进行DNS检查
    try:
        rest_api_host = REST_API_URL.split('//')[1].split('/')[0]
        if not resolve_hostname(rest_api_host):
            return None
    except IndexError:
        logger.error(f"REST_API_URL格式无效：{REST_API_URL}")
        return None


    for attempt in range(MAX_REST_RETRIES):
        try:
            response = requests.get(REST_API_URL, headers=headers, timeout=10) # Added timeout
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            logger.info(f"在第{attempt + 1}次尝试中成功从REST API获取交易对。")
            return response.json()

        except requests.exceptions.HTTPError as e:
            logger.error(f"获取交易对时HTTP错误：{e.response.status_code} - {e.response.text}")
            if 500 <= e.response.status_code < 600:
                logger.info(f"服务器错误，将在{current_wait_time}秒后重试...")
            else: # 客户端错误（4xx）通常不可重试
                logger.error("客户端API错误。请检查您的请求或API密钥。")
                return None # 对于客户端错误不重试
        except requests.exceptions.ConnectionError as e:
            logger.error(f"第{attempt + 1}次尝试连接错误：{e}")
        except requests.exceptions.Timeout as e:
            logger.error(f"第{attempt + 1}次尝试请求超时：{e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"在第{attempt + 1}次REST API请求期间发生意外错误：{e}")

        if attempt < MAX_REST_RETRIES - 1:
            time.sleep(current_wait_time)
            current_wait_time = min(current_wait_time * 2, 60) # 指数退避，最大60秒
        else:
            logger.error("REST API超过最大重试次数。无法获取交易对。")
            return None
    return None


def filter_and_select_symbol(symbols):
    """筛选BTC/USDT永续合约交易对并选择一个。"""
    if not symbols:
        return None

    perpetual_symbols = [
        s for s in symbols
        if s.get('symbol_type') == 'PERPETUAL' and
           s.get('asset_id_base') == 'BTC' and
           s.get('asset_id_quote') == 'USDT'
    ]

    if not perpetual_symbols:
        logger.warning("未找到BTC/USDT永续合约交易对。")
        return None

    selected = perpetual_symbols[0]['symbol_id']
    logger.info(f"使用交易对ID：{selected}")
    return selected

# --- OHLCV历史数据函数 ---
def get_ohlcv_history(symbol_id, period_id, time_start, time_end):
    """获取指定交易对的历史OHLCV数据。
    
    参数:
        symbol_id (str): 交易对ID
        period_id (str): 时间周期，例如"1MIN", "3MIN", "1HRS"
        time_start (str): 开始时间，ISO 8601格式，例如"2024-05-15T00:00:00"
        time_end (str): 结束时间，ISO 8601格式，例如"2024-05-16T00:00:00"
        
    返回:
        list: OHLCV数据列表，如果请求失败则返回None
    """
    if not API_KEY:
        logger.error("在环境变量中未找到API_KEY。无法获取历史数据。")
        return None
        
    url = f"https://rest.coinapi.io/v1/ohlcv/{symbol_id}/history?period_id={period_id}&time_start={time_start}&time_end={time_end}"
    headers = {'X-CoinAPI-Key': API_KEY}
    current_wait_time = INITIAL_REST_WAIT_TIME
    
    logger.info(f"正在请求{period_id}周期的历史OHLCV数据: {time_start} 至 {time_end}")
    
    for attempt in range(MAX_REST_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"成功获取{len(data)}条历史OHLCV数据")
            return data
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"获取历史数据时HTTP错误：{e.response.status_code} - {e.response.text}")
            if 500 <= e.response.status_code < 600:
                logger.info(f"服务器错误，将在{current_wait_time}秒后重试...")
            elif e.response.status_code == 429:
                logger.warning("API速率限制已达到，将在退避后重试")
                # 从响应标头获取建议的等待时间（如果有）
                retry_after = e.response.headers.get('Retry-After')
                if retry_after and retry_after.isdigit():
                    wait_time = int(retry_after)
                else:
                    wait_time = current_wait_time
                    # 指数退避，每次重试等待时间翻倍
                    current_wait_time = min(current_wait_time * 2, 60)
                logger.info(f"等待{wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            else: # 其他客户端错误
                logger.error("客户端API错误。请检查您的请求或API密钥。")
                return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"第{attempt + 1}次尝试连接错误：{e}")
        except requests.exceptions.Timeout as e:
            logger.error(f"第{attempt + 1}次尝试请求超时：{e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"第{attempt + 1}次REST API请求期间发生意外错误：{e}")
            
        if attempt < MAX_REST_RETRIES - 1:
            time.sleep(current_wait_time)
            current_wait_time = min(current_wait_time * 2, 60) # 指数退避，最大60秒
        else:
            logger.error("历史数据API超过最大重试次数。无法获取数据。")
    
    return None

# --- WebSocket函数 ---

def reset_inactivity_timer():
    """重置或启动不活动计时器。"""
    global inactivity_timer
    if inactivity_timer:
        inactivity_timer.cancel()

    # 在启动新计时器之前检查ws_app是否存在且其sock是否打开
    # 这可以防止在WebSocket已经关闭或正在关闭时启动计时器
    if ws_app and ws_app.sock and ws_app.sock.connected:
        inactivity_timer = threading.Timer(WS_INACTIVITY_TIMEOUT, handle_inactivity)
        inactivity_timer.daemon = True # 允许程序退出，即使计时器仍处于活动状态
        inactivity_timer.start()

def handle_inactivity():
    """处理WebSocket不活动状态。"""
    logger.warning(f"{WS_INACTIVITY_TIMEOUT}秒内未收到消息。连接可能已过时。尝试重新连接。")
    with ws_lock:
        if ws_app:
            # ws_app.close()将触发on_close，它处理重新连接。
            # 如果可能，使用非阻塞关闭，或确保它不会死锁。
            # 对于websocket-client，close()可能是阻塞的。
            # 如果close()在这里有问题，直接触发重新连接逻辑会更安全。
            # 但是，如果close()表现良好，让on_close处理它会更干净。
            try:
                if ws_app.sock and ws_app.sock.connected:
                    ws_app.close()
            except Exception as e:
                logger.error(f"由于不活动关闭WebSocket时出错：{e}")
            # 如果close不能可靠或快速地触发on_close，
            # 你可能需要直接向主循环发出重新连接的信号。

def on_message(ws, message):
    """收到消息时的回调函数。"""
    try:
        reset_inactivity_timer() # 收到任何消息时重置计时器
        data = json.loads(message)
        
        if data.get("type") == "heartbeat":
            logger.info("收到服务器的心跳消息。")
        elif data.get("type") == "ohlcv":
            # 获取消息中的周期ID
            period_id = data.get("period_id", "")
            # 过滤出我们关心的周期（1分钟、3分钟、1小时）
            if period_id in ["1MIN", "3MIN", "1HRS"]:
                logger.info(f"收到{period_id}周期的OHLCV数据：{data}")
                # TODO: 在这里添加数据处理逻辑，例如保存到数据库或文件
            else:
                logger.debug(f"收到不关心的周期({period_id})的OHLCV数据，已忽略。")
        elif data.get("type") == "trade":
            logger.info(f"收到交易数据：{data}")
        else:
            logger.info(f"收到其他类型数据：{data}")
    except json.JSONDecodeError:
        logger.error(f"解码JSON消息失败：{message}")
    except Exception as e:
        logger.error(f"处理消息时出错：{e}")

def on_error(ws, error):
    """发生错误时的回调函数。"""
    logger.error(f"WebSocket错误：{error}")
    # 错误发生后通常会调用on_close，它处理重新连接。
    # 如果没有，可能需要在这里也触发重新连接逻辑。

def on_close(ws, close_status_code, close_msg):
    """连接关闭时的回调函数。"""
    global inactivity_timer
    if inactivity_timer:
        inactivity_timer.cancel()
        inactivity_timer = None

    if shutdown_event.is_set():
        logger.info("WebSocket连接已优雅关闭。")
    else:
        logger.warning(f"WebSocket连接意外关闭。状态：{close_status_code}，消息：{close_msg}。将尝试重新连接。")
        # 主循环将处理重新连接。

def on_open(ws):
    """连接打开时的回调函数。"""
    global selected_symbol_id
    logger.info("WebSocket连接已打开。")
    if not selected_symbol_id:
        logger.error("未设置selected_symbol_id。无法订阅。")
        ws.close() # 如果缺少symbol_id则关闭连接
        return

    hello_message = {
        "type": "hello",
        "apikey": API_KEY,
        "subscribe_data_type": ["ohlcv", "trade"],  # 订阅OHLCV和交易数据
        "subscribe_filter_symbol_id": [selected_symbol_id]
    }
    try:
        ws.send(json.dumps(hello_message))
        logger.info(f"已发送交易对订阅请求：{selected_symbol_id}")
        reset_inactivity_timer() # 成功连接和订阅后启动不活动计时器
    except websocket.WebSocketConnectionClosedException:
        logger.error("发送hello消息失败：WebSocket连接已关闭。")
    except Exception as e:
        logger.error(f"发送hello消息时出错：{e}")
        ws.close() # 发送过程中出错时关闭

def connect_websocket():
    """创建并运行WebSocket连接。"""
    global ws_app, selected_symbol_id

    if not selected_symbol_id:
        logger.error("无法连接到WebSocket：未选择交易对ID。")
        return False # 表示连接失败

    if not API_KEY:
        logger.error("在环境变量中未找到API_KEY。无法连接到WebSocket。")
        return False

    # 从WS_URL提取主机名进行DNS检查
    try:
        ws_host = WS_URL.split('//')[1].split('/')[0]
        if not resolve_hostname(ws_host):
            return False
    except IndexError:
        logger.error(f"WS_URL格式无效：{WS_URL}")
        return False

    logger.info(f"尝试连接到WebSocket：{WS_URL}")
    with ws_lock:
        ws_app = websocket.WebSocketApp(WS_URL,
                                        on_open=on_open,
                                        on_message=on_message,
                                        on_error=on_error,
                                        on_close=on_close)
    try:
        ws_app.run_forever(ping_interval=20, ping_timeout=10)
    except Exception as e:
        logger.error(f"WebSocket run_forever遇到错误：{e}")

    # 当run_forever退出时（例如，连接关闭）会执行到这部分
    with ws_lock:
        ws_app = None # 清除全局实例

    if shutdown_event.is_set():
        logger.info("由于关闭信号，WebSocket run_forever循环已退出。")
        return False # 如果正在关闭，不要尝试重新连接
    else:
        logger.info("WebSocket run_forever循环已退出。如果尝试次数允许，将尝试重新连接。")
        return True # 表示应该尝试重新连接


# --- 优雅关闭的信号处理 ---
def signal_handler(signum, frame):
    """处理终止信号以实现优雅关闭。"""
    logger.info(f"收到信号 {signal.Signals(signum).name}。正在启动优雅关闭...")
    shutdown_event.set()
    with ws_lock:
        if ws_app:
            logger.info("正在关闭WebSocket连接...")
            ws_app.close() # 这应该会触发on_close
    if inactivity_timer:
        inactivity_timer.cancel()

# --- 主应用逻辑 ---
def main():
    """主应用函数。"""
    global selected_symbol_id, ws_app

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("应用程序启动中...")

    # 步骤1和2：获取并筛选交易对
    symbols_data = get_symbols_from_rest_api()
    if not symbols_data:
        logger.error("获取交易对数据失败。退出。")
        return

    selected_symbol_id = filter_and_select_symbol(symbols_data)
    if not selected_symbol_id:
        logger.error("选择交易对失败。退出。")
        return
        
    # 获取示例：获取历史OHLCV数据（最近1小时的1分钟K线数据）
    try:
        from datetime import datetime, timedelta
        # 计算最近1小时的时间范围
        time_end = datetime.now().isoformat()
        time_start = (datetime.now() - timedelta(hours=1)).isoformat()
        
        # 获取1分钟周期的历史数据
        logger.info("获取1分钟周期历史数据示例...")
        ohlcv_1min = get_ohlcv_history(selected_symbol_id, "1MIN", time_start, time_end)
        if ohlcv_1min:
            logger.info(f"1分钟周期数据示例: {ohlcv_1min[0] if ohlcv_1min else 'No data'}")
        
        # 获取3分钟周期的历史数据
        logger.info("获取3分钟周期历史数据示例...")
        ohlcv_3min = get_ohlcv_history(selected_symbol_id, "3MIN", time_start, time_end)
        if ohlcv_3min:
            logger.info(f"3分钟周期数据示例: {ohlcv_3min[0] if ohlcv_3min else 'No data'}")
            
        # 获取1小时周期的历史数据（需要更大的时间范围才有意义）
        time_start_1h = (datetime.now() - timedelta(days=7)).isoformat()
        logger.info("获取1小时周期历史数据示例...")
        ohlcv_1h = get_ohlcv_history(selected_symbol_id, "1HRS", time_start_1h, time_end)
        if ohlcv_1h:
            logger.info(f"1小时周期数据示例: {ohlcv_1h[0] if ohlcv_1h else 'No data'}")
    except Exception as e:
        logger.warning(f"获取历史数据示例时出错: {e}")
        logger.info("继续执行WebSocket连接流程...")

    # 步骤3：带有重连逻辑的WebSocket连接
    reconnect_attempts = 0
    backoff_time = WS_INITIAL_BACKOFF_SECONDS

    while not shutdown_event.is_set() and (MAX_WS_RECONNECT_ATTEMPTS == 0 or reconnect_attempts < MAX_WS_RECONNECT_ATTEMPTS):
        if connect_websocket(): # 如果返回True，表示它已退出，应该尝试重新连接
            if shutdown_event.is_set(): # 在connect_websocket返回后再次检查
                break

            reconnect_attempts += 1
            logger.info(f"WebSocket连接丢失/失败。尝试在{backoff_time}秒内重新连接 {reconnect_attempts}/{MAX_WS_RECONNECT_ATTEMPTS if MAX_WS_RECONNECT_ATTEMPTS > 0 else '无限'}...")

            # 等待backoff_time，但频繁检查shutdown_event
            wait_start_time = time.time()
            while time.time() - wait_start_time < backoff_time:
                if shutdown_event.is_set():
                    logger.info("在退避期间启动关闭。中止重新连接。")
                    break
                time.sleep(0.5) # 每0.5秒检查一次shutdown_event

            if shutdown_event.is_set():
                break

            backoff_time = min(backoff_time * 2, WS_MAX_BACKOFF_SECONDS)
        else:
            # connect_websocket返回False，意味着初始设置失败或触发了关闭
            if not shutdown_event.is_set(): # 如果不是优雅关闭，可能是永久性设置错误
                logger.error("在建立连接之前，WebSocket连接过程严重失败。退出。")
            break # 如果connect_websocket表示不重试，则退出循环（例如，设置错误或关闭）


    if shutdown_event.is_set():
        logger.info("应用程序关闭完成。")
    elif reconnect_attempts >= MAX_WS_RECONNECT_ATTEMPTS and MAX_WS_RECONNECT_ATTEMPTS > 0:
        logger.error("达到最大WebSocket重连尝试次数。退出。")
    else:
        logger.info("应用程序已完成。")


if __name__ == "__main__":
    # 如果直接运行此脚本，确保加载.env变量是一个好习惯
    # 例如，通过在同一目录中创建一个.env文件，内容如下：
    # API_KEY="YOUR_API_KEY_HERE"
    # REST_API_URL="https://rest.coinapi.io/v1/symbols"
    # WS_URL="wss://ws.coinapi.io/v1/"
    # MAX_REST_RETRIES=5
    # INITIAL_REST_WAIT_TIME=2
    # MAX_WS_RECONNECT_ATTEMPTS=10 # 0表示无限制
    # WS_INITIAL_BACKOFF_SECONDS=5
    # WS_MAX_BACKOFF_SECONDS=60
    # WS_INACTIVITY_TIMEOUT=70 # CoinAPI每55秒发送一次心跳
    #
    # 支持的OHLCV周期包括：
    # 秒级: 1SEC, 2SEC, 3SEC, 4SEC, 5SEC, 6SEC, 10SEC, 15SEC, 20SEC, 30SEC
    # 分钟级: 1MIN, 2MIN, 3MIN, 4MIN, 5MIN, 6MIN, 10MIN, 15MIN, 20MIN, 30MIN
    # 小时级: 1HRS, 2HRS, 3HRS, 4HRS, 6HRS, 8HRS, 12HRS
    # 天级: 1DAY, 2DAY, 3DAY, 5DAY, 7DAY, 10DAY

    if not API_KEY:
        print("错误：API_KEY未在环境变量或.env文件中设置。")
        print("请创建一个包含API_KEY的.env文件或将其设置为环境变量。")
    else:
        main()
