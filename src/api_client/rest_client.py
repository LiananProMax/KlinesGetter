# src/api_client/rest_client.py
import requests
import time
import logging
from src.config import API_KEY, MAX_REST_RETRIES, INITIAL_REST_WAIT_TIME, PERIODS_FOR_HISTORY
from src.utils.network_utils import resolve_hostname

logger = logging.getLogger(__name__)

class CoinRestApiClient:
    def __init__(self, api_key, base_url):
        if not api_key:
            logger.error("CoinRestApiClient未配置API_KEY。")
            raise ValueError("CoinRestApiClient需要API_KEY")
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {'X-CoinAPI-Key': self.api_key}

    def _resolve_api_hostname(self, url_to_check):
        try:
            api_host = url_to_check.split('//')[1].split('/')[0]
            if not resolve_hostname(api_host):
                return False
        except IndexError:
            logger.error(f"API URL格式无效: {url_to_check}")
            return False
        return True

    def get_symbols(self):
        """使用重试逻辑从CoinAPI REST API获取交易对信息。"""
        url = self.base_url # 假设base_url是来自配置的symbols端点
        if not self._resolve_api_hostname(url):
            return None

        current_wait_time = INITIAL_REST_WAIT_TIME
        for attempt in range(MAX_REST_RETRIES):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                logger.info(f"在第{attempt + 1}次尝试中成功获取交易对。")
                return response.json()
            except requests.exceptions.HTTPError as e:
                logger.error(f"获取交易对时HTTP错误：{e.response.status_code} - {e.response.text}")
                if 500 <= e.response.status_code < 600:
                    logger.info(f"服务器错误，将在{current_wait_time}秒后重试...")
                else:
                    logger.error("客户端API错误。请检查请求或API密钥。")
                    return None
            except requests.exceptions.ConnectionError as e:
                logger.error(f"第{attempt + 1}次尝试连接错误：{e}")
            except requests.exceptions.Timeout as e:
                logger.error(f"第{attempt + 1}次尝试请求超时：{e}")
            except requests.exceptions.RequestException as e:
                logger.error(f"在第{attempt + 1}次REST API请求期间发生意外错误：{e}")

            if attempt < MAX_REST_RETRIES - 1:
                time.sleep(current_wait_time)
                current_wait_time = min(current_wait_time * 2, 60)
            else:
                logger.error("REST API超过最大重试次数。无法获取交易对。")
                return None
        return None

    def filter_and_select_symbol(self, symbols, base_asset='BTC', quote_asset='USDT', symbol_type='PERPETUAL'):
        """根据条件筛选交易对并选择一个。"""
        if not symbols:
            return None

        perpetual_symbols = [
            s for s in symbols
            if s.get('symbol_type') == symbol_type and
               s.get('asset_id_base') == base_asset and
               s.get('asset_id_quote') == quote_asset
        ]

        if not perpetual_symbols:
            logger.warning(f"未找到{base_asset}/{quote_asset} {symbol_type}交易对。")
            return None

        selected = perpetual_symbols[0]['symbol_id']
        logger.info(f"选择交易对ID: {selected}")
        return selected

    def get_ohlcv_history(self, symbol_id, period_id, time_start_iso, time_end_iso):
        """获取指定交易对的历史OHLCV数据。"""
        # 确保历史数据的base_url正确（可能与symbols端点不同）
        # 对于CoinAPI，通常是"https://rest.coinapi.io/v1"
        # 假设结构为：{scheme}://{host}/v1/ohlcv/...
        
        history_base_url = self.base_url.rsplit('/', 1)[0] # 获取类似https://rest.coinapi.io/v1的基础URL
        url = f"{history_base_url}/ohlcv/{symbol_id}/history?period_id={period_id}&time_start={time_start_iso}&time_end={time_end_iso}"

        if period_id not in PERIODS_FOR_HISTORY:
            logger.warning(f"请求的周期'{period_id}'不在配置的PERIODS_FOR_HISTORY {PERIODS_FOR_HISTORY}中。使用默认值'{PERIODS_FOR_HISTORY[0]}'。")
            period_id = PERIODS_FOR_HISTORY[0]
        
        if not self._resolve_api_hostname(url): # 检查特定历史URL的主机名
            return None

        logger.info(f"请求{symbol_id}的{period_id}周期历史OHLCV数据: {time_start_iso}至{time_end_iso}")
        current_wait_time = INITIAL_REST_WAIT_TIME
        for attempt in range(MAX_REST_RETRIES):
            try:
                response = requests.get(url, headers=self.headers, timeout=20) # 为历史数据增加超时时间
                response.raise_for_status()
                data = response.json()
                logger.info(f"成功获取{len(data)}条OHLCV数据点。")
                return data
            except requests.exceptions.HTTPError as e:
                logger.error(f"获取历史数据时HTTP错误：{e.response.status_code} - {e.response.text}")
                if 500 <= e.response.status_code < 600:
                    logger.info(f"服务器错误，将在{current_wait_time}秒后重试...")
                elif e.response.status_code == 429: # 速率限制
                    retry_after = e.response.headers.get('Retry-After')
                    wait_time = int(retry_after) if retry_after and retry_after.isdigit() else current_wait_time
                    logger.warning(f"达到速率限制。将在{wait_time}秒后重试。")
                    time.sleep(wait_time)
                    current_wait_time = min(current_wait_time * 2, 120) # 对于速率限制，使用更长的退避时间
                    continue # 继续下一次尝试迭代
                else:
                    logger.error("获取历史数据时客户端API错误。")
                    return None
            except requests.exceptions.ConnectionError as e:
                logger.error(f"第{attempt + 1}次尝试连接错误：{e}")
            except requests.exceptions.Timeout as e:
                logger.error(f"第{attempt + 1}次尝试请求超时：{e}")
            except requests.exceptions.RequestException as e:
                logger.error(f"第{attempt + 1}次历史数据请求期间发生意外错误：{e}")

            if attempt < MAX_REST_RETRIES - 1:
                time.sleep(current_wait_time)
                current_wait_time = min(current_wait_time * 2, 120) # 历史数据最大120秒
            else:
                logger.error("OHLCV历史API超过最大重试次数。")
        return None