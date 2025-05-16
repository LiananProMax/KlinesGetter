# src/data_handlers/message_processor.py
import json
import logging
from src.config import WS_OHLCV_INTERESTED_PERIODS

logger = logging.getLogger(__name__)

def process_websocket_message(message_str):
    """
    处理来自WebSocket的原始消息字符串，解码JSON，并记录日志。
    这里是添加保存到数据库或进一步处理数据的逻辑的地方。
    """
    try:
        data = json.loads(message_str)
        message_type = data.get("type")

        if message_type == "heartbeat":
            logger.info("收到服务器心跳消息。")
        elif message_type == "ohlcv":
            period_id = data.get("period_id", "")
            if period_id in WS_OHLCV_INTERESTED_PERIODS:
                logger.info(f"收到{period_id}周期的OHLCV数据：{data}")
                # TODO: 在这里添加数据处理逻辑（例如，保存到数据库）
            else:
                logger.debug(f"忽略未监控周期'{period_id}'的OHLCV数据。")
        # 移除对交易数据的处理，因为我们不再订阅交易数据
        # elif message_type == "trade":
        #     logger.info(f"收到交易数据：{data}")
        #     # TODO: 在这里添加数据处理逻辑
        elif message_type == "error":
            logger.error(f"从WebSocket收到错误消息：{data.get('message')}")
        else:
            logger.info(f"收到其他类型消息'{message_type}'：{data}")

    except json.JSONDecodeError:
        logger.error(f"解码JSON消息失败：{message_str}")
    except Exception as e:
        logger.error(f"处理WebSocket消息时出错：{e}")