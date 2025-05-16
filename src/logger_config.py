# src/logger_config.py
import logging
import sys

def setup_logging(level=logging.INFO):
    """
    为应用程序配置基本日志记录。
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout  # 将日志输出到标准输出
    )
    # 如有必要，可以降低来自库的过于详细的日志记录器的级别
    # logging.getLogger("requests").setLevel(logging.WARNING)
    # logging.getLogger("urllib3").setLevel(logging.WARNING)
    # logging.getLogger("websocket").setLevel(logging.INFO) # 根据需要调整