# src/utils/signal_utils.py
import signal
import logging

logger = logging.getLogger(__name__)

def setup_signal_handlers(shutdown_event):
    """
    为SIGINT和SIGTERM设置信号处理程序，以触发优雅关闭。
    """
    def handler(signum, frame):
        logger.info(f"收到信号 {signal.Signals(signum).name}。正在启动优雅关闭...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    logger.info("已注册SIGINT和SIGTERM的信号处理程序。")