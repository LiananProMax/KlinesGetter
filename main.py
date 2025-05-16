# main.py
import logging
from src.logger_config import setup_logging
from src.app import Application
from src import config # 提前检查API_KEY

if __name__ == "__main__":
    # 首先设置日志记录
    setup_logging() # 使用默认的INFO级别

    logger = logging.getLogger(__name__) # 在设置后获取logger

    # 在创建Application实例之前对API_KEY进行基本检查
    if not config.API_KEY:
        logger.critical("在.env或环境变量中未设置API_KEY。")
        logger.critical("请创建一个包含API_KEY的.env文件或将其设置为环境变量。")
        logger.critical("示例.env内容：")
        logger.critical("API_KEY=YOUR_API_KEY_HERE")
        # 为可能一开始不容易看到日志的用户打印语句
        print("错误：未设置API_KEY。请查看日志或README获取设置说明。")
    else:
        logger.info("找到API_KEY。启动应用程序...")
        app = Application()
        app.run()
