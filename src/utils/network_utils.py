# src/utils/network_utils.py
import socket
import logging

logger = logging.getLogger(__name__)

def resolve_hostname(host):
    """将主机名解析为IP地址。"""
    try:
        socket.gethostbyname(host)
        logger.info(f"成功解析主机名：{host}")
        return True
    except socket.gaierror as e:
        logger.error(f"无法解析主机名'{host}'：{e}。请检查网络或DNS设置。")
        return False