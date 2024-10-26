import logging
import os
from logging.handlers import TimedRotatingFileHandler


def setup_logger(name, log_file="./log/haikui_computation_detection.log", level=logging.DEBUG):
    """设置日志配置"""
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger
