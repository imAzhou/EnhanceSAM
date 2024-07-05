import logging

from mmengine.config import Config

def create_logger(logger_file_name, logger_name):
    """
    :param logger_file_name:
    :return:
    """
    logger = logging.getLogger(logger_name)         # 设定日志对象
    logger.setLevel(logging.INFO)        # 设定日志等级

    file_handler = logging.FileHandler(logger_file_name)   # 文件输出
    console_handler = logging.StreamHandler()              # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s : %(message)s ",
        "%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)       # 设置文件输出格式
    console_handler.setFormatter(formatter)    # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger