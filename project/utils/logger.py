import logging
import logging.handlers
from pathlib import Path


LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """모듈별 logger 반환. 최초 호출 시 핸들러 등록."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # 콘솔 핸들러 (INFO 이상)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    # 파일 핸들러 (DEBUG 이상, 10MB × 5개 순환)
    file_handler = logging.handlers.RotatingFileHandler(
        filename    = LOG_DIR / "app.log",
        maxBytes    = 10 * 1024 * 1024,
        backupCount = 5,
        encoding    = "utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger
