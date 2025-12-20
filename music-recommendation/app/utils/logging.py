import logging
import sys
from typing import Optional


class AppLogger:
    _instance: Optional['AppLogger'] = None
    _logger: Optional[logging.Logger] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup_logger()
        return cls._instance

    def _setup_logger(self):
        self._logger = logging.getLogger('music-recommendation')
        self._logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        try:
            file_handler = logging.FileHandler('app.log', encoding='utf-8')
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
        except Exception:
            pass

    def info(self, message: str):
        self._logger.info(message)

    def error(self, message: str):
        self._logger.error(message)

    def warning(self, message: str):
        self._logger.warning(message)

    def debug(self, message: str):
        self._logger.debug(message)

    def critical(self, message: str):
        self._logger.critical(message)


logger = AppLogger()