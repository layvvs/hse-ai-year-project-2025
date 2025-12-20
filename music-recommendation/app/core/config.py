import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(BASE_DIR / '.env.local')
load_dotenv(BASE_DIR / '.env')


class AppConfig(BaseModel):
    host: str
    port: int


class DatabaseConfig(BaseModel):
    database_url: str


class LoggingConfig(BaseModel):
    enable_request_logging: bool = True

    log_blacklist: list[str] = ["/history"]

    max_error_length: int = 1000


def parse_config() -> AppConfig:
    return AppConfig(
        host=os.getenv('HOST', '0.0.0.0'),
        port=os.getenv('PORT', 6969)
    )


def parse_database_config() -> DatabaseConfig:
    return DatabaseConfig(database_url=os.getenv('DATABASE_URL'))


def parse_logging_config() -> LoggingConfig:
    return LoggingConfig()
