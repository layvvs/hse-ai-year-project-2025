import os
from pathlib import Path
from typing import Optional
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


class SecurityConfig(BaseModel):
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    prod_domain: Optional[str] = None


def parse_config() -> AppConfig:
    return AppConfig(
        host=os.getenv('HOST', '0.0.0.0'),
        port=os.getenv('PORT', 6970)
    )


def parse_database_config() -> DatabaseConfig:
    return DatabaseConfig(database_url=os.getenv('DATABASE_URL'))


def parse_logging_config() -> LoggingConfig:
    return LoggingConfig()


def parse_security_config() -> SecurityConfig:
    return SecurityConfig(
        secret_key=os.getenv('SECRET_KEY', 'super_secret_default_key_CHANGE_ME'),
        algorithm=os.getenv('ALGORITHM', 'HS256'),
        access_token_expire_minutes=int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 30)),
        prod_domain=os.getenv('PROD_DOMAIN')
    )
