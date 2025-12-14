import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    host: str
    port: int


def parse_config():
    return AppConfig(
        host=os.getenv('HOST', '0.0.0.0'),
        port=os.getenv('PORT', 6969)
    )
