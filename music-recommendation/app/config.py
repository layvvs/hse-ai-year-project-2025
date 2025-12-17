import os
from app.models.models import AppConfig


def parse_config():
    return AppConfig(
        host=os.getenv('HOST', '0.0.0.0'),
        port=os.getenv('PORT', 6969)
    )
