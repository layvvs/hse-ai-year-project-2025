import time
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import parse_logging_config
from app.utils.logging import logger
from app.database.models.route_logs_model import RouteLog
from app.database.session import async_sessionmaker


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.logging_config = parse_logging_config()

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.should_log_request(request):
            return await call_next(request)

        route_log = self.prepare_route_log(request)
        start_time = time.time()

        try:
            response = await call_next(request)

            route_log.response_status = response.status_code
            route_log.response_headers = dict(response.headers)

        except Exception as e:
            route_log.error_message = str(e)[:self.logging_config.max_error_length]
            route_log.error_type = type(e).__name__[:200]
            raise

        finally:
            duration_ms = int((time.time() - start_time) * 1000)
            route_log.duration_ms = duration_ms
            route_log.finished_at = datetime.now(timezone.utc)

            if self.logging_config.enable_request_logging:
                asyncio.create_task(self.save_route_log(route_log))

        return response

    def should_log_request(self, request: Request) -> bool:
        if not self.logging_config.enable_request_logging:
            return False

        path = request.url.path
        for blacklisted in self.logging_config.log_blacklist:
            if path.startswith(blacklisted):
                return False

        return True

    def prepare_route_log(self, request: Request) -> RouteLog:
        return RouteLog(
            started_at=datetime.now(timezone.utc),
            route_path=str(request.url.path),
            request_headers=dict(request.headers),
            request_query_params=dict(request.query_params),
        )

    async def save_route_log(self, route_log: RouteLog):
        try:
            async with AsyncSessionLocal() as session:
                session.add(route_log)
                await session.commit()

        except asyncio.TimeoutError:
            pass

        except Exception as e:
            logger.warning(f"Failed to save route log: {e}")