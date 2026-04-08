from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import String, DateTime, JSON, Integer, Index
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID
from app.database.session import Base
import uuid


class RouteLog(Base):
    __tablename__ = "route_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    route_path: Mapped[str] = mapped_column(String, nullable=False)

    request_headers: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    request_query_params: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    response_status: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    response_headers: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    error_message: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    error_type: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    __table_args__ = (
        Index('idx_route_logs_route_path', 'route_path'),
        Index('idx_route_logs_duration', 'duration_ms'),
    )
