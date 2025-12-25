from app.vector_search.search import SearchEngine
from app.models.models import ForwardRequest


from json import JSONDecodeError
from pydantic import ValidationError
from fastapi import (
    APIRouter,
    Request,
    HTTPException,
    Depends,
)

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.session import get_session
from app.database.models.route_logs_model import RouteLog
from app.routers.auth import get_current_user, get_admin_user

router = APIRouter(prefix='/search', tags=['search'])


def get_search_engine(request: Request) -> SearchEngine:
    return request.app.state.search_engine


@router.post('/forward')
async def handle_forward(
    request: Request,
    search_engine: SearchEngine = Depends(get_search_engine),
    current_user=Depends(get_current_user),
):
    try:
        search_params_raw = await request.json()
        search_params = ForwardRequest(**search_params_raw)
        results = await search_engine.search(search_params)
        return results
    except (JSONDecodeError, ValidationError):
        raise HTTPException(
            status_code=400,
            detail='Некорректные данные, тело должно содержать ключи genres, instruments и tags типа array'
        )
    except Exception:
        raise HTTPException(
            status_code=403,
            detail='Модель не смогла обработать данные'
        )


@router.get('/history')
async def handle_history(session: AsyncSession = Depends(get_session)):
    hist = (
        select(RouteLog)
        .order_by(RouteLog.started_at.desc())
        .limit(500)
    )
    res = await session.execute(hist)
    logs = res.scalars().all()
    return [
        {
            'id': str(i.id),
            'started_at': i.started_at,
            'finished_at': i.finished_at,
            'route_path': i.route_path,
            'response_status': i.response_status,
            'duration_ms': i.duration_ms,
            'error_message': i.error_message,
        }
        for i in logs
    ]


@router.delete('/history')
async def handle_clear_history(
    session: AsyncSession = Depends(get_session),
    current_user=Depends(get_admin_user),
):
    await session.execute(RouteLog.__table__.delete())
    await session.commit()
    return {'result': 'ok'}


@router.get('/stats')
async def handle_stats(session: AsyncSession = Depends(get_session)):
    stat = select(
        func.count(RouteLog.id).label('total'),
        func.avg(RouteLog.duration_ms).label('avg_dur_ms'),
        func.min(RouteLog.duration_ms).label('min_dur_ms'),
        func.max(RouteLog.duration_ms).label('max_dur_ms'),
        func.percentile_cont(0.5).within_group(RouteLog.duration_ms).label('p50'),
        func.percentile_cont(0.95).within_group(RouteLog.duration_ms).label('p95'),
        func.percentile_cont(0.99).within_group(RouteLog.duration_ms).label('p99'),
        func.count(RouteLog.error_message).label('errors'),
        func.max(RouteLog.started_at).label('last_request_at'),
    )
    res = await session.execute(stat)
    out = res.one()
    success = out.total - out.errors
    error_rate = (out.errors / out.total) if out.total else 0
    return {
        'total_requests': out.total,
        'successful_requests': success,
        'errors': out.errors,
        'error_rate': error_rate,
        'avg_duration_ms': out.avg_dur_ms,
        'min_duration_ms': out.min_dur_ms,
        'max_duration_ms': out.max_dur_ms,
        'p50_duration_ms': out.p50,
        'p95_duration_ms': out.p95,
        'p99_duration_ms': out.p99,
        'last_request_at': out.last_request_at,
    }
