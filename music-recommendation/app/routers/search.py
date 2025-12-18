from app.vector_search.search import SearchEngine
from app.models.models import ForwardRequest


from json import JSONDecodeError
from pydantic import ValidationError
from fastapi import (
    APIRouter,
    Request,
    HTTPException,
    Header,
    Depends,
)


router = APIRouter(prefix='/search', tags=['search'])


def get_search_engine(request: Request) -> SearchEngine:
    return request.app.state.search_engine


@router.post('/forward')
async def handle_forward(
    request: Request,
    search_engine: SearchEngine = Depends(get_search_engine)
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
async def handle_history():
    return {}


@router.delete('/history')
async def handle_clear_history(x_delete_token: str = Header(...)):
    return {}


@router.get('/stats')
async def handle_stats():
    return {}
