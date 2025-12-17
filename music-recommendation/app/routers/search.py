from json import JSONDecodeError
from fastapi import (
    APIRouter,
    Request,
    HTTPException,
    Header,
    Depends,
)
from app.vector_search.search import SearchEngine
from app.vector_search.utils import Tags


router = APIRouter(prefix='/search', tags=['search'])


def get_search_engine(request: Request) -> SearchEngine:
    return request.app.state.search_engine


@router.post('/forward')
async def handle_forward(
    request: Request,
    search_engine: SearchEngine = Depends(get_search_engine)
):
    try:
        json_data: Tags = await request.json()
        results = await search_engine.search(json_data)
        return results
    except JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректные данные")


@router.get('/history')
async def handle_history():
    return {}


@router.delete('/history')
async def handle_clear_history(x_delete_token: str = Header(...)):
    return {}


@router.get('/stats')
async def handle_stats():
    return {}
