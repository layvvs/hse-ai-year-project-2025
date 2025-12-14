from fastapi import APIRouter, Request, Header, HTTPException
from json import JSONDecodeError


router = APIRouter(prefix='/search', tags=['search'])


@router.post('/forward')
async def handle_forward(request: Request):
    try:
        json_data = await request.json()
    except JSONDecodeError:
        return HTTPException(status_code=400, detail="Некорректные данные")
    return json_data


@router.get('/history')
async def handle_history():
    return {}


@router.delete('/history')
async def handle_clear_history(x_delete_token: str = Header(...)):
    return {}


@router.get('/stats')
async def handle_stats():
    return {}
