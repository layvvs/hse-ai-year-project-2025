from fastapi import APIRouter, HTTPException, Request, Depends
from jose import jwt, JWTError
from app.schemas.user import UserAuthSchema, UserReadSchema
from app.database.dao.user_dao import UserDAO
from app.core.security import verify_password, create_access_token, get_password_hash, security_settings
from app.database.session import session_maker


router = APIRouter(prefix='/auth', tags=['Auth'])


async def get_current_user(request: Request):
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Вы не авторизованы')
    token = auth_header.split(' ', 1)[1]

    try:
        payload = jwt.decode(token, security_settings.secret_key, algorithms=[security_settings.algorithm])
        user_id: str = payload.get('sub')
        if user_id is None:
            raise HTTPException(status_code=401, detail='Неверный токен')
    except JWTError:
        raise HTTPException(status_code=401, detail='Токен испорчен или истек')

    async with session_maker() as session:
        user = await UserDAO.find_by_id(session, int(user_id))
        if user is None:
            raise HTTPException(status_code=401, detail='Пользователь не найден')
        return user


async def get_admin_user(current_user=Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail='Недостаточно прав')
    return current_user


@router.post('/register', response_model=UserReadSchema)
async def register(user_data: UserAuthSchema):
    async with session_maker() as session:
        existing_user = await UserDAO.find_by_username(session, user_data.username)
        if existing_user:
            raise HTTPException(status_code=409, detail='Пользователь уже существует')

        new_user = await UserDAO.create(
            session,
            username=user_data.username,
            hashed_password=get_password_hash(user_data.password)
        )
    return new_user


@router.post('/login')
async def login(user_data: UserAuthSchema):
    async with session_maker() as session:
        user = await UserDAO.find_by_username(session, user_data.username)

    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail='Неверный логин или пароль')

    access_token = create_access_token(data={'sub': str(user.id)})
    return {'access_token': access_token, 'token_type': 'bearer'}
