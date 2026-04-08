from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.models.users_model import User


class UserDAO:
    @classmethod
    async def find_by_username(cls, session: AsyncSession, username: str):
        query = select(User).where(User.username == username)
        result = await session.execute(query)
        return result.scalar_one_or_none()

    @classmethod
    async def find_by_id(cls, session: AsyncSession, user_id: int):
        return await session.get(User, user_id)

    @classmethod
    async def create(cls, session: AsyncSession, **user_data):
        user = User(**user_data)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user
