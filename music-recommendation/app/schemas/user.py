from pydantic import BaseModel, ConfigDict

class UserAuthSchema(BaseModel):
    username: str
    password: str

class UserReadSchema(BaseModel):
    id: int
    username: str
    is_active: bool
    is_admin: bool

    model_config = ConfigDict(from_attributes=True)