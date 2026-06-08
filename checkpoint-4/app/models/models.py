from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    item_id: int
    score: float
    rank: int


class SearchResultsList(BaseModel):
    search_results: list[SearchResult]


class AppConfig(BaseModel):
    host: str
    port: int


class ForwardRequest(BaseModel):
    uid: int
    k: int = Field(default=10, ge=1, le=100)
