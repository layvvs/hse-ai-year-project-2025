from pydantic import BaseModel
from datetime import datetime


class SearchResult(BaseModel):
    song_name: str
    artist_name: str
    album_name: str
    releasedate: datetime
    vocalinstrumental: str
    artist_gender: str
    track_speed: str
    genres: list[str]
    instruments: list[str]
    tags: list[str]
    confidence: float


class SearchResultsList(BaseModel):
    search_results: list[SearchResult]


class AppConfig(BaseModel):
    host: str
    port: int
