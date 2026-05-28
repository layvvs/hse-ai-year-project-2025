from pydantic import BaseModel, field_validator
from datetime import date


class SearchResult(BaseModel):
    song_name: str
    artist_name: str
    album_name: str
    releasedate: date
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


class ForwardRequest(BaseModel):
    genres: list[str]
    instruments: list[str]
    tags: list[str]

    @field_validator('genres', 'instruments', 'tags', mode='before')
    @classmethod
    def set_unknown_if_empty(cls, v):
        if not v:
            return ['unknown']
        return v

    def get_all_tags(self) -> tuple[list[str]]:
        return (self.genres, self.instruments, self.tags)
