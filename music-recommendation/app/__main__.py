from starlette.middleware.cors import CORSMiddleware

from app.core.middleware import RequestLoggerMiddleware

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
import uvicorn

from app.core.config import parse_config
from app.database.session import dispose_engine
from app.routers.search import router
from app.vector_search.search import SearchEngine
from utils.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Lifespan")

    app.state.search_engine = SearchEngine()

    yield

    await dispose_engine()


app = FastAPI(
    title='Music recommendation system',
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestLoggerMiddleware)
app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "Music Recommendation System API",
        "docs": "/docs"
    }


if __name__ == "__main__":
    config = parse_config()
    uvicorn.run(app, host=config.host, port=config.port)
