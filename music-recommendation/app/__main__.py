from app.routers.search import router
from app.config import parse_config
from app.vector_search.search import SearchEngine

from fastapi import FastAPI
import uvicorn


app = FastAPI(title='Music recommendation system')
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    app.state.search_engine = SearchEngine()


if __name__ == "__main__":
    config = parse_config()
    uvicorn.run(app, host=config.host, port=config.port)
