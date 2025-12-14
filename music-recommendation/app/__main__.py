from app.routers.search import router
from app.config import parse_config

from fastapi import FastAPI
import uvicorn


app = FastAPI(title='Music recommendation system')
app.include_router(router)


if __name__ == "__main__":
    config = parse_config()
    uvicorn.run(app, host=config.host, port=config.port)
