from routers import search
from config import parse_config

from fastapi import FastAPI
import uvicorn


app = FastAPI(title='Music recommendation system')
app.include_router(search.router)


if __name__ == "__main__":
    config = parse_config()
    uvicorn.run(app, host=config.host, port=config.port)
