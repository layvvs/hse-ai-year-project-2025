FROM python:3.13-slim

WORKDIR /app

COPY . .

RUN pip install -r app/requirements.txt

CMD ["sh", "-c", "cd app && alembic upgrade head && cd .. && python -m uvicorn app.__main__:app --host 0.0.0.0 --port 6970"]