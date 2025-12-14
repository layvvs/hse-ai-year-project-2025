FROM python:3.13-slim

WORKDIR /app

COPY . .

RUN pip3 install -r app/requirements.txt

CMD ["python3", "-m", "app"]
