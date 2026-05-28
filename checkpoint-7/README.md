# Music Recommendation System

- **backend** — FastAPI, ELSA + LightGBM reranker
- **frontend** — Streamlit UI с плеером и SHAP-объяснениями
- **recommendations** — пайплайн инференса и артефакты модели

## Структура

```
demo/
├── backend/           # FastAPI: /health, /recommend, /audio
├── frontend/          # Streamlit UI
├── recommendations/   # engine.py, models.py, data/, models_weights/
├── music/             # Jamendo mp3 для превью (опционально)
├── metadata/          # Jamendo JSON-метаданные (опционально)
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
└── Makefile
```

## Быстрый старт (Docker)

```bash
# Пулит веса моделей и данные
git lfs pull

cd demo

# Собрать и запустить
make run
```

Открой в браузере:

| Сервис | URL |
|--------|-----|
| Frontend (UI) | http://localhost:8501 |
| Swagger | http://localhost:8000/docs |

Первый старт backend занимает **30–60 секунд** — грузится ELSA и строится feature store.

## Makefile

| Команда | Что делает |
|---------|------------|
| `make build` | Собирает Docker-образы |
| `make run` | `build` + `docker compose up -d` |
| `make stop` | Останавливает контейнеры |
| `make restart` | Перезапуск |
| `make logs` | Логи обоих сервисов |
| `make health` | Проверка `GET /health` |
| `make clean` | Остановка + удаление локальных образов |


## Аудио-превью (опционально)

Модель рекомендует **Yambda item_id**, а для проигрывания используются **Jamendo mp3** из `music/`. 

Без `music/` рекомендации работают, но плеер покажет «Аудио-превью недоступно».

## Как работают предсказания

Система — **двухступенчатая**: сначала ELSA быстро сужает каталог до кандидатов, потом LightGBM переранжирует их по скору

### Общая схема

```
Браузер (http://localhost:8501)
    │
    │  like / dislike / skip
    ▼
Streamlit frontend
    │  GET /recommend?user_id=u_abc&exclude=7092180,2981607,...
    │  (BACKEND_INTERNAL_URL → http://backend:8000 в Docker)
    ▼
FastAPI backend
    │  1. user_id → yambda_uid (SHA1)
    │  2. ELSA → top-100 кандидатов
    │  3. сбор 27 фичей на каждую пару (uid, item_id)
    │  4. LightGBM lambdarank → score + сортировка
    │  5. первый item_id ∉ exclude
    │  6. SHAP (pred_contrib) для объяснения
    │  7. Jamendo preview (title/artist/mp3) — отдельно от модели
    ▼
Ответ: { track, model, contributions, rationale }
```
