# Как запустить демо

Два процесса: **backend** (FastAPI с моделью) на `:8000` и **frontend** (Streamlit) на `:8501`. Запускаются в двух терминалах из корня репо.

## 0. Что должно быть на месте

```
hse-ai-year-project-2025/
├── backend/
├── frontend/
├── checkpoint-4/app/recommendations/    ← Богданов pipeline (с LFS-артефактами)
├── music/                               ← Jamendo mp3 (~2000 шт.)
└── metadata/                            ← Jamendo JSON-метаданные
```

Артефакты модели весят суммарно ~330MB и лежат в LFS. Если после клона они выглядят как 130-байтовые файлы (LFS-указатели) — подтяни их:

```bash
git lfs install
git lfs pull --include="checkpoint-4/app/recommendations/**"
```

## 1. Зависимости

Создай и активируй venv (Python 3.11), затем поставь зависимости обоих сервисов:

```bash
# Windows PowerShell / cmd
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# обе пачки зависимостей
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

## 2. Запуск backend

Терминал №1 (из корня репо):

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Первый старт грузит ELSA-пикл (~200MB) и считает feature_store — занимает 30–60 секунд. Когда увидишь `Application startup complete` — готов.

Быстрая проверка:

```bash
curl http://127.0.0.1:8000/health
```

Должно вернуть `{"status":"ok","known_users":6477,"n_items":200250,...}`.

## 3. Запуск frontend

Терминал №2 (из корня репо, тот же venv):

```bash
streamlit run frontend/app.py
```

Streamlit автоматически откроет браузер на `http://localhost:8501`. Если порт `:8501` занят — Streamlit сам подцепит следующий свободный.

## 4. Эндпойнты backend

| Эндпойнт | Что делает |
|---|---|
| `GET /health` | Статус + размеры (users / items / preview pool / feature count) |
| `GET /recommend?user_id=<str>&exclude=<csv-ids>&k=<int>` | Следующая рекомендация (track + model + SHAP contributions + rationale) |
| `GET /audio/{stem}.mp3` | Стриминг mp3 из `music/` |

Документация Swagger: `http://127.0.0.1:8000/docs`.

## 5. Конфигурация

Если backend крутится не на `127.0.0.1:8000`, фронту скажи через env-переменную:

```bash
# Windows PowerShell
$env:BACKEND_URL = "http://10.0.0.5:8000"; streamlit run frontend/app.py

# macOS / Linux
BACKEND_URL=http://10.0.0.5:8000 streamlit run frontend/app.py
```

## 6. Что куда писать данные

- `frontend/data/user_history/<user_id>.json` — история действий каждого синтетического пользователя (создаётся автоматически, gitignored).
- `backend/` ничего не пишет на диск.

## 7. Проблемы

**`ModuleNotFoundError: No module named 'app.recommendations'`** — `checkpoint-4/app/__init__.py` должен быть пустым. Если случайно поменял — верни пустой файл.

**Файлы модели читаются как 130 байт** — это LFS-указатели, не подтянутые блобы. Сделай `git lfs pull --include="checkpoint-4/app/recommendations/**"`.

**Backend стартует, но `/recommend` возвращает 500** — глянь логи uvicorn. Чаще всего это нехватка RAM на ELSA (нужно ~1.5GB свободной).

**Аудио не играет** — проверь что backend поднят с CORS-мидлварью (она уже в коде). В DevTools браузера → Network → запрос `/audio/...` должен быть 200, не CORS-блокирован.

**`Backend недоступен`** на фронте — uvicorn не поднят либо порт другой, см. п.5.
