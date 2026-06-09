# Music Reco — frontend prototype

Streamlit-фронт. Все рекомендации идут через FastAPI-backend, который обёртывает Богданов pipeline (`ELSA + LightGBM lambdarank` из `checkpoint-7-ranker`).

## Что внутри
- **Синтетический user_id** (`u_<uuid12>`) — генерится при первом заходе, хранится в `st.session_state` + JSON в `data/user_history/<user_id>.json`. На backend он детерминированно мэппится через SHA1 в одного из обученных Yambda `uid` — так что одна и та же сессия всегда видит свой профиль.
- **Кнопки**: Назад · Не нравится · Пропустить · Нравится. Дизлайки/пропуски/лайки идут в exclude-лист, чтобы backend не предлагал одно и то же.
- **Плеер** — кастомный (HTML+JS, не нативный) с play/pause, перемоткой, таймером. Звук тянется по HTTP с `/audio/<stem>.mp3` бэкенда (Jamendo mp3 из `music/`).
- **Объяснение**: имя модели + текстовое обоснование + SHAP contributions (top-10) из `lgb.Booster.predict(..., pred_contrib=True)`. Зелёное — поднимает скор, красное — опускает.
- **История** — сайдбар: последние действия + Yambda uid, на который замапили пользователя.

## Запуск
```bash
# из корня репо, в двух терминалах:
uvicorn backend.main:app --host 127.0.0.1 --port 8000
streamlit run frontend/app.py
```
Если backend живёт не на `127.0.0.1:8000` — `BACKEND_URL=http://host:port streamlit run frontend/app.py`.

## Известный гэп
Богданова модель работает на Yambda `item_id` (целые числа), а в `music/` лежит Jamendo (string-id формата `1000957`). Перекрытия нет, поэтому **превью аудио — это Jamendo-трек, выбранный детерминированно по `item_id % len(catalog)`**. Сам item_id, score, rank и SHAP contributions — настоящие, от модели.
