"""HTTP client for the FastAPI backend.

The UI talks only to `recommend()`. Backend (uvicorn) runs separately and is
responsible for actually invoking Bogdan's ELSA + LightGBM pipeline.

Set BACKEND_URL env var to point at a different host/port (default
http://127.0.0.1:8000).
"""

from __future__ import annotations

import os
from typing import Any
from urllib import error, parse, request

import json as _json

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

_HEALTH_CACHE: dict[str, Any] | None = None


def _get_json(path: str, params: dict[str, str] | None = None, timeout: float = 30.0) -> Any:
    url = f"{BACKEND_URL}{path}"
    if params:
        url = f"{url}?{parse.urlencode(params)}"
    req = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=timeout) as resp:
        return _json.loads(resp.read().decode("utf-8"))


def backend_health() -> dict[str, Any] | None:
    global _HEALTH_CACHE
    if _HEALTH_CACHE is not None:
        return _HEALTH_CACHE
    try:
        _HEALTH_CACHE = _get_json("/health", timeout=3.0)
    except (error.URLError, TimeoutError, ConnectionError):
        return None
    return _HEALTH_CACHE


def catalog_size() -> int:
    h = backend_health()
    return int(h["n_items"]) if h else 0


def recommend(
    user_id: str,
    seen_ids: set[str],
) -> dict[str, Any]:
    """Ask backend for the next recommendation.

    Returns an envelope with `track`, `model`, `contributions`, `rationale`,
    `expected_value`. `audio_url` inside the track is relative to BACKEND_URL.
    """
    exclude = ",".join(sorted(seen_ids))
    try:
        envelope = _get_json("/recommend", {"user_id": user_id, "exclude": exclude})
    except error.HTTPError as exc:
        raise RuntimeError(f"Backend ответил {exc.code}: {exc.read().decode('utf-8', 'ignore')}") from exc
    except (error.URLError, TimeoutError, ConnectionError) as exc:
        raise RuntimeError(
            f"Не дозвонился до backend на {BACKEND_URL}. "
            "Запусти `uvicorn backend.main:app` из корня репо."
        ) from exc

    # Map relative audio_url to absolute so st.audio can fetch it directly.
    track = envelope.get("track", {})
    if track.get("audio_url") and track["audio_url"].startswith("/"):
        track["audio_url"] = f"{BACKEND_URL}{track['audio_url']}"
    return envelope
