"""FastAPI backend wrapping Bogdan's ELSA + LightGBM pipeline.

Exposes:
    GET /health              — load status + sizes
    GET /recommend           — next recommendation envelope for a synthetic user_id
    GET /audio/{stem}.mp3    — streams a Jamendo mp3 from music/

Notes:
    * Bogdan's models work on Yambda integer item_ids. Our `music/` is Jamendo.
      We map every synthetic Streamlit user_id → a deterministic Yambda uid
      taken from the trained mappings (so the same browser session always sees
      a consistent profile). For audio, we round-robin Yambda item_id over
      Jamendo mp3 stems we have on disk — the recommended item_id is what the
      model produced; the preview audio is whatever we can actually play.
"""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoint-4"
MUSIC_DIR = ROOT / "music"
META_DIR = ROOT / "metadata"

# Make Bogdan's `from app.recommendations...` imports resolvable.
sys.path.insert(0, str(CHECKPOINT_DIR))

from app.recommendations.engine import (  # noqa: E402
    CANDIDATE_K,
    DEFAULT_K,
    RecommendationEngine,
    load_engine,
)


# ────────────────────────────── lifespan ──────────────────────────────


STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine = load_engine()
    known_uids: list[int] = sorted(engine.user2id.keys())
    mp3_stems: list[str] = sorted(p.stem for p in MUSIC_DIR.glob("*.mp3")) if MUSIC_DIR.exists() else []

    STATE["engine"] = engine
    STATE["known_uids"] = known_uids
    STATE["mp3_stems"] = mp3_stems
    STATE["feature_cols"] = engine.feature_cols

    yield
    STATE.clear()


app = FastAPI(
    title="Music Reco Backend",
    description="ELSA candidates + LightGBM lambdarank reranker (Bogdan, checkpoint-7)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ────────────────────────────── helpers ──────────────────────────────


def _hash_to_uid(user_id: str) -> int:
    """Deterministically map synthetic user_id → a known Yambda uid."""
    known = STATE["known_uids"]
    h = int(hashlib.sha1(user_id.encode("utf-8")).hexdigest(), 16)
    return int(known[h % len(known)])


@lru_cache(maxsize=4096)
def _jamendo_meta(stem: str) -> dict[str, Any] | None:
    path = META_DIR / f"{stem}.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    info = raw.get("musicinfo") or {}
    tags = info.get("tags") or {}
    genres = tags.get("genres") or []
    duration = int(raw.get("duration") or 0)
    m, s = divmod(duration, 60)
    return {
        "title":         raw.get("name") or "Untitled",
        "artist":        raw.get("artist_name") or "Unknown",
        "album":         raw.get("album_name") or "",
        "year":          (raw.get("releasedate") or "")[:4] or "—",
        "duration_str":  f"{m}:{s:02d}" if duration else "—",
        "primary_genre": (genres[0] if genres else "—"),
        "speed":         info.get("speed") or "—",
        "vocal":         info.get("vocalinstrumental") or "—",
    }


def _preview_for(item_id: int) -> str | None:
    stems = STATE["mp3_stems"]
    if not stems:
        return None
    return stems[item_id % len(stems)]


def _shap_contributions(
    engine: RecommendationEngine,
    uid: int,
    item_id: int,
) -> tuple[list[dict[str, float]], float]:
    """Per-item LightGBM SHAP contributions for explanation."""
    feature_store = engine.feature_store
    candidates = feature_store.build_candidates_frame(uid, [item_id])
    X = candidates[engine.feature_cols]
    contribs = engine.reranker_model.predict(X, pred_contrib=True)
    contribs = np.asarray(contribs).reshape(-1)
    expected = float(contribs[-1])
    per_feature = contribs[:-1]
    pairs = sorted(
        zip(engine.feature_cols, per_feature),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    out = [{"feature": f, "contribution": float(v)} for f, v in pairs]
    return out, expected


def _build_rationale(top_contribs: list[dict[str, float]], yambda_uid: int) -> str:
    top = top_contribs[:3]
    parts = ", ".join(f"{c['feature']} ({c['contribution']:+.2f})" for c in top)
    return (
        f"ELSA отобрал кандидата для uid={yambda_uid}, LightGBM ранкер "
        f"поднял его наверх. Топ-3 вклада: {parts}."
    )


# ────────────────────────────── routes ──────────────────────────────


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if STATE.get("engine") else "loading",
        "known_users": len(STATE.get("known_uids", [])),
        "n_items":     len(STATE.get("engine").id2item) if STATE.get("engine") else 0,
        "preview_pool": len(STATE.get("mp3_stems", [])),
        "feature_count": len(STATE.get("feature_cols", [])),
        "candidate_k": CANDIDATE_K,
        "default_k":   DEFAULT_K,
    }


@app.get("/recommend")
def recommend(
    user_id: str = Query(..., description="Synthetic user id from the frontend"),
    exclude: str = Query("", description="Comma-separated item_ids already seen"),
    k: int = Query(50, ge=1, le=CANDIDATE_K),
) -> dict[str, Any]:
    engine: RecommendationEngine = STATE["engine"]
    yambda_uid = _hash_to_uid(user_id)
    excluded = {int(x) for x in exclude.split(",") if x.strip()}

    try:
        recs = engine.recommend(yambda_uid, k=k)
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    pick = next((r for r in recs if int(r["item_id"]) not in excluded), None)
    if pick is None:
        raise HTTPException(
            status_code=404,
            detail="Все кандидаты в exclude-листе; попробуй сбросить историю.",
        )

    item_id = int(pick["item_id"])
    score = float(pick["score"])
    rank = int(pick["rank"])

    top_contribs, expected = _shap_contributions(engine, yambda_uid, item_id)
    preview_stem = _preview_for(item_id)
    meta = _jamendo_meta(preview_stem) if preview_stem else None
    if meta is None:
        meta = {
            "title": f"Item {item_id}", "artist": "Unknown", "album": "",
            "year": "—", "duration_str": "—",
            "primary_genre": "—", "speed": "—", "vocal": "—",
        }

    return {
        "track": {
            "track_id":      str(item_id),
            "yambda_uid":    yambda_uid,
            "score":         score,
            "rank":          rank,
            "preview_stem":  preview_stem,
            "audio_url":     f"/audio/{preview_stem}.mp3" if preview_stem else None,
            **meta,
        },
        "model": "ELSA → LightGBM lambdarank",
        "expected_value": expected,
        "contributions": top_contribs,
        "rationale": _build_rationale(top_contribs, yambda_uid),
    }


@app.get("/audio/{stem}.mp3")
def audio(stem: str) -> FileResponse:
    path = MUSIC_DIR / f"{stem}.mp3"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"mp3 not found: {stem}")
    return FileResponse(path, media_type="audio/mpeg")
