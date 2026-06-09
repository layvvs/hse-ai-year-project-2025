"""Synthetic user identity + on-disk history for the prototype frontend.

Each browser session gets a UUID stored in st.session_state. History (likes,
dislikes, skips, previously-shown tracks, current cursor) is persisted to a
per-user JSON file so we can track user behavior across reruns / restarts.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import streamlit as st

DATA_DIR = Path(__file__).parent / "data" / "user_history"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class UserHistory:
    user_id: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    queue: list[dict[str, Any]] = field(default_factory=list)
    cursor: int = -1

    @property
    def current(self) -> dict[str, Any] | None:
        if 0 <= self.cursor < len(self.queue):
            return self.queue[self.cursor]
        return None

    def record(self, action: str, track: dict[str, Any]) -> None:
        self.actions.append(
            {"action": action, "track_id": track.get("track_id"), "title": track.get("title")}
        )

    def liked_track_ids(self) -> set[str]:
        return {a["track_id"] for a in self.actions if a["action"] == "like"}

    def disliked_track_ids(self) -> set[str]:
        return {a["track_id"] for a in self.actions if a["action"] == "dislike"}


def _path_for(user_id: str) -> Path:
    return DATA_DIR / f"{user_id}.json"


def load_history(user_id: str) -> UserHistory:
    path = _path_for(user_id)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return UserHistory(
            user_id=raw["user_id"],
            actions=raw.get("actions", []),
            queue=raw.get("queue", []),
            cursor=raw.get("cursor", -1),
        )
    return UserHistory(user_id=user_id)


def save_history(history: UserHistory) -> None:
    _path_for(history.user_id).write_text(
        json.dumps(asdict(history), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_or_create_user() -> UserHistory:
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"u_{uuid.uuid4().hex[:12]}"
    if "history" not in st.session_state:
        st.session_state.history = load_history(st.session_state.user_id)
    return st.session_state.history


def reset_user() -> None:
    for key in ("user_id", "history"):
        st.session_state.pop(key, None)
