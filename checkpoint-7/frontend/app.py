"""Streamlit prototype for the music recommender.

Talks to the FastAPI backend (ELSA + LightGBM pipeline).
Run via Docker: make run (from demo/)
Or locally:  streamlit run frontend/app.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from recommender import BACKEND_URL, backend_health, recommend
from user_session import get_or_create_user, reset_user, save_history

st.set_page_config(page_title="Music Reco", layout="wide")

BG    = "#0b0b0c"
SURF  = "#141416"
INK   = "#f5f5f5"
MUTED = "#8b8b8f"
LINE  = "#232327"
POS   = "#a3e635"
NEG   = "#fb7185"

ACTION_RU = {"like": "нравится", "dislike": "не нравится", "skip": "пропуск"}
SPEED_RU = {
    "veryslow": "очень медленно", "slow": "медленно",
    "medium":   "средне",          "fast": "быстро",
    "veryfast": "очень быстро",
}
VOCAL_RU = {"instrumental": "инструментал", "vocal": "вокал"}


def _ru(mapping: dict[str, str], value: str) -> str:
    return mapping.get(value, value)


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {BG}; color: {INK}; }}
        .block-container {{ padding-top: 3rem; max-width: 980px; }}
        .stApp, .stApp p, .stApp span, .stApp div, .stApp label {{ color: {INK}; }}
        .reco-eyebrow {{
            font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase;
            color: {MUTED}; font-weight: 500; margin-bottom: 6px;
        }}
        .reco-title {{
            font-size: 28px; font-weight: 600; color: {INK};
            line-height: 1.2; margin: 0 0 4px 0; letter-spacing: -0.01em;
        }}
        .reco-meta {{ color: {MUTED}; font-size: 14px; margin-bottom: 14px; }}
        .reco-meta b {{ color: {INK}; font-weight: 500; }}
        .reco-tags {{ color: {MUTED}; font-size: 12px; margin-bottom: 18px; }}
        .reco-tags span {{ margin-right: 14px; }}
        .reco-id {{
            color: {MUTED}; font-size: 11px;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            margin-top: 16px;
        }}
        .reco-divider {{ height: 1px; background: {LINE}; margin: 28px 0; border: 0; }}
        .reco-model {{
            display: inline-flex; align-items: center; gap: 8px;
            color: {INK}; font-size: 13px; font-weight: 500;
        }}
        .reco-model .dot {{
            width: 8px; height: 8px; border-radius: 50%; display: inline-block;
            background: {INK};
        }}
        .reco-section-title {{
            font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase;
            color: {MUTED}; font-weight: 500; margin: 24px 0 10px 0;
        }}
        .reco-rationale {{ color: {INK}; font-size: 14px; line-height: 1.55; }}
        .stButton > button {{
            border-radius: 8px !important; font-weight: 500 !important;
            padding: 0.5rem 0.75rem !important;
            border: 1px solid {LINE} !important;
            background: {SURF} !important; color: {INK} !important;
            box-shadow: none !important;
            transition: border-color 120ms ease, background 120ms ease !important;
        }}
        .stButton > button:hover {{
            border-color: {INK} !important; background: #1d1d20 !important;
        }}
        .stButton > button:disabled {{
            color: #3f3f46 !important; border-color: {LINE} !important;
            background: {SURF} !important;
        }}
        section[data-testid="stSidebar"] {{
            background: #08080a; border-right: 1px solid {LINE};
        }}
        section[data-testid="stSidebar"] * {{ color: {INK}; }}
        section[data-testid="stSidebar"] .stCode, .stApp code, .stApp pre {{
            background: {SURF} !important; color: {INK} !important;
            border: 1px solid {LINE} !important;
        }}
        [data-testid="stDataFrame"] {{
            background: {SURF}; border-radius: 6px; border: 1px solid {LINE};
        }}
        [data-testid="stCaptionContainer"] {{ color: {MUTED} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_player(audio_url: str | None) -> None:
    if not audio_url:
        st.caption("Аудио-превью недоступно для этого item_id.")
        return
    html = f"""
    <!DOCTYPE html>
    <html><head><style>
        html, body {{ margin: 0; padding: 0; background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        .player {{ display: flex; align-items: center; gap: 14px; padding: 4px 0; }}
        .pp {{ width: 38px; height: 38px; border: 0; border-radius: 50%;
            background: #f5f5f5; cursor: pointer; padding: 0;
            display: flex; align-items: center; justify-content: center;
            flex-shrink: 0; transition: background 120ms ease, transform 120ms ease; }}
        .pp:hover {{ background: #ffffff; transform: scale(1.04); }}
        .pp:active {{ transform: scale(0.96); }}
        .ico-play {{ width: 0; height: 0; margin-left: 3px;
            border-left: 10px solid #0b0b0c;
            border-top: 6px solid transparent;
            border-bottom: 6px solid transparent; }}
        .ico-pause {{ width: 4px; height: 12px; box-shadow: 6px 0 0 0 #0b0b0c; background: #0b0b0c; }}
        .bar {{ flex: 1; height: 18px; cursor: pointer; position: relative;
            display: flex; align-items: center; }}
        .track {{ position: absolute; left: 0; right: 0; height: 3px;
            background: #2a2a2e; border-radius: 2px; }}
        .fill {{ position: absolute; left: 0; height: 3px;
            background: #f5f5f5; border-radius: 2px; width: 0%;
            transition: width 60ms linear; }}
        .thumb {{ position: absolute; width: 10px; height: 10px; border-radius: 50%;
            background: #f5f5f5; transform: translateX(-50%); left: 0%; opacity: 0;
            transition: opacity 120ms ease; }}
        .bar:hover .thumb {{ opacity: 1; }}
        .vol-group {{ display: flex; align-items: center; gap: 8px;
            flex-shrink: 0; cursor: pointer; }}
        .vol-ico {{ position: relative; width: 14px; height: 14px;
            opacity: 0.85; transition: opacity 120ms ease; }}
        .vol-group:hover .vol-ico {{ opacity: 1; }}
        .vol-ico::before {{ content: ''; position: absolute;
            left: 0; top: 4px; bottom: 4px; width: 4px;
            background: #8b8b8f; border-radius: 1px; }}
        .vol-ico::after {{ content: ''; position: absolute;
            left: 4px; top: 0; width: 0; height: 0;
            border-left: 7px solid #8b8b8f;
            border-top: 7px solid transparent;
            border-bottom: 7px solid transparent; }}
        .vol-group.muted .vol-ico::before {{ background: #3f3f46; }}
        .vol-group.muted .vol-ico::after {{ border-left-color: #3f3f46; }}
        .vol-bar {{ width: 64px; height: 18px; position: relative;
            display: flex; align-items: center; }}
        .vol-track {{ position: absolute; left: 0; right: 0; height: 3px;
            background: #2a2a2e; border-radius: 2px; }}
        .vol-fill {{ position: absolute; left: 0; height: 3px;
            background: #f5f5f5; border-radius: 2px; width: 80%;
            transition: width 60ms linear; }}
        .time {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 12px; color: #8b8b8f; min-width: 86px; text-align: right;
            font-variant-numeric: tabular-nums; }}
        .sep {{ margin: 0 4px; color: #3f3f46; }}
    </style></head>
    <body>
        <div class="player">
            <button class="pp" id="pp"><span class="ico-play" id="ico"></span></button>
            <div class="bar" id="bar">
                <div class="track"></div>
                <div class="fill" id="fill"></div>
                <div class="thumb" id="thumb"></div>
            </div>
            <div class="vol-group" id="volGroup" title="Громкость">
                <span class="vol-ico"></span>
                <div class="vol-bar" id="volBar">
                    <div class="vol-track"></div>
                    <div class="vol-fill" id="volFill"></div>
                </div>
            </div>
            <div class="time"><span id="cur">0:00</span><span class="sep">/</span><span id="dur">0:00</span></div>
            <audio id="aud" src="{audio_url}" preload="metadata"></audio>
        </div>
        <script>
            const aud = document.getElementById('aud');
            const pp = document.getElementById('pp'), ico = document.getElementById('ico');
            const fill = document.getElementById('fill'), thumb = document.getElementById('thumb');
            const bar = document.getElementById('bar');
            const cur = document.getElementById('cur'), dur = document.getElementById('dur');
            const volGroup = document.getElementById('volGroup');
            const volBar = document.getElementById('volBar'), volFill = document.getElementById('volFill');
            const fmt = t => {{
                if (!isFinite(t)) return '0:00';
                const m = Math.floor(t / 60), s = Math.floor(t % 60);
                return m + ':' + (s < 10 ? '0' : '') + s;
            }};
            const VOL_KEY = 'mrPlayerVol';
            const setVolume = v => {{
                v = Math.max(0, Math.min(1, v));
                aud.volume = v;
                volFill.style.width = (v * 100) + '%';
                volGroup.classList.toggle('muted', v === 0);
            }};
            const stored = parseFloat(localStorage.getItem(VOL_KEY));
            setVolume(isFinite(stored) ? stored : 0.8);

            pp.addEventListener('click', () => {{ if (aud.paused) aud.play(); else aud.pause(); }});
            aud.addEventListener('play',  () => {{ ico.className = 'ico-pause'; }});
            aud.addEventListener('pause', () => {{ ico.className = 'ico-play'; }});
            aud.addEventListener('ended', () => {{ ico.className = 'ico-play'; }});
            aud.addEventListener('loadedmetadata', () => {{ dur.textContent = fmt(aud.duration); }});
            aud.addEventListener('timeupdate', () => {{
                cur.textContent = fmt(aud.currentTime);
                if (aud.duration) {{
                    const pct = 100 * aud.currentTime / aud.duration;
                    fill.style.width = pct + '%';
                    thumb.style.left = pct + '%';
                }}
            }});
            bar.addEventListener('click', e => {{
                const r = bar.getBoundingClientRect();
                const pct = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
                if (aud.duration) aud.currentTime = pct * aud.duration;
            }});

            const seekVol = e => {{
                const r = volBar.getBoundingClientRect();
                const v = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
                setVolume(v);
                localStorage.setItem(VOL_KEY, String(v));
            }};
            volBar.addEventListener('click', seekVol);
            volBar.addEventListener('mousedown', () => {{
                const move = e => seekVol(e);
                const up = () => {{
                    window.removeEventListener('mousemove', move);
                    window.removeEventListener('mouseup', up);
                }};
                window.addEventListener('mousemove', move);
                window.addEventListener('mouseup', up);
            }});
        </script>
    </body></html>
    """
    components.html(html, height=56)


def _next_recommendation(history) -> None:
    seen = {item["track"]["track_id"] for item in history.queue}
    envelope = recommend(user_id=history.user_id, seen_ids=seen)
    history.queue.append(envelope)
    history.cursor = len(history.queue) - 1


def _ensure_current(history) -> None:
    if history.current is None:
        _next_recommendation(history)


def _act(history, action: str) -> None:
    current = history.current
    if current is None:
        return
    history.record(action, current["track"])
    save_history(history)
    if action in ("like", "dislike", "skip"):
        _next_recommendation(history)
        save_history(history)


def _prev(history) -> None:
    if history.cursor > 0:
        history.cursor -= 1
        save_history(history)


# ────────────────────────────── render ──────────────────────────────

_inject_css()

health = backend_health()
if health is None:
    st.error(
        f"Backend недоступен на {BACKEND_URL}.\n\n"
        "Запусти из директории demo:\n```\nmake run\n```\n"
        "или локально:\n```\nuvicorn backend.main:app --host 127.0.0.1 --port 8000\n```"
    )
    st.stop()

history = get_or_create_user()

try:
    _ensure_current(history)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.markdown('<div class="reco-eyebrow">Пользователь</div>', unsafe_allow_html=True)
    st.code(history.user_id, language="text")
    current_uid = history.current["track"].get("yambda_uid") if history.current else None
    if current_uid is not None:
        st.caption(f"Yambda uid: `{current_uid}` (детерминированный маппинг)")
    if st.button("Новый пользователь", use_container_width=True):
        reset_user()
        st.rerun()

    st.markdown('<hr class="reco-divider">', unsafe_allow_html=True)
    st.markdown('<div class="reco-eyebrow">История</div>', unsafe_allow_html=True)
    if history.actions:
        df = (
            pd.DataFrame(history.actions[::-1])
            .assign(action=lambda d: d["action"].map(lambda a: ACTION_RU.get(a, a)))
            .rename(columns={"action": "действие", "title": "название"})
        )
        st.dataframe(df, hide_index=True, use_container_width=True, height=360)
    else:
        st.caption("Пока пусто.")

    st.markdown('<hr class="reco-divider">', unsafe_allow_html=True)
    st.markdown('<div class="reco-eyebrow">Backend</div>', unsafe_allow_html=True)
    st.caption(
        f"users: {health['known_users']:,}  ·  items: {health['n_items']:,}  ·  "
        f"features: {health['feature_count']}"
    )

current = history.current
track = current["track"]
model = current["model"]
contribs: list[dict[str, float]] = current["contributions"]
rationale = current["rationale"]

st.markdown('<div class="reco-eyebrow">Сейчас играет</div>', unsafe_allow_html=True)
st.markdown(f'<div class="reco-title">{track["title"]}</div>', unsafe_allow_html=True)
album_suffix = f' · {track["album"]}' if track["album"] else ""
st.markdown(
    f'<div class="reco-meta"><b>{track["artist"]}</b>{album_suffix}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="reco-tags">
        <span>{track["primary_genre"]}</span>
        <span>{track["year"]}</span>
        <span>{track["duration_str"]}</span>
        <span>{_ru(SPEED_RU, track["speed"])}</span>
        <span>{_ru(VOCAL_RU, track["vocal"])}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

_render_player(track.get("audio_url"))

st.write("")
b_prev, b_dislike, b_skip, b_like = st.columns(4)
with b_prev:
    st.button("Назад", use_container_width=True,
              disabled=history.cursor == 0, on_click=_prev, args=(history,))
with b_dislike:
    st.button("Не нравится", use_container_width=True,
              on_click=_act, args=(history, "dislike"))
with b_skip:
    st.button("Пропустить", use_container_width=True,
              on_click=_act, args=(history, "skip"))
with b_like:
    st.button("Нравится", use_container_width=True,
              on_click=_act, args=(history, "like"))

st.caption(
    f"{history.cursor + 1} / {len(history.queue)}  ·  "
    f"item_id {track['track_id']}  ·  score {track['score']:.3f}  ·  rank {track['rank']}"
)

st.markdown('<hr class="reco-divider">', unsafe_allow_html=True)

st.markdown(
    f'<div class="reco-model"><span class="dot"></span>'
    f'<span>Рекомендовала модель</span><b style="font-weight:600;">{model}</b></div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="reco-section-title">Обоснование</div>', unsafe_allow_html=True)
st.markdown(f'<div class="reco-rationale">{rationale}</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="reco-section-title">SHAP contributions (top 10)</div>',
    unsafe_allow_html=True,
)
contrib_df = (
    pd.DataFrame(contribs)
    .head(10)
    .assign(sign=lambda d: d["contribution"].apply(lambda v: "positive" if v >= 0 else "negative"))
)

chart = (
    alt.Chart(contrib_df)
    .mark_bar(size=14)
    .encode(
        x=alt.X("contribution:Q", title=None,
                axis=alt.Axis(grid=False, domain=False,
                              tickColor=LINE, labelColor=MUTED, format="+.2f")),
        y=alt.Y("feature:N", sort=alt.SortField("contribution", order="descending"),
                title=None,
                axis=alt.Axis(domain=False, ticks=False, labelColor=INK, labelFontSize=12)),
        color=alt.Color(
            "sign:N",
            scale=alt.Scale(domain=["positive", "negative"], range=[POS, NEG]),
            legend=None,
        ),
        tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("contribution:Q", format="+.4f")],
    )
    .properties(height=max(180, 28 * len(contrib_df)), background="transparent")
    .configure_view(strokeWidth=0)
)
st.altair_chart(chart, use_container_width=True)
st.caption(
    f"LightGBM pred_contrib для item_id {track['track_id']}. "
    f"Expected value = {current['expected_value']:+.3f}. "
    "Зелёное — поднимает скор, красное — опускает."
)
