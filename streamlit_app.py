"""Production-style Streamlit dashboard for the AI Insight Engine."""

from __future__ import annotations

from datetime import datetime
from html import escape
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from analysis import (
    analyze_sentiment,
    build_analysis,
    get_model,
    llm_sentiment,
    llm_topic_summary,
    semantic_search,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = APP_DIR / "ai_insight_engine_youth_comments.csv"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
NAV_ITEMS = {
    "Overview": "⌂",
    "Topic Lab": "◈",
    "Semantic Search": "⌕",
    "Sentiment": "◒",
    "LLM Studio": "✦",
    "Exports": "⇩",
    "Quality Lab": "⌁",
}
ACCENTS = ["#68e8ff", "#9d8cff", "#ffb76b", "#5ce1a8", "#ff7e9f"]

st.set_page_config(
    page_title="AI Insight Engine · Signal Desk",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
      --ink: #050914;
      --ink-2: #080e1d;
      --panel: rgba(13, 22, 43, .88);
      --panel-2: rgba(16, 28, 54, .74);
      --line: rgba(151, 179, 226, .16);
      --muted: #8291ad;
      --text: #f0f5ff;
      --cyan: #68e8ff;
      --violet: #9d8cff;
      --amber: #ffb76b;
      --green: #5ce1a8;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', 'Pretendard', sans-serif; }
    .stApp {
      color: var(--text);
      background:
        radial-gradient(circle at 85% 8%, rgba(64, 111, 255, .18), transparent 24rem),
        radial-gradient(circle at 95% 82%, rgba(255, 107, 50, .12), transparent 22rem),
        linear-gradient(135deg, #050914 0%, #071021 50%, #0a1020 100%);
    }
    [data-testid="stHeader"] { background: rgba(5, 9, 20, .55); }
    [data-testid="stToolbar"] { right: .65rem; }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, rgba(5, 11, 25, .98), rgba(7, 13, 27, .94));
      border-right: 1px solid rgba(132, 164, 214, .14);
    }
    [data-testid="stSidebar"] > div:first-child { padding: 1.1rem .85rem 1rem; }
    [data-testid="stSidebar"] .block-container { padding-top: 0; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: .28rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
      border: 1px solid transparent; border-radius: 13px; padding: .65rem .72rem;
      transition: all .2s ease; color: #98a8c7;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
      background: rgba(104, 232, 255, .08); color: #eaf7ff; border-color: rgba(104, 232, 255, .12);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
      background: linear-gradient(90deg, rgba(104, 232, 255, .16), rgba(157, 140, 255, .08));
      color: #f6fbff; border-color: rgba(104, 232, 255, .23); box-shadow: inset 3px 0 0 var(--cyan);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label p { font-size: .86rem; font-weight: 600; margin: 0; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
      background: rgba(13, 22, 43, .72); border: 1px dashed rgba(104, 232, 255, .25); border-radius: 14px;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] small { color: var(--muted); }
    .block-container { max-width: 1460px; padding: 2rem 2.6rem 4rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', 'DM Sans', sans-serif; letter-spacing: -.035em; }
    h1 { font-size: clamp(2rem, 3.2vw, 3.1rem); }
    h2 { font-size: 1.42rem; }
    h3 { font-size: 1.05rem; }
    .eyebrow { color: var(--cyan); font-size: .69rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .muted { color: var(--muted); }
    .brand-lockup { display: flex; align-items: center; gap: .68rem; padding: .25rem .35rem 1.25rem; }
    .brand-mark { width: 2.2rem; height: 2.2rem; border-radius: 11px; display: grid; place-items: center;
      color: #071020; font-size: 1.3rem; font-weight: 800; background: linear-gradient(135deg, var(--cyan), var(--violet));
      box-shadow: 0 0 28px rgba(104, 232, 255, .25); }
    .brand-name { color: #f7fbff; font-family: 'Space Grotesk', sans-serif; font-size: 1.02rem; font-weight: 700; }
    .brand-sub { color: #7789a9; font-size: .68rem; margin-top: .12rem; }
    .topline { display: flex; align-items: flex-start; justify-content: space-between; gap: 1.5rem; margin-bottom: 1.7rem; }
    .topline h1 { margin: .25rem 0 .28rem; }
    .topline p { color: var(--muted); margin: 0; max-width: 720px; }
    .top-meta { display: flex; align-items: center; gap: .55rem; padding-top: .45rem; white-space: nowrap; }
    .live-pill, .soft-pill { border: 1px solid rgba(104, 232, 255, .2); background: rgba(104, 232, 255, .08);
      border-radius: 999px; padding: .44rem .72rem; color: #bcefff; font-size: .74rem; font-weight: 600; }
    .live-dot { display: inline-block; width: .42rem; height: .42rem; background: var(--green); border-radius: 50%; margin-right: .35rem; box-shadow: 0 0 10px var(--green); }
    .hero-card, .dashboard-card { border: 1px solid var(--line); border-radius: 19px; background: linear-gradient(140deg, rgba(16, 29, 57, .92), rgba(9, 17, 35, .82)); box-shadow: 0 16px 50px rgba(0,0,0,.17); }
    .hero-card { padding: 1.4rem 1.5rem; min-height: 182px; position: relative; overflow: hidden; }
    .hero-card:after { content: ''; position: absolute; width: 18rem; height: 18rem; border-radius: 50%; right: -7rem; top: -8rem; background: radial-gradient(circle, rgba(104,232,255,.18), transparent 66%); pointer-events: none; }
    .hero-card h2 { margin: .45rem 0 .45rem; font-size: 1.65rem; }
    .hero-card .hero-copy { max-width: 600px; color: #a5b5d2; line-height: 1.55; }
    .hero-meta { display: flex; gap: .55rem; flex-wrap: wrap; margin-top: 1.05rem; }
    .hero-meta span { border: 1px solid rgba(151,179,226,.15); border-radius: 8px; padding: .36rem .55rem; color: #b6c5df; font-size: .73rem; background: rgba(6,12,27,.32); }
    .heartbeat { padding: 1.25rem 1.35rem; min-height: 182px; }
    .heartbeat .date { color: #f4f7ff; font-family: 'Space Grotesk', sans-serif; font-size: 1.24rem; font-weight: 600; margin: .52rem 0 .1rem; }
    .heartbeat .big-number { color: var(--cyan); font-family: 'Space Grotesk', sans-serif; font-size: 2.3rem; font-weight: 600; margin: 1.1rem 0 .1rem; }
    .heartbeat .small { color: var(--muted); font-size: .76rem; }
    .metric-card { min-height: 118px; padding: 1rem 1.05rem; border: 1px solid var(--line); border-radius: 16px; background: rgba(10, 19, 39, .74); }
    .metric-label { color: #8b9bb8; font-size: .75rem; }
    .metric-value { color: #f7fbff; font-family: 'Space Grotesk', sans-serif; font-size: 1.85rem; font-weight: 600; margin-top: .45rem; }
    .metric-note { color: #65e3b0; font-size: .7rem; margin-top: .18rem; }
    .section-head { display: flex; align-items: baseline; justify-content: space-between; margin: 1.65rem 0 .72rem; }
    .section-head h2, .section-head h3 { margin: 0; }
    .section-head span { color: var(--muted); font-size: .73rem; }
    .topic-list { display: grid; gap: .6rem; }
    .topic-row { display: flex; align-items: center; gap: .82rem; padding: .78rem .86rem; border: 1px solid rgba(151,179,226,.12); border-radius: 13px; background: rgba(11,20,41,.7); }
    .topic-index { width: 2rem; height: 2rem; flex: 0 0 auto; display: grid; place-items: center; border-radius: 9px; color: #061020; font-weight: 800; font-size: .75rem; }
    .topic-body { min-width: 0; flex: 1; }
    .topic-title { color: #eaf3ff; font-size: .83rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .topic-sub { color: #7d8eac; font-size: .7rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: .16rem; }
    .topic-count { color: #dbe8ff; font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 600; }
    .status-card { padding: 1.1rem 1.15rem; border: 1px solid var(--line); border-radius: 17px; background: rgba(10, 19, 39, .7); }
    .status-line { display: flex; justify-content: space-between; align-items: center; padding: .5rem 0; border-bottom: 1px solid rgba(151,179,226,.1); font-size: .76rem; }
    .status-line:last-child { border-bottom: 0; padding-bottom: 0; }
    .status-ok { color: var(--green); }
    .status-value { color: #a7b7d2; }
    .queue-list { display: grid; gap: .55rem; }
    .queue-item { display: flex; align-items: center; gap: .72rem; padding: .74rem .8rem; border: 1px solid rgba(151,179,226,.11); border-radius: 12px; background: rgba(11,20,41,.66); }
    .queue-dot { width: .48rem; height: .48rem; border-radius: 50%; flex: 0 0 auto; }
    .queue-text { color: #dce7fb; font-size: .78rem; line-height: 1.35; }
    .queue-text small { display: block; color: #7889a8; font-size: .68rem; margin-top: .12rem; }
    .dashboard-card { padding: 1.1rem 1.2rem; }
    .stPlotlyChart, [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid rgba(151,179,226,.11); border-radius: 15px; overflow: hidden; background: rgba(9, 17, 35, .56); }
    [data-testid="stDataFrame"] { padding: .25rem; }
    [data-testid="stMetric"] { border: 0; }
    .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
      border: 1px solid rgba(104,232,255,.23); border-radius: 10px; color: #dff8ff; background: rgba(104,232,255,.1); font-weight: 600; transition: all .2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { border-color: var(--cyan); background: rgba(104,232,255,.18); color: #fff; transform: translateY(-1px); }
    input, textarea, [data-baseweb="select"] > div { background-color: rgba(9, 17, 35, .84) !important; border-color: rgba(151,179,226,.18) !important; }
    [data-testid="stExpander"] { border-color: rgba(151,179,226,.14); border-radius: 13px; background: rgba(10,19,39,.45); }
    @media (max-width: 900px) {
      .block-container { padding: 1.25rem 1rem 3rem; }
      .topline { flex-direction: column; gap: .5rem; }
      .top-meta { padding-top: 0; }
    }
    @media (prefers-reduced-motion: reduce) { *, *:before, *:after { transition: none !important; animation: none !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def cached_analysis(csv_bytes: bytes, n_clusters: int) -> dict[str, Any]:
    """Cache the expensive embedding and clustering result for this CSV/k pair."""

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
            handle.write(csv_bytes)
            temporary_path = handle.name
        return build_analysis(temporary_path, int(n_clusters))
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


cached_analysis = st.cache_data(show_spinner=False)(cached_analysis)


def sentiment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["cluster", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["positive", "neutral", "negative"], fill_value=0)
        .reset_index()
    )
    summary["total"] = summary[["positive", "neutral", "negative"]].sum(axis=1)
    for column in ("positive", "neutral", "negative"):
        summary[f"{column}_ratio"] = summary[column] / summary["total"].replace(0, 1)
    return summary


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def secret_value(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
        return str(value) if value else ""
    except Exception:
        return ""


def run_chunked_llm_sentiment(
    frame: pd.DataFrame,
    api_key: str,
    base_url: str,
    model: str,
    chunk_size: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify in chunks so a large CSV does not exceed one model context window."""

    chunks: list[pd.DataFrame] = []
    starts = range(0, len(frame), chunk_size)
    progress = st.progress(0, text="LLM 감성 분석 준비 중")
    total_chunks = max(1, (len(frame) + chunk_size - 1) // chunk_size)

    for chunk_number, start in enumerate(starts, start=1):
        chunk = frame.iloc[start : start + chunk_size].copy()
        annotated_chunk, _ = llm_sentiment(chunk, api_key, base_url, model)
        chunks.append(annotated_chunk[["id", "sentiment"]])
        progress.progress(
            min(chunk_number / total_chunks, 1.0),
            text=f"LLM 감성 분석 중: {chunk_number}/{total_chunks}",
        )

    progress.empty()
    labels = pd.concat(chunks, ignore_index=True)
    annotated = frame.drop(columns=["sentiment"], errors="ignore").merge(labels, on="id", how="left")
    annotated["sentiment"] = annotated["sentiment"].fillna("neutral")
    return annotated, sentiment_summary(annotated)


def reset_derived_state() -> None:
    for key in ("sentiment_frame", "sentiment_summary", "search_frame", "llm_summary"):
        st.session_state.pop(key, None)


def render_sidebar() -> tuple[bytes, str, int, bool, str, str]:
    with st.sidebar:
        st.markdown(
            '<div class="brand-lockup"><div class="brand-mark">◈</div><div><div class="brand-name">Signal Desk</div><div class="brand-sub">AI Insight Engine</div></div></div>',
            unsafe_allow_html=True,
        )
        nav = st.radio(
            "Workspace",
            list(NAV_ITEMS),
            format_func=lambda item: f"{NAV_ITEMS[item]}   {item}",
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown('<div class="eyebrow">DATA SOURCE</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "CSV file",
            type=["csv"],
            label_visibility="collapsed",
            help="text 컬럼이 포함된 CSV를 선택하세요.",
        )

        if uploaded is None:
            if not DEFAULT_CSV.exists():
                st.error("기본 CSV 파일을 찾을 수 없습니다.")
                st.stop()
            source_bytes = DEFAULT_CSV.read_bytes()
            source_name = DEFAULT_CSV.name
            st.caption(f"Default dataset · {source_name}")
        else:
            source_bytes = uploaded.getvalue()
            source_name = uploaded.name
            st.caption(f"Uploaded · {source_name}")

        st.markdown('<div class="eyebrow" style="margin-top:1rem;">MODEL CONTROL</div>', unsafe_allow_html=True)
        n_clusters = st.slider("Topics (k)", 2, 15, 7, 1, label_visibility="collapsed")
        analyze_clicked = st.button("Run analysis", type="primary", use_container_width=True)
        st.divider()
        st.markdown('<div class="status-line"><span>Workspace</span><span class="status-ok">● Live</span></div>', unsafe_allow_html=True)
        st.caption("Python 3.14 · CPU inference")

    signature = hashlib.sha256(source_bytes).hexdigest()
    return source_bytes, source_name, n_clusters, analyze_clicked, signature, nav


def ensure_analysis(
    source_bytes: bytes,
    source_name: str,
    n_clusters: int,
    analyze_clicked: bool,
    source_signature: str,
) -> dict[str, Any]:
    previous_signature = st.session_state.get("source_signature")
    previous_k = st.session_state.get("analysis_k")
    should_run = analyze_clicked or "analysis" not in st.session_state or previous_signature != source_signature

    if should_run:
        with st.spinner("CSV 정제, 문장 임베딩, 토픽 지도를 계산하는 중입니다..."):
            try:
                result = cached_analysis(source_bytes, int(n_clusters))
            except Exception as error:
                st.error(f"분석에 실패했습니다: {error}")
                st.info("CSV에 text 컬럼이 있는지, 파일 인코딩이 UTF-8 또는 CP949인지 확인해주세요.")
                st.stop()
        st.session_state["analysis"] = result
        st.session_state["source_signature"] = source_signature
        st.session_state["source_name"] = source_name
        st.session_state["analysis_k"] = int(n_clusters)
        st.session_state["last_run"] = datetime.now().strftime("%b %d, %Y · %H:%M")
        reset_derived_state()
    elif previous_k != int(n_clusters):
        st.sidebar.warning("k가 변경되었습니다. Run analysis를 눌러 새 결과를 계산하세요.")
    return st.session_state["analysis"]


def style_figure(figure: go.Figure, height: int | None = None) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, sans-serif", "color": "#b9c8e2"},
        margin={"l": 8, "r": 8, "t": 35, "b": 8},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 11}},
        hoverlabel={"bgcolor": "#101c37", "font": {"color": "#f0f5ff"}},
    )
    if height:
        figure.update_layout(height=height)
    figure.update_xaxes(showgrid=False, zeroline=False, color="#8291ad")
    figure.update_yaxes(gridcolor="rgba(151,179,226,.10)", zeroline=False, color="#8291ad")
    return figure


def topic_volume_figure(frame: pd.DataFrame) -> go.Figure:
    counts = frame.groupby("cluster").size().sort_index()
    colors = [ACCENTS[index % len(ACCENTS)] for index in range(len(counts))]
    figure = go.Figure(
        go.Bar(
            x=[f"T{int(value):02d}" for value in counts.index],
            y=counts.values,
            marker={"color": colors, "line": {"width": 0}},
            hovertemplate="%{x}<br>%{y:,} comments<extra></extra>",
        )
    )
    figure.update_layout(title="Topic volume", xaxis_title=None, yaxis_title=None, showlegend=False)
    return style_figure(figure, 250)


def signal_mix_figure(result: dict[str, Any]) -> go.Figure:
    sentiment_frame = st.session_state.get("sentiment_frame")
    if sentiment_frame is not None:
        values = sentiment_frame["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
        labels = ["Positive", "Neutral", "Negative"]
        colors = ["#5ce1a8", "#68e8ff", "#ff7e9f"]
    else:
        values = result["df"]["cluster"].value_counts().sort_index()
        labels = [f"Topic {int(value)}" for value in values.index]
        colors = [ACCENTS[index % len(ACCENTS)] for index in range(len(values))]
    figure = go.Figure(go.Pie(labels=labels, values=values.values, hole=.72, marker={"colors": colors, "line": {"color": "#0b1428", "width": 4}}, textinfo="none", hovertemplate="%{label}<br>%{value:,}<extra></extra>"))
    figure.update_layout(showlegend=False, annotations=[{"text": f"{int(values.sum()):,}<br><span style='font-size:11px'>signals</span>", "showarrow": False, "font": {"size": 22, "color": "#f0f5ff"}}])
    return style_figure(figure, 220)


def render_topline(result: dict[str, Any]) -> None:
    silhouette = result["silhouette_scores"]
    best_score = float(silhouette["silhouette_score"].max()) if not silhouette.empty else 0.0
    st.markdown(
        f"""
        <div class="topline">
          <div>
            <div class="eyebrow">INSIGHT OPERATIONS / LIVE WORKSPACE</div>
            <h1>Signal desk</h1>
            <p>청년 의견의 흐름을 읽고, 반복되는 신호를 실행 가능한 토픽으로 바꿉니다.</p>
          </div>
          <div class="top-meta"><span class="live-pill"><span class="live-dot"></span>Analysis ready</span><span class="soft-pill">k={int(st.session_state.get('analysis_k', 7))} · {best_score:.3f}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(result: dict[str, Any], selected_cluster: int | None) -> None:
    frame = result["df"]
    topic_summary = result["topic_summary"].copy()
    source_name = escape(str(st.session_state.get("source_name", DEFAULT_CSV.name)))
    recommended_k = int(result["recommended_k"])
    best_score = float(result["silhouette_scores"]["silhouette_score"].max()) if not result["silhouette_scores"].empty else 0.0
    top_keywords = ", ".join(topic_summary["keywords"].dropna().astype(str).head(2).tolist()) or "No keywords yet"

    hero_col, heartbeat_col = st.columns([1.58, .85], gap="medium")
    with hero_col:
        st.markdown(
            f"""
            <div class="hero-card">
              <div class="eyebrow">GOOD EVENING, ANALYST</div>
              <h2>Find the signal<br>inside the noise.</h2>
              <div class="hero-copy">현재 <b>{len(frame):,}개</b>의 의견이 분석되었습니다. 가장 눈에 띄는 언어 신호는 <b>{escape(top_keywords)}</b>입니다.</div>
              <div class="hero-meta"><span>Source · {source_name}</span><span>Embedding · multilingual MiniLM</span><span>Last run · {escape(str(st.session_state.get('last_run', 'now')))}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with heartbeat_col:
        st.markdown(
            f"""
            <div class="dashboard-card heartbeat">
              <div class="eyebrow">DATASET HEARTBEAT</div>
              <div class="date">{datetime.now().strftime('%b %d, %Y')}</div>
              <div class="small">Signal collection is online</div>
              <div class="big-number">{len(frame):,}</div>
              <div class="small">unique comments · UTF-8 normalized</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    metric_cols = st.columns(4, gap="small")
    metrics = [
        ("Comments processed", f"{len(frame):,}", "cleaned + deduplicated"),
        ("Topics discovered", f"{frame['cluster'].nunique():,}", f"current k = {st.session_state.get('analysis_k', 7)}"),
        ("Recommended k", str(recommended_k), "silhouette selected"),
        ("Best separation", f"{best_score:.4f}", "cosine silhouette"),
    ]
    for column, (label, value, note) in zip(metric_cols, metrics):
        with column:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">● {note}</div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.42, .9], gap="medium")
    with left:
        st.markdown('<div class="section-head"><h2>Topic pulse</h2><span>Volume by discovered cluster</span></div>', unsafe_allow_html=True)
        st.plotly_chart(topic_volume_figure(frame), use_container_width=True, config={"displayModeBar": False}, theme=None)
        st.markdown('<div class="section-head"><h3>Priority signals</h3><span>Top topics by volume</span></div>', unsafe_allow_html=True)
        display_summary = topic_summary if selected_cluster is None else topic_summary[topic_summary["cluster"] == selected_cluster]
        rows = []
        for index, row in enumerate(display_summary.head(5).itertuples(index=False)):
            keyword = escape(str(row.keywords or "No keyword"))
            representative = escape(str(row.representative_comment or "No representative comment"))
            rows.append(f'<div class="topic-row"><div class="topic-index" style="background:{ACCENTS[index % len(ACCENTS)]}">T{int(row.cluster):02d}</div><div class="topic-body"><div class="topic-title">{keyword}</div><div class="topic-sub">{representative}</div></div><div class="topic-count">{int(row.count):,}</div></div>')
        st.markdown(f'<div class="topic-list">{"".join(rows) if rows else "<div class=muted>No topic matches this filter.</div>"}</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-head"><h2>Signal composition</h2><span>Current workspace</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.plotly_chart(signal_mix_figure(result), use_container_width=True, config={"displayModeBar": False}, theme=None)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-head"><h3>System status</h3><span>Healthy</span></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="status-card">
              <div class="status-line"><span>CSV validation</span><span class="status-ok">● Passed</span></div>
              <div class="status-line"><span>Embedding model</span><span class="status-ok">● Ready</span></div>
              <div class="status-line"><span>Cluster map</span><span class="status-ok">● PCA + UMAP</span></div>
              <div class="status-line"><span>LLM enrichment</span><span class="status-value">○ Optional</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-head"><h2>Action queue</h2><span>What to do next</span></div>', unsafe_allow_html=True)
    llm_items = st.session_state.get("llm_summary")
    queue = []
    if isinstance(llm_items, list) and llm_items:
        for item in llm_items[:3]:
            queue.append(("#ffb76b", str(item.get("action", "Review this topic")), str(item.get("title", "LLM recommendation"))))
    else:
        for row in topic_summary.head(3).itertuples(index=False):
            queue.append((ACCENTS[int(row.cluster) % len(ACCENTS)], f"Review Cluster {int(row.cluster)} representative comments", str(row.keywords or "Topic signal")))
    st.markdown('<div class="queue-list">' + "".join(f'<div class="queue-item"><span class="queue-dot" style="background:{color};box-shadow:0 0 11px {color}"></span><div class="queue-text">{escape(text)}<small>{escape(sub)}</small></div></div>' for color, text, sub in queue) + '</div>', unsafe_allow_html=True)

    with st.expander("View topic summary table"):
        st.dataframe(display_summary, use_container_width=True, hide_index=True)


def render_topic_lab(result: dict[str, Any], selected_cluster: int | None) -> None:
    st.markdown('<div class="eyebrow">EXPLORATION</div><h2>Topic Lab</h2><p class="muted">발견된 토픽의 구조와 대표 의견을 확인합니다.</p>', unsafe_allow_html=True)
    map_choice = st.radio("Projection", ["PCA", "UMAP"], horizontal=True)
    figure = result["topic_map_pca"] if map_choice == "PCA" else result["topic_map_umap"]
    st.plotly_chart(style_figure(figure, 540), use_container_width=True, config={"displayModeBar": False}, theme=None)
    summary = result["topic_summary"]
    if selected_cluster is not None:
        summary = summary[summary["cluster"] == selected_cluster]
    st.markdown('<div class="section-head"><h3>Topic registry</h3><span>Keywords + representative comments</span></div>', unsafe_allow_html=True)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    reps = result["representatives"]
    if selected_cluster is not None:
        reps = reps[reps["cluster"] == selected_cluster]
    for cluster_id in summary["cluster"].tolist():
        cluster_reps = reps[reps["cluster"] == cluster_id]
        with st.expander(f"Cluster {int(cluster_id)} representative comments"):
            st.dataframe(cluster_reps, use_container_width=True, hide_index=True)


def render_semantic_search(result: dict[str, Any], selected_cluster: int | None) -> None:
    st.markdown('<div class="eyebrow">RETRIEVAL</div><h2>Semantic Search</h2><p class="muted">검색어와 의미적으로 가까운 원문 의견을 찾아 조사 속도를 높입니다.</p>', unsafe_allow_html=True)
    with st.form("semantic_search_form"):
        query = st.text_input("Search query", placeholder="예: 청년 취업 지원 정보를 한곳에서 보고 싶어요")
        col_1, col_2, col_3 = st.columns([1, 1, 1])
        with col_1:
            top_k = st.slider("Top-K", 1, 20, 5)
        with col_2:
            threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.0, .05)
        with col_3:
            st.markdown('<div class="soft-pill" style="margin-top:1.85rem;display:inline-block;">Cosine similarity · multilingual</div>', unsafe_allow_html=True)
        search_clicked = st.form_submit_button("Search signals", type="primary")

    if search_clicked:
        if not query.strip():
            st.warning("검색어를 입력해주세요.")
        else:
            source_frame = result["df"]
            source_embeddings = result["embeddings"]
            if selected_cluster is not None:
                mask = source_frame["cluster"].eq(selected_cluster).to_numpy()
                source_frame = source_frame.loc[mask].reset_index(drop=True)
                source_embeddings = source_embeddings[mask]
            with st.spinner("검색 문장을 임베딩하는 중입니다..."):
                st.session_state["search_frame"] = semantic_search(query, get_model(), source_embeddings, source_frame, int(top_k), float(threshold))

    search_frame = st.session_state.get("search_frame")
    if search_frame is None:
        st.markdown('<div class="dashboard-card"><span class="muted">검색어를 입력하면 가장 가까운 의견이 이곳에 표시됩니다.</span></div>', unsafe_allow_html=True)
        return
    if search_frame.empty:
        st.warning("threshold를 만족하는 결과가 없습니다. threshold를 낮춰보세요.")
    else:
        st.markdown(f'<div class="soft-pill" style="display:inline-block;margin:.9rem 0 .65rem;">{len(search_frame)} signals matched</div>', unsafe_allow_html=True)
        st.dataframe(search_frame, use_container_width=True, hide_index=True)
        st.download_button("Download search results", csv_bytes(search_frame), "ai-insight-search-results.csv", "text/csv")


def render_sentiment(result: dict[str, Any], selected_cluster: int | None) -> None:
    st.markdown('<div class="eyebrow">EMOTION LAYER</div><h2>Sentiment</h2><p class="muted">토픽별 감정 온도를 확인해 어떤 이슈가 가장 날카로운지 봅니다.</p>', unsafe_allow_html=True)
    if st.button("Run multilingual sentiment model", type="primary"):
        with st.spinner("감성 모델을 다운로드하고 의견을 분류 중입니다..."):
            try:
                annotated, summary = analyze_sentiment(result["df"])
                st.session_state["sentiment_frame"] = annotated
                st.session_state["sentiment_summary"] = summary
            except Exception as error:
                st.error(f"감성 분석에 실패했습니다: {error}")

    summary = st.session_state.get("sentiment_summary")
    annotated = st.session_state.get("sentiment_frame")
    if summary is None or annotated is None:
        st.markdown('<div class="dashboard-card"><span class="muted">다국어 감성 모델을 실행하면 positive / neutral / negative 비율이 표시됩니다.</span></div>', unsafe_allow_html=True)
        return
    display_summary = summary if selected_cluster is None else summary[summary["cluster"] == selected_cluster]
    cards = st.columns(3)
    for column, label, color in zip(cards, ["positive", "neutral", "negative"], ["#5ce1a8", "#68e8ff", "#ff7e9f"]):
        with column:
            total = int(display_summary[label].sum())
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label.title()}</div><div class="metric-value" style="color:{color}">{total:,}</div><div class="metric-note" style="color:{color}">sentiment signals</div></div>', unsafe_allow_html=True)
    st.write("")
    st.dataframe(display_summary, use_container_width=True, hide_index=True)
    st.bar_chart(display_summary.set_index("cluster")[["positive", "neutral", "negative"]])
    with st.expander("View labeled comments"):
        display_comments = annotated if selected_cluster is None else annotated[annotated["cluster"] == selected_cluster]
        st.dataframe(display_comments, use_container_width=True, hide_index=True)
    st.download_button("Download annotated comments", csv_bytes(annotated), "ai-insight-annotated-comments.csv", "text/csv", key="download_sentiment")


def render_llm(result: dict[str, Any]) -> None:
    st.markdown('<div class="eyebrow">AUGMENTATION STUDIO</div><h2>LLM Studio</h2><p class="muted">클러스터 결과를 Issue · Root Cause · Action 언어로 정리합니다.</p>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-card"><b>Private by default.</b><br><span class="muted">API key는 코드나 저장소에 넣지 않고 현재 세션 또는 Streamlit Secrets에서만 읽습니다.</span></div>', unsafe_allow_html=True)
    api_key = st.text_input("LLM API key", value=secret_value("OPENAI_API_KEY"), type="password", placeholder="sk-...")
    col_1, col_2 = st.columns(2)
    with col_1:
        model = st.text_input("Model", DEFAULT_MODEL)
    with col_2:
        base_url = st.text_input("OpenAI-compatible base URL", DEFAULT_BASE_URL)
    action_col, sentiment_col = st.columns(2)
    with action_col:
        if st.button("Generate Issue / Root Cause / Action", type="primary", use_container_width=True):
            try:
                with st.spinner("토픽별 인사이트를 생성 중입니다..."):
                    st.session_state["llm_summary"] = llm_topic_summary(result["topic_summary"], api_key, base_url, model)
            except Exception as error:
                st.error(f"LLM 요약에 실패했습니다: {error}")
    with sentiment_col:
        if st.button("Classify sentiment with LLM", use_container_width=True):
            try:
                with st.spinner("LLM 감성 분석을 실행 중입니다..."):
                    annotated, summary = run_chunked_llm_sentiment(result["df"], api_key, base_url, model)
                    st.session_state["sentiment_frame"] = annotated
                    st.session_state["sentiment_summary"] = summary
            except Exception as error:
                st.error(f"LLM 감성 분석에 실패했습니다: {error}")
    if st.session_state.get("llm_summary") is not None:
        st.markdown('<div class="section-head"><h3>Generated action cards</h3><span>LLM output</span></div>', unsafe_allow_html=True)
        st.json(st.session_state["llm_summary"])


def render_exports(result: dict[str, Any]) -> None:
    st.markdown('<div class="eyebrow">HANDOFF</div><h2>Exports</h2><p class="muted">분석 결과를 조사·보고서·후속 조치에 사용할 수 있는 CSV로 내보냅니다.</p>', unsafe_allow_html=True)
    sentiment_frame = st.session_state.get("sentiment_frame")
    annotated = sentiment_frame if sentiment_frame is not None else result["df"]
    export_cards = st.columns(3)
    exports = [
        ("Topic summary", result["topic_summary"], "ai-insight-topic-summary.csv", "키워드, 대표 의견, count"),
        ("Clustered comments", result["df"], "ai-insight-clustered-comments.csv", "원문 + cluster label"),
        ("Annotated comments", annotated, "ai-insight-annotated-comments.csv", "감성 label 포함"),
    ]
    for column, (title, data, filename, description) in zip(export_cards, exports):
        with column:
            st.markdown(f'<div class="dashboard-card"><div class="eyebrow">CSV</div><h3>{title}</h3><p class="muted">{description}</p></div>', unsafe_allow_html=True)
            st.download_button(f"Download {title}", csv_bytes(data), filename, "text/csv", use_container_width=True, key=filename)
    search_frame = st.session_state.get("search_frame")
    if search_frame is not None:
        st.markdown('<div class="section-head"><h3>Latest search export</h3><span>Most recent query</span></div>', unsafe_allow_html=True)
        st.dataframe(search_frame, use_container_width=True, hide_index=True)
        st.download_button("Download latest search results", csv_bytes(search_frame), "ai-insight-search-results.csv", "text/csv", key="download_export_search")


def render_quality_lab(result: dict[str, Any]) -> None:
    st.markdown('<div class="eyebrow">MODEL QUALITY</div><h2>Quality Lab</h2><p class="muted">k 선택, 문장 길이, 의미 유사도 실험을 확인합니다.</p>', unsafe_allow_html=True)
    silhouette = result["silhouette_scores"]
    figure = px.line(silhouette, x="k", y="silhouette_score", markers=True, title="Silhouette score by k")
    st.plotly_chart(style_figure(figure, 360), use_container_width=True, config={"displayModeBar": False}, theme=None)
    col_1, col_2 = st.columns(2)
    inspection = result["df"][["id", "text"]].copy()
    inspection["n_chars"] = inspection["text"].str.len()
    with col_1:
        st.markdown('<div class="section-head"><h3>Sentence length profile</h3><span>n_chars</span></div>', unsafe_allow_html=True)
        st.dataframe(inspection["n_chars"].describe().to_frame(), use_container_width=True)
    with col_2:
        st.markdown('<div class="section-head"><h3>Random inspection</h3><span>5 comments</span></div>', unsafe_allow_html=True)
        st.dataframe(inspection.sample(min(5, len(inspection)), random_state=42), use_container_width=True, hide_index=True)
    if st.button("Run cosine similarity mini experiment"):
        examples = ["지역기업 채용 정보를 찾기 어렵습니다.", "취업할 만한 회사 정보를 한곳에서 보고 싶어요.", "버스 배차간격이 너무 깁니다."]
        with st.spinner("문장 pair를 비교 중입니다..."):
            vectors = get_model().encode(examples, normalize_embeddings=True, show_progress_bar=False)
        scores = cosine_similarity(vectors)
        st.dataframe(pd.DataFrame({"pair": ["채용 ↔ 취업 회사", "채용 ↔ 버스"], "cosine_similarity": [round(float(scores[0, 1]), 4), round(float(scores[0, 2]), 4)]}), use_container_width=True, hide_index=True)


def main() -> None:
    source_bytes, source_name, n_clusters, analyze_clicked, signature, nav = render_sidebar()
    result = ensure_analysis(source_bytes, source_name, n_clusters, analyze_clicked, signature)
    render_topline(result)

    cluster_values = sorted(int(value) for value in result["df"]["cluster"].unique())
    filter_col, context_col = st.columns([1, 3])
    with filter_col:
        selected_option = st.selectbox("Cluster filter", ["All"] + cluster_values, format_func=lambda value: value if value == "All" else f"Cluster {value}")
    with context_col:
        selected_text = "All clusters" if selected_option == "All" else f"Cluster {selected_option} only"
        st.markdown(f'<div class="soft-pill" style="display:inline-block;margin-top:1.83rem;">{selected_text} · {escape(str(st.session_state.get("source_name", source_name)))}</div>', unsafe_allow_html=True)
    selected_cluster = None if selected_option == "All" else int(selected_option)

    if nav == "Overview":
        render_overview(result, selected_cluster)
    elif nav == "Topic Lab":
        render_topic_lab(result, selected_cluster)
    elif nav == "Semantic Search":
        render_semantic_search(result, selected_cluster)
    elif nav == "Sentiment":
        render_sentiment(result, selected_cluster)
    elif nav == "LLM Studio":
        render_llm(result)
    elif nav == "Exports":
        render_exports(result)
    else:
        render_quality_lab(result)


if __name__ == "__main__":
    main()
