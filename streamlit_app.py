"""Streamlit entrypoint for the AI Insight Engine."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
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

st.set_page_config(
    page_title="AI Insight Engine",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1480px; padding-top: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.65rem; }
    .hero { padding: 1.2rem 1.35rem; border: 1px solid rgba(120, 180, 255, .2);
            border-radius: 18px; background: linear-gradient(135deg, #111a35, #182a45); }
    .muted { color: #9eacc5; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_bytes_with_project_parser(data: bytes) -> pd.DataFrame:
    """Write uploaded bytes to a temporary path for the shared CSV parser."""

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
            handle.write(data)
            temporary_path = handle.name
        from analysis import read_and_clean_csv

        return read_and_clean_csv(temporary_path)
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
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


def sentiment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Build cluster-level sentiment counts and ratios."""

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
    annotated = frame.drop(columns=["sentiment"], errors="ignore").merge(
        labels, on="id", how="left"
    )
    annotated["sentiment"] = annotated["sentiment"].fillna("neutral")
    return annotated, sentiment_summary(annotated)


def reset_derived_state() -> None:
    for key in ("sentiment_frame", "sentiment_summary", "search_frame", "llm_summary"):
        st.session_state.pop(key, None)


def render_sidebar() -> tuple[bytes, str, int, bool, str]:
    with st.sidebar:
        st.markdown("## 🧭 AI Insight Engine")
        st.caption("Cluster → Insight → Search → Map → Sentiment → Action")
        st.divider()

        uploaded = st.file_uploader(
            "분석할 CSV",
            type=["csv"],
            help="text 컬럼이 필요합니다. 파일을 선택하지 않으면 기본 CSV를 사용합니다.",
        )

        if uploaded is None:
            if not DEFAULT_CSV.exists():
                st.error("기본 CSV 파일을 찾을 수 없습니다.")
                st.stop()
            source_bytes = DEFAULT_CSV.read_bytes()
            source_name = DEFAULT_CSV.name
            st.success(f"기본 CSV 사용 중: {source_name}")
        else:
            source_bytes = uploaded.getvalue()
            source_name = uploaded.name
            st.success(f"업로드 파일: {source_name}")

        n_clusters = st.slider(
            "클러스터 수 (k)",
            min_value=2,
            max_value=15,
            value=7,
            step=1,
            help="K-Means에 사용할 클러스터 수입니다.",
        )
        analyze_clicked = st.button("분석 실행", type="primary", use_container_width=True)

        st.divider()
        st.markdown("### 실행 환경")
        st.caption("Streamlit Community Cloud / Python 3.12 권장")
        st.caption("첫 분석 시 임베딩 모델 다운로드로 시간이 걸릴 수 있습니다.")

    signature = hashlib.sha256(source_bytes).hexdigest()
    return source_bytes, source_name, n_clusters, analyze_clicked, signature


def ensure_analysis(
    source_bytes: bytes,
    source_name: str,
    n_clusters: int,
    analyze_clicked: bool,
    source_signature: str,
) -> dict[str, Any]:
    previous_signature = st.session_state.get("source_signature")
    previous_k = st.session_state.get("analysis_k")
    should_run = (
        analyze_clicked
        or "analysis" not in st.session_state
        or previous_signature != source_signature
    )

    if should_run:
        with st.spinner(
            "CSV 정제, 문장 임베딩, K-Means, silhouette, PCA/UMAP을 계산 중입니다..."
        ):
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
        reset_derived_state()
    elif previous_k != int(n_clusters):
        st.sidebar.warning("k가 변경되었습니다. 새 결과를 보려면 ‘분석 실행’을 누르세요.")

    return st.session_state["analysis"]


def render_header(result: dict[str, Any]) -> None:
    frame = result["df"]
    silhouette = result["silhouette_scores"]
    best_score = float(silhouette["silhouette_score"].max()) if not silhouette.empty else 0.0

    st.markdown(
        """
        <div class="hero">
          <h1>AI Insight Engine</h1>
          <p class="muted">청년 의견 데이터를 클러스터링하고, 의미 기반 검색과 LLM 인사이트로 연결합니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("분석 의견", f"{len(frame):,}개")
    metric_2.metric("현재 토픽", f"{frame['cluster'].nunique():,}개")
    metric_3.metric("추천 k", f"{result['recommended_k']}")
    metric_4.metric("최고 silhouette", f"{best_score:.4f}")


def render_overview(result: dict[str, Any], selected_cluster: int | None) -> None:
    frame = result["df"]
    topic_summary = result["topic_summary"].copy()
    representatives = result["representatives"]

    if selected_cluster is not None:
        topic_summary = topic_summary[topic_summary["cluster"] == selected_cluster]
        representatives = representatives[representatives["cluster"] == selected_cluster]

    st.subheader("Topic Summary")
    st.dataframe(topic_summary, use_container_width=True, hide_index=True)

    st.subheader("Representative Comments")
    for row in topic_summary.itertuples(index=False):
        with st.expander(f"Cluster {row.cluster} · {row.count}개 · {row.keywords or '키워드 없음'}"):
            comments = representatives[representatives["cluster"] == row.cluster]
            st.dataframe(comments, use_container_width=True, hide_index=True)

    st.caption(f"정제 후 {len(frame):,}개의 고유 의견을 분석했습니다.")


def render_topic_map(result: dict[str, Any], selected_cluster: int | None) -> None:
    choice = st.radio("시각화 방법", ["PCA", "UMAP"], horizontal=True)
    figure = result["topic_map_pca"] if choice == "PCA" else result["topic_map_umap"]
    if selected_cluster is not None:
        st.info(f"현재 Cluster {selected_cluster}가 선택되어 있습니다. 지도는 전체 의견을 표시합니다.")
    st.plotly_chart(figure, use_container_width=True, theme=None)


def render_semantic_search(result: dict[str, Any], selected_cluster: int | None) -> None:
    st.subheader("Semantic Search")
    st.caption("문장을 입력하면 SentenceTransformer 임베딩의 cosine similarity로 가까운 의견을 찾습니다.")

    with st.form("semantic_search_form"):
        query = st.text_input("검색어", placeholder="예: 청년 취업 지원 정보를 한곳에서 보고 싶어요")
        col_1, col_2 = st.columns(2)
        with col_1:
            top_k = st.slider("Top-K", min_value=1, max_value=20, value=5)
        with col_2:
            threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        search_clicked = st.form_submit_button("검색 실행", type="primary")

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
            with st.spinner("검색 문장을 임베딩하고 유사도를 계산 중입니다..."):
                st.session_state["search_frame"] = semantic_search(
                    query=query,
                    model=get_model(),
                    embeddings=source_embeddings,
                    frame=source_frame,
                    top_k=int(top_k),
                    threshold=float(threshold),
                )

    search_frame = st.session_state.get("search_frame")
    if search_frame is None:
        st.info("검색어를 입력하고 검색 실행을 눌러주세요.")
        return

    if search_frame.empty:
        st.warning("threshold를 만족하는 결과가 없습니다. threshold를 낮춰보세요.")
    else:
        st.dataframe(search_frame, use_container_width=True, hide_index=True)
        st.download_button(
            "검색 결과 CSV 다운로드",
            data=csv_bytes(search_frame),
            file_name="ai-insight-search-results.csv",
            mime="text/csv",
        )


def render_sentiment(result: dict[str, Any], selected_cluster: int | None) -> None:
    st.subheader("Sentiment Analysis")
    st.caption("다국어 Transformers 감성 모델을 요청할 때만 실행합니다.")

    if st.button("다국어 감성 모델 실행", type="primary"):
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
        st.info("감성 모델 실행 버튼을 눌러 결과를 계산하세요.")
        return

    display_summary = summary if selected_cluster is None else summary[summary["cluster"] == selected_cluster]
    st.dataframe(display_summary, use_container_width=True, hide_index=True)
    chart_data = display_summary.set_index("cluster")[["positive", "neutral", "negative"]]
    st.bar_chart(chart_data)

    display_comments = annotated if selected_cluster is None else annotated[annotated["cluster"] == selected_cluster]
    with st.expander("감성 라벨이 붙은 의견 보기"):
        st.dataframe(display_comments, use_container_width=True, hide_index=True)

    st.download_button(
        "감성 분석 CSV 다운로드",
        data=csv_bytes(annotated),
        file_name="ai-insight-annotated-comments.csv",
        mime="text/csv",
        key="download_model_sentiment",
    )


def render_llm(result: dict[str, Any]) -> None:
    st.subheader("LLM Console")
    st.caption("API key는 입력한 세션에서만 사용합니다. 저장소나 코드에는 넣지 마세요.")

    secret_key = secret_value("OPENAI_API_KEY")
    api_key = st.text_input(
        "LLM API key",
        value=secret_key,
        type="password",
        placeholder="sk-...",
        help="Streamlit Cloud에서는 App settings의 Secrets에 OPENAI_API_KEY를 넣을 수도 있습니다.",
    )
    col_1, col_2 = st.columns(2)
    with col_1:
        model = st.text_input("Model", value=DEFAULT_MODEL)
    with col_2:
        base_url = st.text_input("OpenAI-compatible base URL", value=DEFAULT_BASE_URL)

    summary_col, sentiment_col = st.columns(2)
    with summary_col:
        if st.button("Issue / Root Cause / Action 생성", type="primary", use_container_width=True):
            try:
                with st.spinner("토픽별 인사이트를 생성 중입니다..."):
                    st.session_state["llm_summary"] = llm_topic_summary(
                        result["topic_summary"], api_key, base_url, model
                    )
            except Exception as error:
                st.error(f"LLM 요약에 실패했습니다: {error}")
    with sentiment_col:
        if st.button("LLM 감성 분석 실행", use_container_width=True):
            try:
                with st.spinner("LLM으로 의견 감성을 분류 중입니다..."):
                    annotated, summary = run_chunked_llm_sentiment(
                        result["df"], api_key, base_url, model
                    )
                    st.session_state["sentiment_frame"] = annotated
                    st.session_state["sentiment_summary"] = summary
            except Exception as error:
                st.error(f"LLM 감성 분석에 실패했습니다: {error}")

    if st.session_state.get("llm_summary") is not None:
        st.json(st.session_state["llm_summary"])


def render_downloads(result: dict[str, Any]) -> None:
    st.subheader("Downloads")
    sentiment_frame = st.session_state.get("sentiment_frame")
    annotated = sentiment_frame if sentiment_frame is not None else result["df"]

    download_col_1, download_col_2 = st.columns(2)
    with download_col_1:
        st.download_button(
            "토픽 요약 CSV",
            data=csv_bytes(result["topic_summary"]),
            file_name="ai-insight-topic-summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_col_2:
        st.download_button(
            "클러스터 결과 CSV",
            data=csv_bytes(result["df"]),
            file_name="ai-insight-clustered-comments.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.download_button(
        "주석·감성 의견 CSV",
        data=csv_bytes(annotated),
        file_name="ai-insight-annotated-comments.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_part_one(result: dict[str, Any]) -> None:
    st.subheader("Part 1 — Raw Text → Embedding → Clustering")
    st.markdown(
        "`CSV → Validation/Cleaning → Sentence Embedding → Similarity → K-Means → Evaluation → 2D Visualization`"
    )
    st.write(
        "Embedding은 문장을 숫자 벡터로 바꾸고, K-Means는 벡터의 거리를 기준으로 의견을 묶습니다. "
        "silhouette score는 k 선택을 데이터 관점에서 비교하는 데 사용합니다."
    )

    silhouette = result["silhouette_scores"]
    st.dataframe(silhouette, use_container_width=True, hide_index=True)
    if not silhouette.empty:
        figure = px.line(
            silhouette,
            x="k",
            y="silhouette_score",
            markers=True,
            title="Silhouette score by k",
        )
        st.plotly_chart(figure, use_container_width=True, theme=None)

    inspection = result["df"][["id", "text"]].copy()
    inspection["n_chars"] = inspection["text"].str.len()
    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown("**문장 길이 요약**")
        st.dataframe(inspection["n_chars"].describe().to_frame(), use_container_width=True)
    with col_2:
        st.markdown("**랜덤 문장 5개**")
        st.dataframe(inspection.sample(min(5, len(inspection)), random_state=42), use_container_width=True, hide_index=True)

    st.info(
        "사용 모델: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 · "
        "embedding은 batch_size=64, normalize_embeddings=True로 계산했습니다."
    )

    if st.button("Mini Experiment: cosine similarity 계산"):
        examples = [
            "지역기업 채용 정보를 찾기 어렵습니다.",
            "취업할 만한 회사 정보를 한곳에서 보고 싶어요.",
            "버스 배차간격이 너무 깁니다.",
        ]
        with st.spinner("문장 pair의 의미 유사도를 계산 중입니다..."):
            mini_embeddings = get_model().encode(
                examples,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        mini_scores = cosine_similarity(mini_embeddings)
        mini_results = pd.DataFrame(
            {
                "pair": [
                    "채용 정보 ↔ 취업 회사 정보",
                    "채용 정보 ↔ 버스 배차간격",
                ],
                "cosine_similarity": [
                    round(float(mini_scores[0, 1]), 4),
                    round(float(mini_scores[0, 2]), 4),
                ],
            }
        )
        st.dataframe(mini_results, use_container_width=True, hide_index=True)


def main() -> None:
    source_bytes, source_name, n_clusters, analyze_clicked, source_signature = render_sidebar()
    result = ensure_analysis(
        source_bytes,
        source_name,
        n_clusters,
        analyze_clicked,
        source_signature,
    )

    render_header(result)
    st.caption(f"현재 데이터: {st.session_state.get('source_name', source_name)} · 분석 k={st.session_state.get('analysis_k', n_clusters)}")

    clusters = sorted(int(value) for value in result["df"]["cluster"].unique())
    selected_option = st.selectbox(
        "표시할 cluster",
        options=["전체"] + clusters,
        format_func=lambda value: value if value == "전체" else f"Cluster {value}",
    )
    selected_cluster = None if selected_option == "전체" else int(selected_option)

    tabs = st.tabs(
        [
            "Overview",
            "Topic Map",
            "Semantic Search",
            "Sentiment",
            "LLM Console",
            "Downloads",
            "Part 1",
        ]
    )
    with tabs[0]:
        render_overview(result, selected_cluster)
    with tabs[1]:
        render_topic_map(result, selected_cluster)
    with tabs[2]:
        render_semantic_search(result, selected_cluster)
    with tabs[3]:
        render_sentiment(result, selected_cluster)
    with tabs[4]:
        render_llm(result)
    with tabs[5]:
        render_downloads(result)
    with tabs[6]:
        render_part_one(result)


if __name__ == "__main__":
    main()
