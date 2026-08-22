"""Advanced Gradio app for local execution or Hugging Face Spaces."""

from pathlib import Path
import os
import tempfile

import gradio as gr

from analysis import (
    analyze_sentiment,
    build_analysis,
    get_model,
    llm_sentiment,
    llm_topic_summary,
    semantic_search,
)

DEFAULT_CSV = Path(__file__).with_name("ai_insight_engine_youth_comments.csv")


def run_analysis(file_path, n_clusters):
    if file_path is None:
        raise gr.Error("CSV 파일을 먼저 업로드해주세요.")
    try:
        result = build_analysis(file_path, int(n_clusters))
    except Exception as error:
        raise gr.Error(f"분석 중 오류가 발생했습니다: {error}") from error

    frame = result["df"]
    message = f"### 분석된 의견 수: {len(frame):,}개 · {len(result['topic_summary']):,}개 토픽"
    recommendation = f"추천 k: {result['recommended_k']} · silhouette 최고값: {result['silhouette_scores']['silhouette_score'].max():.4f}"
    state = {"result": result, "sentiment_frame": None, "sentiment_summary": None, "search_frame": None}
    return message, recommendation, result["topic_summary"], result["topic_map_pca"], result["topic_map_umap"], state


def run_semantic_search(query, top_k, threshold, state):
    if not state:
        raise gr.Error("먼저 CSV 파일을 업로드하고 Analyze를 실행해주세요.")
    if not query or not query.strip():
        raise gr.Error("검색어를 입력해주세요.")
    result = state["result"]
    search_frame = semantic_search(
        query=query,
        model=get_model(),
        embeddings=result["embeddings"],
        frame=result["df"],
        top_k=int(top_k),
        threshold=float(threshold),
    )
    state["search_frame"] = search_frame
    return search_frame, state


def run_model_sentiment(state):
    if not state:
        raise gr.Error("먼저 분석을 실행해주세요.")
    try:
        annotated, summary = analyze_sentiment(state["result"]["df"])
    except Exception as error:
        raise gr.Error(f"감성 모델 실행 중 오류가 발생했습니다: {error}") from error
    state["sentiment_frame"] = annotated
    state["sentiment_summary"] = summary
    return summary, state


def run_llm_summary(api_key, base_url, model, state):
    if not state:
        raise gr.Error("먼저 분석을 실행해주세요.")
    try:
        return llm_topic_summary(state["result"]["topic_summary"], api_key, base_url, model)
    except Exception as error:
        raise gr.Error(f"LLM 요약 중 오류가 발생했습니다: {error}") from error


def run_llm_sentiment(api_key, base_url, model, state):
    if not state:
        raise gr.Error("먼저 분석을 실행해주세요.")
    try:
        annotated, summary = llm_sentiment(state["result"]["df"], api_key, base_url, model)
    except Exception as error:
        raise gr.Error(f"LLM 감성 분석 중 오류가 발생했습니다: {error}") from error
    state["sentiment_frame"] = annotated
    state["sentiment_summary"] = summary
    return summary, state


def write_download(frame, filename):
    if frame is None:
        return None
    path = Path(tempfile.gettempdir()) / filename
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def export_summary(state):
    if not state:
        raise gr.Error("먼저 분석을 실행해주세요.")
    return write_download(state["result"]["topic_summary"], "ai-insight-topic-summary.csv")


def export_comments(state):
    if not state:
        raise gr.Error("먼저 분석을 실행해주세요.")
    frame = state.get("sentiment_frame") if state.get("sentiment_frame") is not None else state["result"]["df"]
    return write_download(frame, "ai-insight-annotated-comments.csv")


def export_search(state):
    if not state or state.get("search_frame") is None:
        raise gr.Error("먼저 Semantic Search를 실행해주세요.")
    return write_download(state["search_frame"], "ai-insight-search-results.csv")


with gr.Blocks(title="AI Insight Engine — Advanced", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # AI Insight Engine — Advanced
        **Cluster → Insight → Search → Map → Sentiment → Action**

        같은 CSV로 Part 1의 embedding / silhouette부터 Part 2의 검색 / LLM 요약까지 실행합니다.
        """
    )
    analysis_state = gr.State(None)

    with gr.Row():
        csv_input = gr.File(label="CSV Upload", file_types=[".csv"], type="filepath", value=str(DEFAULT_CSV) if DEFAULT_CSV.exists() else None)
        n_clusters_input = gr.Slider(minimum=2, maximum=15, value=7, step=1, label="Number of Topics (k)")
    analyze_button = gr.Button("Analyze", variant="primary")
    count_output = gr.Markdown("### 분석된 의견 수: -")
    recommendation_output = gr.Markdown("추천 k: -")
    topic_summary_output = gr.Dataframe(label="Topic Summary", interactive=False)

    with gr.Tabs():
        with gr.Tab("PCA Topic Map"):
            pca_output = gr.Plot(label="PCA Topic Map")
        with gr.Tab("UMAP Topic Map"):
            umap_output = gr.Plot(label="UMAP Topic Map")

    gr.Markdown("## Semantic Search")
    with gr.Row():
        query_input = gr.Textbox(label="Query", placeholder="예: 취업 지원이 필요해요")
        top_k_input = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Top-K")
        threshold_input = gr.Slider(minimum=0, maximum=1, value=0, step=0.05, label="Similarity threshold")
    search_button = gr.Button("Semantic Search")
    search_output = gr.Dataframe(label="Semantic Search Results", interactive=False)
    export_search_button = gr.Button("Download search results CSV")

    gr.Markdown("## Sentiment Analysis")
    sentiment_button = gr.Button("Run multilingual sentiment model")
    sentiment_output = gr.Dataframe(label="Positive / Neutral / Negative by cluster", interactive=False)

    gr.Markdown("## LLM Augmentation")
    gr.Markdown("API key는 입력값으로만 사용됩니다. 로컬/Spaces에서는 서버 요청으로, Pages 정적 앱에서는 브라우저 요청으로 provider에 전송됩니다.")
    with gr.Row():
        llm_key = gr.Textbox(label="LLM API key", type="password", placeholder="sk-...")
        llm_model = gr.Textbox(label="Model", value="gpt-4o-mini")
        llm_base_url = gr.Textbox(label="OpenAI-compatible base URL", value="https://api.openai.com/v1")
    with gr.Row():
        llm_summary_button = gr.Button("Generate Issue / Root Cause / Action", variant="primary")
        llm_sentiment_button = gr.Button("Classify sentiment with LLM")
    llm_summary_output = gr.JSON(label="LLM Topic Summary")

    gr.Markdown("## Downloads")
    with gr.Row():
        export_summary_button = gr.Button("Download topic summary CSV")
        export_comments_button = gr.Button("Download annotated comments CSV")
    summary_file = gr.File(label="Topic summary file")
    comments_file = gr.File(label="Annotated comments file")
    search_file = gr.File(label="Search results file")

    analyze_button.click(run_analysis, inputs=[csv_input, n_clusters_input], outputs=[count_output, recommendation_output, topic_summary_output, pca_output, umap_output, analysis_state])
    search_button.click(run_semantic_search, inputs=[query_input, top_k_input, threshold_input, analysis_state], outputs=[search_output, analysis_state])
    sentiment_button.click(run_model_sentiment, inputs=[analysis_state], outputs=[sentiment_output, analysis_state])
    llm_summary_button.click(run_llm_summary, inputs=[llm_key, llm_base_url, llm_model, analysis_state], outputs=[llm_summary_output])
    llm_sentiment_button.click(run_llm_sentiment, inputs=[llm_key, llm_base_url, llm_model, analysis_state], outputs=[sentiment_output, analysis_state])
    export_summary_button.click(export_summary, inputs=[analysis_state], outputs=[summary_file])
    export_comments_button.click(export_comments, inputs=[analysis_state], outputs=[comments_file])
    export_search_button.click(export_search, inputs=[analysis_state], outputs=[search_file])


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
