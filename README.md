# AI Insight Engine

CSV 의견 데이터를 `Cluster → Insight → Search → Map` 흐름으로 탐색하는 웹 앱입니다.

## 구성

```text
.
├─ docs/                         # GitHub Pages 정적 웹 앱
│  ├─ index.html
│  ├─ styles.css
│  ├─ app.js
│  ├─ ai_insight_engine_youth_comments.csv
│  └─ .nojekyll
├─ analysis.py                   # Python 분석 함수
├─ app.py                        # Gradio 로컬/Spaces 앱
├─ requirements.txt              # 고정된 Python 실행 환경
├─ Dockerfile                    # 재현 가능한 CPU 이미지
├─ docker-compose.yml            # 로컬 컨테이너 실행
├─ ai_insight_engine_youth_comments.csv  # 기본 CSV 데이터
├─ sample-data.csv               # 바로 테스트할 샘플 데이터
└─ .github/workflows/
   └─ deploy-pages.yml          # GitHub Pages 자동 배포
```

## GitHub Pages 배포

1. 이 폴더를 GitHub 저장소의 루트에 올립니다.
2. 저장소의 `Settings → Pages`에서 Source를 `GitHub Actions`로 선택합니다.
3. `main` 브랜치에 push하면 `.github/workflows/deploy-pages.yml`이 `docs/`를 배포합니다.

정적 Pages 버전은 기본으로 `ai_insight_engine_youth_comments.csv`를 자동 분석하고, 이후 다른 CSV를 선택할 수 있습니다. 업로드한 CSV를 브라우저 안에서만 처리하므로 CSV 내용은 서버로 전송되지 않습니다. 브라우저에서 실행할 수 있도록 SentenceTransformer 대신 TF-IDF + 단어/문자 n-gram 벡터를 사용하므로, Python 버전과 검색 점수가 같지는 않습니다.

## Advanced Challenge 기능

- PCA / UMAP topic map 비교
- `k=3~10` silhouette score와 추천 k
- cluster 카드·지도 클릭 필터
- semantic search similarity threshold
- topic summary / annotated comments / search results CSV 다운로드
- topic별 positive / neutral / negative 비율
- 입력한 LLM API key로 Issue / Root Cause / Action 요약 및 감성 분류

GitHub Pages의 LLM 기능은 입력한 키를 브라우저 메모리에만 두고 OpenAI-compatible endpoint로 직접 요청합니다. API key를 저장하거나 코드에 넣지 않지만, 공개 페이지에서 직접 입력하는 키는 노출 위험이 있으므로 공용 PC에서는 사용하지 마세요. 로컬/Gradio 버전에서는 같은 key 입력을 Python 서버에서 사용합니다.

## Python/Gradio 실행

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

실행 후 Gradio가 출력하는 로컬 주소를 엽니다. 기본 CSV가 파일 입력에 미리 선택됩니다. CSV에는 반드시 `text` 컬럼이 있어야 합니다. `id` 컬럼은 선택 사항입니다.

## Docker 실행

Docker Desktop이 실행된 상태에서:

```powershell
docker compose build
docker compose up
```

브라우저에서 `http://localhost:7860`을 엽니다. 첫 실행 때 SentenceTransformer와 감성 모델을 Hugging Face에서 내려받으므로 시간이 걸릴 수 있습니다. 모델 캐시는 `huggingface-cache` Docker volume에 보존되어 다음 실행부터 재사용됩니다.

이미지를 직접 실행하려면:

```powershell
docker build -t ai-insight-engine .
docker run --rm -p 7860:7860 -e GRADIO_SERVER_NAME=0.0.0.0 ai-insight-engine
```

Docker 빌드에는 `pip check`와 Python 컴파일 검사가 포함되어 의존성 충돌이나 기본 문법 오류가 있으면 이미지 생성이 실패합니다. `requirements.txt`는 Python 3.11 기준으로 직접 의존성을 고정했습니다.

## CSV 예시

```csv
id,text
1,"청년들이 이용할 수 있는 취업 지원 정보가 한곳에 모이면 좋겠습니다."
2,"지역 청년을 위한 상담 시간이 더 다양했으면 합니다."
```
