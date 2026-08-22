# AI Insight Engine

청년 의견 CSV를 **Raw Text → Embedding → Clustering → Insight → Semantic Search → Visualization** 흐름으로 분석하는 웹 앱입니다.

- GitHub Pages: <https://seojaeohcode.github.io/solveton_class/>
- Repository: <https://github.com/seojaeohcode/solveton_class>
- 기본 데이터: `ai_insight_engine_youth_comments.csv`

## 주요 기능

### Part 1 — Raw Text → Embedding → Clustering

- CSV 인코딩 자동 처리: UTF-8-SIG, UTF-8, CP949
- `text` 컬럼 검증 및 결측·빈 문자열·중복 데이터 정제
- 문장 길이 통계와 샘플 문장 확인
- SentenceTransformer 기반 다국어 문장 임베딩
- K-Means 클러스터링 및 cluster별 대표 의견 확인
- `k=3~10` silhouette score 비교와 추천 k 표시
- PCA 2차원 시각화

### Part 2 — Cluster → Insight → Search → Web App

- cluster별 키워드와 대표 의견
- PCA / UMAP-Lite 토픽 맵
- 토픽 카드 또는 지도 점을 클릭하는 필터링
- 의미 기반 검색과 similarity threshold 조절
- 검색 결과 CSV 다운로드
- 분석 요약·주석 의견·cluster 결과 CSV 다운로드
- topic별 positive / neutral / negative 비율
- 입력한 LLM API key를 이용한 Issue / Root Cause / Action 요약
- LLM 기반 토픽 감성 분석

## GitHub Pages에서 사용하기

정적 웹 앱은 브라우저에서 바로 실행됩니다.

1. [AI Insight Engine](https://seojaeohcode.github.io/solveton_class/)에 접속합니다.
2. 기본 CSV가 자동으로 선택·분석됩니다.
3. 다른 파일을 사용하려면 `text` 컬럼이 포함된 CSV를 업로드합니다.
4. 검색어, cluster, similarity threshold, PCA/UMAP-Lite 옵션을 조절합니다.

GitHub Pages 버전은 서버 없이 실행되므로 Python 패키지를 설치할 필요가 없습니다. 브라우저 호환성을 위해 정적 버전은 TF-IDF와 단어·문자 n-gram을 사용하며, Python/Gradio 버전의 SentenceTransformer 임베딩과 결과가 완전히 같지는 않습니다.

`main` 브랜치에 push하면 `.github/workflows/deploy-pages.yml`이 `docs/` 폴더를 자동 배포합니다.

## LLM API key 사용

웹 앱의 `LLM Console`에서 다음 값을 입력할 수 있습니다.

- API key
- model name: 기본값 `gpt-4o-mini`
- OpenAI-compatible base URL: 기본값 `https://api.openai.com/v1`

Pages 버전에서는 입력한 key를 브라우저 메모리에만 보관하고 저장하지 않습니다. 그래도 공개 웹 페이지에서 직접 입력한 key는 브라우저 네트워크 요청에 사용되므로 공용 PC나 타인과 공유하는 환경에서는 사용하지 마세요. 사용이 끝나면 `Clear key`를 누르고, 필요하면 API 제공자의 대시보드에서 key를 폐기하세요.

LLM key가 없어도 클러스터링, 검색, 지도, 기본 감성 분석, CSV 다운로드 기능은 사용할 수 있습니다.

## 로컬 Python/Gradio 실행

Python 3.11 환경을 권장합니다. 의존성 버전은 `requirements.txt`에 고정되어 있습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

실행 후 <http://localhost:7860>을 엽니다.

처음 실행하면 다음 모델을 다운로드할 수 있습니다.

- Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Sentiment: `cardiffnlp/twitter-xlm-roberta-base-sentiment`

## Docker 실행

Docker Desktop을 실행한 뒤 프로젝트 루트에서 실행합니다.

```powershell
docker compose build
docker compose up
```

실행 후 <http://localhost:7860>에 접속합니다.

단일 이미지로 실행하려면 다음 명령을 사용합니다.

```powershell
docker build -t ai-insight-engine .
docker run --rm -p 7860:7860 `
  -e GRADIO_SERVER_NAME=0.0.0.0 `
  ai-insight-engine
```

Docker 이미지는 Python 3.11 기반이며 다음 검사를 포함합니다.

- 고정 버전 의존성 설치
- `pip check`
- Python 문법 컴파일 검사
- CPU 실행 환경 설정

Hugging Face 모델 캐시는 `huggingface-cache` Docker volume에 보존되어 다음 실행부터 재사용됩니다.

## CSV 형식

분석 대상 컬럼인 `text`는 필수입니다. `id`는 선택 사항입니다.

```csv
id,text
1,"청년들이 이용할 수 있는 취업 지원 정보가 한곳에 모이면 좋겠습니다."
2,"지역 청년을 위한 상담 시간이 더 다양했으면 합니다."
```

업로드 시 다음 정제가 수행됩니다.

- 앞뒤 공백 제거
- null text 제거
- 빈 문자열 제거
- 중복 text 제거
- index 재설정

## 프로젝트 구조

```text
.
├─ docs/                              # GitHub Pages 정적 웹 앱
│  ├─ index.html
│  ├─ styles.css
│  ├─ app.js
│  ├─ ai_insight_engine_youth_comments.csv
│  └─ .nojekyll
├─ analysis.py                        # Python 분석 함수
├─ app.py                             # Gradio 앱
├─ requirements.txt                   # 고정 Python 의존성
├─ Dockerfile                         # 재현 가능한 CPU 이미지
├─ docker-compose.yml                 # Docker Compose 실행 설정
├─ ai_insight_engine_youth_comments.csv # 기본 CSV
├─ sample-data.csv                    # 테스트용 샘플 CSV
├─ LICENSE
└─ .github/workflows/deploy-pages.yml # Pages 자동 배포
```

## 문제 해결

### GitHub Pages가 갱신되지 않는 경우

1. GitHub 저장소의 `Actions` 탭에서 `Deploy AI Insight Engine to GitHub Pages` 실행 결과를 확인합니다.
2. 브라우저에서 `Ctrl+F5`로 캐시를 새로고침합니다.
3. Pages 주소가 `https://seojaeohcode.github.io/solveton_class/`인지 확인합니다.

### CSV가 로드되지 않는 경우

- 파일 확장자가 `.csv`인지 확인합니다.
- 컬럼명이 정확히 `text`인지 확인합니다.
- 로컬 파일을 직접 여는 대신 HTTP 서버나 GitHub Pages 주소로 접속합니다.

### Docker 실행이 느린 경우

첫 실행에서는 임베딩·감성 모델을 다운로드하므로 시간이 걸릴 수 있습니다. 이후에는 Docker volume의 모델 캐시를 사용합니다.

## 라이선스

이 프로젝트의 라이선스는 저장소의 [LICENSE](./LICENSE) 파일을 확인하세요.
