# AI Insight Engine — Streamlit

청년 의견 CSV를 **Raw Text → Embedding → Clustering → Insight → Semantic Search → Visualization** 흐름으로 분석하는 Streamlit 웹 앱입니다.

- GitHub 저장소: <https://github.com/seojaeohcode/solveton_class>
- Streamlit 배포: Streamlit Community Cloud에서 이 저장소를 연결해 실행
- 기본 CSV: `ai_insight_engine_youth_comments.csv`

## 주요 기능

### Part 1 — Raw Text → Embedding → Clustering

- UTF-8-SIG, UTF-8, CP949 CSV 인코딩 자동 처리
- `text` 컬럼 검증
- null, 빈 문자열, 중복 의견 제거
- 문장 길이 통계와 랜덤 문장 확인
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 문장 임베딩
- K-Means 클러스터링
- `k=3~10` silhouette score 비교 및 추천 k
- PCA / UMAP 토픽 맵

### Part 2 — Cluster → Insight → Search → Web App

- cluster별 키워드와 대표 의견
- cluster 선택 필터
- SentenceTransformer 기반 semantic search
- similarity threshold와 Top-K 조절
- 다국어 Transformers 감성 분석
- LLM Issue / Root Cause / Action 요약
- LLM 감성 분석: 긴 CSV를 여러 chunk로 나누어 처리
- 토픽 요약, 클러스터, 감성, 검색 결과 CSV 다운로드

## Streamlit Community Cloud 배포

### 1. GitHub 저장소 연결

1. [Streamlit Community Cloud](https://share.streamlit.io/)에 GitHub 계정으로 로그인합니다.
2. `Create app` 또는 `Deploy an app`을 선택합니다.
3. Repository에 `seojaeohcode/solveton_class`를 선택합니다.
4. Branch는 `main`으로 선택합니다.
5. Main file path는 `streamlit_app.py`로 입력합니다.
6. Python version은 `3.12`를 선택합니다.
7. Deploy를 누릅니다.

이 저장소는 Streamlit Cloud가 바로 인식할 수 있도록 진입 파일과 `requirements.txt`를 루트에 두었습니다. 이후 `main` 브랜치에 push하면 Streamlit Cloud가 변경사항을 감지해 앱을 다시 배포합니다.

### 2. LLM Secrets 설정

LLM key는 GitHub에 올리지 않습니다. Streamlit 앱의 `Settings → Secrets`에 다음처럼 입력할 수 있습니다.

```toml
OPENAI_API_KEY = "your-api-key"
```

또는 앱 화면의 `LLM Console`에서 세션별로 직접 입력할 수 있습니다. `secrets.toml`은 `.gitignore`에 등록되어 있습니다.

LLM key가 없어도 클러스터링, 지도, 검색, 기본 감성 모델, CSV 다운로드는 사용할 수 있습니다.

## 로컬 실행

Python 3.12를 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

실행 후 <http://localhost:8501>을 엽니다.

처음 분석할 때 임베딩 모델과 감성 모델을 Hugging Face에서 다운로드하므로 시간이 걸릴 수 있습니다. 이후에는 프로세스 캐시를 사용합니다.

## Docker 실행

Docker Desktop을 실행한 뒤 프로젝트 루트에서 실행합니다.

```powershell
docker compose build
docker compose up
```

실행 후 <http://localhost:8501>에 접속합니다.

단일 이미지로 실행하려면:

```powershell
docker build -t ai-insight-engine .
docker run --rm -p 8501:8501 `
  -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 `
  ai-insight-engine
```

Docker 설정은 Python 3.12, Streamlit 포트 8501, CPU 기반 과학 계산 패키지, Hugging Face 모델 캐시 volume을 사용합니다. 이미지 생성 시 `pip check`와 Python 컴파일 검사도 실행합니다.

## CSV 형식

`text` 컬럼은 필수이고 `id` 컬럼은 선택 사항입니다.

```csv
id,text
1,"청년들이 이용할 수 있는 취업 지원 정보가 한곳에 모이면 좋겠습니다."
2,"지역 청년을 위한 상담 시간이 더 다양했으면 합니다."
```

분석 전에 다음 정제가 수행됩니다.

- text 앞뒤 공백 제거
- null text 제거
- 빈 문자열 제거
- 중복 text 제거
- index 재설정
- id가 없으면 자동 생성

## 의존성 관리

Streamlit Cloud와 Docker에서 같은 `requirements.txt`를 사용합니다.

- Streamlit `1.62.0`
- Python 3.12 호환 버전 고정
- NumPy, pandas, scikit-learn, Plotly 고정
- SentenceTransformers 및 Transformers 고정
- UMAP 선택 기능 포함

Streamlit Cloud에서 의존성 문제가 발생하면 앱 설정의 Python version이 `3.12`인지, 저장소 루트의 `requirements.txt`와 `streamlit_app.py`를 선택했는지 먼저 확인하세요.

## 프로젝트 구조

```text
.
├─ streamlit_app.py                   # Streamlit Cloud 진입 파일
├─ analysis.py                        # 임베딩·클러스터·검색·감성·LLM 함수
├─ requirements.txt                   # Streamlit Cloud/Docker 공통 의존성
├─ .streamlit/config.toml             # Streamlit 테마와 서버 설정
├─ Dockerfile                         # Streamlit용 재현 가능한 이미지
├─ docker-compose.yml                 # 로컬 컨테이너 실행 설정
├─ ai_insight_engine_youth_comments.csv # 기본 CSV
├─ sample-data.csv                    # 테스트용 샘플 CSV
├─ docs/                              # 이전 정적 버전 보관본
└─ LICENSE
```

## 보안 주의사항

- API key를 Python 코드, CSV, README, GitHub Actions 파일에 입력하지 않습니다.
- Streamlit Cloud에서는 `Settings → Secrets`를 사용합니다.
- 앱 화면에 직접 입력한 key는 현재 세션에서 LLM 요청에 사용됩니다.
- 노출이 의심되는 key는 즉시 API 제공자 대시보드에서 폐기하고 새로 발급합니다.

## 라이선스

자세한 내용은 [LICENSE](./LICENSE) 파일을 확인하세요.
