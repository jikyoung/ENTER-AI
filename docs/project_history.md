# ENTER-AI 프로젝트 활동 히스토리

> 2023-12부터 2026-05-26까지의 전체 작업 정리. Phase 1(팀) → Phase 2(솔로 리팩토링) → Phase 3(시나리오 재정의).

총 **104개 커밋, 5개 PR, 1개 진행 중 feature 브랜치**.

---

## Phase 1 — 팀 프로젝트 (2023-12-31 ~ 2024-01-12)

KT 에이블스쿨 7인 팀(AIVLE-ENTER) 협업 시기. main 브랜치에 직접 푸시 + 머지 다수.

### 구축 항목

| 영역 | 내용 |
|---|---|
| 크롤러 | 4개 사이트 spider (clien, quesarzone, MiniGigiKorea, google_play) + Splash Lua |
| 크롤링 파이프라인 | `crawl_pipeline.py` — Docker 기동 + CSV 병합 |
| LLM 모델 추상화 | `SetTemplate` — 유저별 configs.yaml 로드 / 동적 모델 변경 |
| 모델 시도 | Llama-2-13B-Chat-GPTQ → Mistral-7B-Instruct-v0.2 → ChatOpenAI 최종 |
| VectorDB / RAG | FAISS + OpenAIEmbeddings + MultiQueryRetriever |
| PDF 보고서 | ReportLab + 한국어 폰트 + Mermaid 다이어그램 |
| API 서버 | FastAPI `/report` 엔드포인트 + start_crawl 로직 |

### 주요 버그 픽스
- `clien.py` spider 버그 (두 번 패치)
- `crawl_pipeline.py merge_csv_files` 버그
- get_crawl_data 예외 처리
- datetime 포맷 (파일경로 특수문자 문제)

### 결과
한국어 보고서 생성기 동작은 함. 다만 시나리오 모호, 데이터 품질 미검증, 보고서 정성 가치 낮음.

---

## Phase 2 — 솔로 리팩토링 (2026-03-20 ~ 2026-03-25)

가장 굵직한 작업 시기. 6일간 PR 5개 + LangGraph 도입. 전부 Claude Sonnet 4.6과 페어 프로그래밍.

### 3월 20일: 성능 개선 (main 직접)

#### `44f0e70` Refactor: LLM 파이프라인 성능 개선
- ChatOpenAI 교체, LLMChain 싱글톤, async 추가
- 결과: **LLM 필터링 ~20분 → ~2분 (10.7배)**

#### `c387b1f` Refactor: 크롤러 병렬화
- subprocess.Popen 병렬화로 3개 크롤러 동시 실행
- Docker Splash 중복 실행 방지
- 결과: **크롤링 ~8분 → ~2.5분**

### 3월 23일: 벤치마크 실측

#### `2caf7ad` Benchmark
- `benchmark_crawl.py` — **2.7배 (6.9분 → 2.6분) 확정**
- `locustfile.py` 부하 테스트 (동시 5명, 에러율 0%, 평균 32.7초)

### 3월 25일: PR 5개 연속 머지 + LangGraph

#### PR #1 — feature/topic-clustering (`9c48b41`)
- **TopicPipeline** 추가: FAISS 임베딩 → KMeans → LLM 클러스터 명명 + "무관" 자동 제외
- 감성 분석(async) + 토픽 클러스터링(thread executor)을 `asyncio.gather`로 병렬
- 보고서 프롬프트 "SWOT/BCG 전략 분석" → **"커뮤니티 여론 분석"**
- scikit-learn 의존성 추가
- 후속: `5171d4b` (빈 클러스터/이상 토픽 필터링), `7d070a6` (Semaphore 50 + context 테스트)

#### PR #2 — refactor/project-structure (`32c2c0d`)
- `llm_model/` → `filter_pipeline/` 리네임 (실제 역할 반영)
- `LangchainPipline` → `FilterChain` 클래스명 변경
- 미사용 파일 삭제 (llama2_pipline.py, test.ipynb)
- 테스트 파일을 `tests/`로 통합
- 테스트 유저 데이터 삭제 (asdf1234, pig1234, star1234)

#### PR #3 — refactor/cleanup-root (`32bcbf8`)
- `temp/` 삭제 (개발 초기 임시 파일, 노트북 11개)
- 루트 `__pycache__/` 삭제
- `benchmark_*.py`, `locustfile.py` → `tests/`

#### PR #4 — feature/incremental-crawling (`b6c0904`)
- `VectorPipeline.get_existing_urls()` — 기존 FAISS 문서 url 집합 추출
- `VectorPipeline.merge_into_store()` — url 기준 중복 제거 후 병합
- start_crawl: `embedding_and_store` → `merge_into_store`
- 응답에 추가된 문서 수(`added`) 반환

#### PR #5 — feature/dynamic-filter (`4c0a87b`)
- `FilterChain`: keyword 파라미터화 (**KT 하드코딩 제거**)
  - "중고거래/판매글/스포츠/단순언급" 명시적으로 no 처리
- `boardcategory/documentcategory` 규칙 기반 사전 필터 (LLM 호출 전)
- 감성 분석: keyword 맥락 기반 분류
- 토픽 클러스터링: keyword 맥락 네이밍 + "무관" 자동 제거
- 결과: **9개 토픽 → 5개**

#### 현재 HEAD — feature/langgraph-multi-agent (`ec42301`) ⭐
**아직 main 머지 안 됨**. LangGraph Multi-Agent 도입.

- `ReportState` (TypedDict) — keyword/sentiment/topics/context/insight/draft/critique/iterations/final_report 누적
- **SentimentAgent** (gpt-4o-mini) — 감성 수치 → 인사이트
- **TopicAgent** (gpt-4o-mini) — 토픽 클러스터 → 이슈 + 우선순위
- **WriterAgent** (gpt-4o) — 인사이트 + FAISS context → 보고서 초안
- **CriticAgent** (gpt-4o-mini) — 체크리스트 → PASS or RETRY (최대 2회)
- Conditional edge: `should_retry` (final_report 있으면 done, 아니면 retry → writer)
- `apps.py` `/report` 엔드포인트를 ReportAgent로 교체, PDF 변환은 `to_pdf()` 재활용

### Phase 2 종합 성과

| 항목 | Before | After | 배수 |
|---|---|---|---|
| LLM 필터링 | ~20분 (동기) | ~2분 (asyncio.gather) | **10.7배** |
| RAG 첫 청크 응답 | 17.76초 | 10.09초 | **43% 단축** |
| 크롤러 병렬화 | 6.9분 | 2.6분 (Popen) | **2.7배** |
| 토픽 클러스터 정제 | 9개 (노이즈) | 5개 (정제) | "무관" 제거 |
| 부하테스트 | - | 에러율 0%, 평균 32.7초 (동시 5명) | - |
| 데이터 수집 | - | 4사이트 약 3만 건 | - |
| 보고서 생성 구조 | Chain (단일 호출) | **LangGraph Multi-Agent (4노드 + retry)** | 구조 진화 |

---

## Phase 3 — 시나리오 재정의 (2026-05-26, 하루)

Phase 2까지 만든 시스템을 들고 **"이게 진짜 portfolio-grade인가?"** 자문에서 시작.

### 1) 데이터 품질 실측 (`analysis/day0_audit.py`)

**핵심 발견**:
- KT 760건 → unique URL **1개** (검색 URL 100%)
- SKT 5,154건 → unique URL **4개** (사이트별 1개씩)
- SKT의 **67%가 google_play 리뷰** (3,478건)
- SKT 키워드 hit율 23.5%, KT 65.4% (출처 차이로 인한 비대칭)
- KT_v1 = KT_v2 동일 데이터 (중복 저장)

**결론**: 게시글 단위 citation 불가능, 커뮤니티 데이터의 정체성이 사실상 "구글플레이 리뷰 + 사이트별 검색 결과"

### 2) 자산 점검

| 자산 | 결과 |
|---|---|
| TopicPipeline (KMeans) | ✅ 그대로 재활용 |
| FilterChain (yes/no) | ✅ 5+ 카테고리로 확장 |
| Sentiment 배치 처리 | ✅ 재활용 |
| LangGraph 골격 | ✅ 노드 재구성하여 재활용 |
| GooglePlay 수집 | ✅ reviewId/score/appVersion 이미 추출 가능 |
| 크롤러 URL 버그 | ⚠️ `clien.py:127` 1줄 수정으로 해결 가능 |
| PDF 보고서 (ReportLab) | ❌ Insight Card로 교체 |
| MultiQueryRetriever | ❌ 앱 리뷰는 RAG 불필요 |
| 커뮤니티 크롤러 (clien/quesarzone/MiniGigi) | ❌ v1 분석 대상 제외 |

### 3) 시장 분석

UPUP, AppFollow, Appbot, Play Console, Mobile Action, Sensor Tower 포지셔닝 분석 → **"리뷰 분석 SaaS는 포화"** 결론.

차별화 축을 "리뷰 보여주기"가 아니라 **"의사결정으로 변환"**으로 잡음.

### 4) 사용자 자체 검증

본인이 직접 작성한 스크립트로 데이터 재확인:
- `analysis/check_url_uniqueness.py` — URL 진단
- `analysis/check_googleplay_metadata.py` — google_play 메타데이터
- 아이폰17 키워드 1년치 커뮤니티 데이터 → 품질 부족 확정
- 결론: 커뮤니티 데이터는 v1에서 제외

### 5) V0 MVP 데모 (`analysis/review_action_demo.py`)

배민 vs 쿠팡이츠 500건씩 분석. **LLM 0번 호출**.

**검증된 시그널**:

| 항목 | 배민 | 쿠팡이츠 |
|---|---|---|
| 평균 평점 | 3.64 | 4.59 |
| 1-2점 비율 | **32.2%** | 8.2% |
| food_quality 비율 | 27.2% | 4.8% |
| ads_popup | **3.4%** | 0% |
| short review (<15자) | 42.6% | **70.0%** |

→ **"통계/규칙 1차 + LLM은 대표만"** 전략의 실효성 검증 완료.

### 6) 시나리오 토론 라운드

| 라운드 | 결정 |
|---|---|
| 1차 | 커뮤니티 데이터 vs google_play 선택 — google_play 우세 |
| 2차 | 통신 3사 비교? → 다양한 5개 앱? → 결국 도메인 안 박음 |
| 3차 | Release Feedback Agent (버전 회귀 탐지 중심) 검토 |
| 4차 | Actionable Review Intelligence Agent로 확장 (UPUP 분석 후) |
| 5차 | **Review Signal Agent로 최종 확정** + ReviewOps Agent 부제 |
| 6차 | Slack/Email/Notion/Card News를 **v2가 아닌 처음부터 아키텍처에 포함** |
| 7차 | 도메인 박지 않고 **Universal Issue Taxonomy** 우선 |
| 8차 | 비교는 **보조 기능**으로 강등 |

### Phase 3 산출물

```
analysis/
├── day0_audit.py                  # 데이터 진단 스크립트
├── check_url_uniqueness.py        # URL 진단 (사용자 작성)
├── check_googleplay_metadata.py   # GP 메타데이터 확인 (사용자 작성)
├── review_action_demo.py          # V0 MVP 데모
└── out/
    ├── audit_summary.json
    ├── clien_url_test.csv
    ├── iphone17_raw/              # 1년치 검증
    ├── label_sample_KT_v1.csv
    ├── label_sample_KT_v2.csv
    ├── label_sample_SKT.csv
    └── review_action_demo/
        ├── reviews_raw.csv         # 1000건
        └── summary.json            # 비교 결과
```

---

## 전체 흐름 한 줄

> 팀에서 4사이트 커뮤니티 RAG 보고서 시스템을 구축 → 솔로 리팩토링으로 성능 10배 + LangGraph Multi-Agent 도입 → 데이터 품질 실측 후 "리뷰 분석 도구"에서 "리뷰 운영 인텔리전스 플랫폼(ReviewOps Agent)"으로 시나리오 전체 재정의.

---

## 핵심 의사결정 로그

| 결정 | Why |
|---|---|
| Llama2 → ChatOpenAI | 한국어 품질 + 응답속도 + 안정성 |
| ReportLab PDF (당시) | 팀 산출물 요구사항 |
| TopicPipeline KMeans 도입 | 보고서 프롬프트 정확도 ↑, 데이터 기반 토픽 |
| LangGraph 전환 | 단일 Chain → 비판/재작성 가능한 multi-agent |
| 동적 필터 (keyword 파라미터화) | KT 하드코딩 제거, 어떤 앱/주제든 분석 가능 |
| 커뮤니티 데이터 v1 폐기 (Phase 3) | citation 불가, 노이즈 70%+, 사용자 직접 검증 |
| Review Signal Agent로 재정의 | UPUP/AppFollow와 차별화 가능한 유일한 축 |
| Slack/Email/Notion 처음부터 포함 | "분석 도구가 아니라 운영 플랫폼" 포지셔닝 |
| Universal Taxonomy 우선 | 도메인 박으면 확장성 X |
| 비교는 보조 기능 | 단일 앱 분석 흐름 우선 |
