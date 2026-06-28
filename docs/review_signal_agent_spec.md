# Review Signal Agent — 제품 명세

## 1. Product Definition

**한 줄 정의**
> 리뷰는 AI가 읽고, 서비스팀은 카드만 봅니다.

**제품 한 줄**
> Review Signal Agent는 앱 리뷰에서 중요한 사용자 신호를 감지하고, 팀의 업무 도구로 자동 전달하는 ReviewOps Agent다.

**핵심 가치 (왜 다른가)**
- ❌ 리뷰 데이터를 보여주는 서비스 (AppFollow, UPUP)
- ✅ 리뷰에서 **실행할 일을 만들어주는** 서비스

**한 줄 슬로건**
> "리뷰는 AI가 읽습니다. PM은 카드만 봅니다."

---

## 2. 문제 정의

- 앱 리뷰는 매일 쌓이지만 서비스팀이 바로 활용하기 어려움
- 짧은 칭찬, 감정적 불만, 앱 무관 경험, 중복 리뷰가 섞여 있음
- 담당자가 모든 리뷰를 읽고 진짜 문제를 찾기에는 시간 부족
- 기존 도구(AppFollow 등)는 데이터를 보여주는 데서 멈춤 — 의사결정/액션 단계로 연결되지 않음

---

## 3. 핵심 사용자

- PM (제품 우선순위 결정)
- CX 담당자 (고객 응대 / 정책 개선)
- QA 담당자 (회귀 이슈 발견)
- 앱 운영 담당자 (장애 대응)
- 서비스 기획자 (개선 백로그 발굴)

---

## 4. 5-Layer Service Architecture

```
┌─────────────────────────────────────────────────────────┐
│  1. Ingestion Layer                                      │
│     - Google Play Collector                              │
│     - App Store Connect Collector (v2)                   │
│     - CSV Upload Collector                               │
│     - Community / CS 연동 (v3)                            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  2. Intelligence Layer                                   │
│     - Quality Gate (데이터 충분성 / 노이즈 진단)              │
│     - Signal Filter (useful / noise / positive 분류)      │
│     - Issue Engine (embedding + clustering + merging)    │
│     - Confidence Scorer                                  │
│     - Refusal / Trend Detection                          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  3. Knowledge Layer                                      │
│     - Universal Issue Taxonomy                           │
│     - Action Template Library                            │
│     - Team / Owner Mapping                               │
│     - 과거 이슈 기록 (history)                              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  4. Delivery Layer                                       │
│     - Web Dashboard (Streamlit)                          │
│     - Slack 알림                                          │
│     - Email 주간 브리핑                                    │
│     - Notion 티켓 자동 생성                                 │
│     - Card News (외부 공유용)                              │
│     - Markdown / JSON Export                             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  5. Feedback Layer                                       │
│     - 카드 유용성 평가 (useful / not useful)                 │
│     - 이슈 병합 / 분리 수동 조정                              │
│     - 담당자 배정                                          │
│     - resolved 여부 추적                                  │
│     - 다음 분석에 반영 (learning)                           │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Universal Issue Taxonomy

도메인 먼저 박지 않고, **공통 이슈 구조 우선**. 앱 맥락은 보조 정보.

```yaml
universal_taxonomy:
  - login_auth              # 로그인 / 인증 / 본인확인
  - payment_subscription    # 결제 / 구독 / 자동결제
  - refund_cancellation     # 환불 / 취소
  - customer_support        # 고객지원 / CS 응대
  - app_error_performance   # 앱 오류 / 성능 / 크래시
  - ui_ux                   # UI / UX / 사용성
  - notification_ads        # 알림 / 광고 / 푸시
  - account_security        # 계정 / 보안
  - price_benefit           # 가격 / 혜택 / 쿠폰
  - content_quality         # 콘텐츠 / 상품 / 서비스 품질
  - delivery_processing     # 배송 / 처리 지연
  - policy_terms            # 정책 / 약관 불만
  - positive_experience     # 긍정 경험 (What's Working)
  - feature_request         # 기능 요청
  - other_uncategorized     # 기타 / 판단 보류
```

### 앱 맥락 보조 태그 (예시)

```yaml
toss:
  context_tags: [송금, 한도, 계좌연결]
baemin:
  context_tags: [배달, 라이더, 쿠폰, 음식점]
kakao:
  context_tags: [채팅, 친구, 톡서랍, 이모티콘]
```

### 3단계 구조
1. Universal Issue Layer (공통)
2. App Context Layer (자동 감지 or 사용자 선택)
3. Action Template Layer (이슈 + 맥락 조합)

---

## 6. MVP 범위 (Phase 1)

### 포함 ✅
- Google Play 앱 검색 + 후보 선택
- 앱 등록 (단일 또는 + 경쟁 앱)
- 최근 500~1000개 리뷰 수집
- Quality Gate
- Signal Filter (Universal Taxonomy 기반)
- Issue Grouping (KMeans + LLM Semantic Merging)
- Confidence Scoring
- Refusal 로직
- Insight Card 생성 (5 부정 + 2 긍정)
- Quality Report 생성
- Streamlit Signal Board
- Markdown / JSON Export

### v1에서 제외 (단, 아키텍처에는 포함) ⏳
- Slack 알림 → Phase 2
- Email 주간 브리핑 → Phase 2
- Notion 티켓 자동 생성 → Phase 2
- Card News 생성 → Phase 2
- App Store Connect → Phase 4
- 실시간 cron 모니터링 → Phase 3
- 다국어 지원 → Phase 4
- 풀 Next.js 대시보드 → Phase 4

---

## 7. Implementation Phases

### Phase 1: Service Core (~2주)
- 앱 검색/등록
- 리뷰 수집 + 저장
- Quality Gate
- Signal Filter
- Issue Engine (clustering + merging)
- Insight Card 생성
- Streamlit 대시보드
- Evaluation 측정

### Phase 2: Delivery (~1-2주)
- Slack webhook 알림
- Email 주간 브리핑 (SMTP)
- Notion 티켓 자동 생성 (Notion API)
- Markdown export 완성

### Phase 3: Operations (~2주)
- Scheduler (cron / APScheduler)
- 주간/월간 리포트
- 이슈 상태 추적 (open / resolved)
- 담당자 매핑 / 알림
- 카드 유용성 피드백 수집

### Phase 4: Scale (~지속)
- App Store Connect 연동
- 다도메인 taxonomy 확장
- 조직별 workspace
- 팀별 권한 시스템
- Feedback Learning

---

## 8. Insight Card 구조

### Issue Card
```yaml
issue_id: ISS-2026-0524-001
title: 본인인증 / 로그인 실패 반복 발생
severity: High
priority: P1
confidence: 0.87

evidence:
  count: 42
  avg_rating: 1.4
  recent_trend: "최근 7일 증가"
  representative_reviews:
    - review_id: "abc123"
      score: 1
      content: "인증에서 계속 멈춥니다"
      date: "2026-05-23"
    - review_id: "def456"
    - review_id: "ghi789"

categories:
  primary: login_auth
  context_tags: [본인인증, 무한로딩]

keywords: ["인증", "로그인", "본인확인", "무한로딩"]

impact:
  description: "가입/결제/서비스 진입 차단 가능성"
  affected_versions: ["6.1.7", "6.1.6"]  # 가능 시
  market_context: "경쟁 앱 대비 우리 앱에서 더 자주 언급됨"  # 선택

owner:
  primary: App
  secondary: [Backend, CX]

recommended_actions:
  - "v6.1.7 인증 플로우 변경사항 확인"
  - "특정 OS/버전 재현 테스트"
  - "고객센터 답변 템플릿 업데이트"

decision_basis: "..."
```

### What's Working Card
```yaml
- 사용자가 긍정적으로 언급한 기능
- 대표 리뷰 3개
- 반복 빈도
- 유지/강화해야 할 강점
```

### Quality Report
```yaml
- 전체 리뷰 수
- 분석 대상 리뷰 수
- 제외 리뷰 수 + 제외 사유 분포
- 데이터 충분성 등급 (충분 / 보통 / 부족)
- 판단 보류 항목
- 데이터 한계 명시
```

### 카드 총 9장 구성 (MVP)
- Issue Card 5장 (부정)
- What's Working Card 2장 (긍정)
- Quality Report 1장
- Refusal Summary 1장 (판단 보류 목록)

---

## 9. Confidence 공식

```python
confidence = (
    0.30 * evidence_count_score    # 정규화된 리뷰 수 (max=50)
  + 0.20 * recency_score           # 최근성 (exp(-days/30))
  + 0.20 * cluster_purity          # 클러스터 응집도 (silhouette)
  + 0.15 * helpful_signal          # thumbs_up 평균 z-score
  + 0.10 * rating_consistency      # 평점-내용 일치도
  + 0.05 * version_signal          # 옵셔널 boost (있을 때)
)

threshold_refuse: < 0.5  → 카드 미생성
threshold_show:   0.5~0.75
threshold_strong: ≥ 0.75
```

---

## 10. Action Template Library (환각 차단)

```yaml
# action_templates.yaml
login_auth:
  - "{{auth_method}} API 실패 로그 확인 (최근 7일)"
  - "특정 OS/버전 재현 테스트: {{top_versions}}"
  - "고객센터 답변 템플릿 업데이트"
  - "App + Backend 협의 필요"

payment_subscription:
  - "결제 게이트웨이 응답 코드 분석"
  - "{{payment_method}} 통합 테스트"
  - "구독 갱신 알림 UX 점검"

# ... (15개 카테고리 × 3-5개 액션)
```

→ LLM은 **변수만 채움**, 액션 문장 자체는 generate 안 함. 환각 차단.

---

## 11. 비교 기능 (보조)

**우선순위는 단일 앱 중심.** 비교는 카드 내 부속 섹션 또는 사용자 옵션.

### 카드 내 Market Context 섹션
```
Market Context (선택):
- 동일 카테고리 앱에서도 유사 이슈 발견
- 경쟁 앱 대비 우리 앱에서 더 자주 언급됨
- 경쟁 앱에서는 거의 발견되지 않음
```

### "+ 경쟁 앱 추가" 버튼
사용자가 명시적으로 비교 모드 활성화 시에만 작동.

### 비교 타입 2가지
- **Competitive comparison**: 동일 시장 (토스 vs 카카오뱅크) → 경쟁 비교
- **Operational benchmark**: 다른 시장 (토스 vs 카카오톡) → 운영 품질 벤치마크

---

## 12. 필터링 기준 (Signal Filter)

### 제외 또는 낮은 가중치
- "좋아요", "굿", "최고" 같은 초단문
- 구체성 없는 긍정 리뷰
- 앱 기능과 무관한 개인 경험
- 욕설 또는 감정 표현 중심 리뷰
- 중복 리뷰 (cosine similarity > 0.9)
- 너무 오래된 리뷰
- 별점과 본문이 불일치하는 리뷰

### 우선순위 높임
- 1-2점 리뷰
- 구체적 상황 설명
- 로그인/인증/결제/송금/고객센터/오류 언급
- helpful count 높은 리뷰
- 최근 반복되는 이슈
- 여러 리뷰에서 같은 표현 반복

### 투명성: 제외 사유 공개
```
500개 리뷰 중
- 182개: 분석 대상
- 144개: 짧은 긍정 리뷰 제외
- 71개: 앱 기능과 무관한 개인 경험 제외
- 53개: 중복/유사 리뷰로 병합
- 50개: 감정 표현 중심이라 낮은 가중치
```

---

## 13. Delivery Channels

| 채널 | 용도 | 구현 phase |
|---|---|---|
| Dashboard (Streamlit) | 전체 현황 / 상세 / 근거 확인 | Phase 1 |
| Slack | 긴급 이슈 즉시 알림, 담당 채널 자동 공유 | Phase 2 |
| Email | 주간 리뷰 브리핑, 임원 리더 공유용 | Phase 2 |
| Notion | 이슈 카드 → 실행 티켓 (담당자 / 상태 / 우선순위) | Phase 2 |
| Card News | 외부 공유 / 비개발 직군 보고용 | Phase 2 |
| Markdown / JSON Export | 백업 / 외부 도구 연동 | Phase 1 |

### 알림 정책
- 기본값: **주간 + 이상 신호 즉시**
- 즉시 알림 조건:
  - 1-2점 리뷰 평소 대비 급증
  - 같은 이슈 N건 이상 반복
  - helpful count 높은 부정 리뷰 등장
  - 특정 버전 업데이트 이후 부정 리뷰 증가
  - 결제 / 인증 / 고객센터 치명 이슈

---

## 14. Evaluation 지표

| 지표 | 측정 방법 | 목표 |
|---|---|---|
| Issue grouping accuracy | 골든셋 50건 vs 클러스터 결과 (ARI) | ≥ 0.7 |
| Card precision | 생성된 카드 10장 중 진짜 이슈 비율 (사람 검증) | ≥ 80% |
| Refusal accuracy | confidence<0.5 카드 중 실제 데이터 부족 비율 | ≥ 80% |
| Citation accuracy | 카드의 근거 review_id가 실제 그 이슈를 말하는가 | ≥ 90% |

---

## 15. 기술 스택

| 영역 | 도구 | 이유 |
|---|---|---|
| 워크플로우 | LangGraph | 기존 자산, conditional routing |
| 분류 / 머징 LLM | gpt-4o-mini | 비용 우선, 작업 단순 |
| 대화 (Ask AI) | gpt-4o | 사용자 직접 대화, 품질 중요 |
| 임베딩 | text-embedding-3-small | 최신 / 경제적 |
| 클러스터링 | sklearn KMeans | TopicPipeline 재활용 |
| 데이터 수집 | google-play-scraper | 기존 자산, 검증됨 |
| UI | Streamlit | MVP 속도 우선 |
| 저장 | SQLite (Phase 1) → PostgreSQL (Phase 3) | 단계적 확장 |

---

## 16. 디렉토리 구조 (점진적 이전)

```
ENTER-MAIN/ENTER-AI/
├── project/              # 기존 (그대로, archive)
│   └── server/modules/
├── crawler/              # 기존 (google_crawl만 활용)
├── analysis/             # V0 검증 자산 (보존)
│   ├── day0_audit.py
│   ├── review_action_demo.py
│   └── out/
└── signal_agent/         # NEW — 새 프로젝트
    ├── ingestion/
    │   ├── collector_googleplay.py
    │   └── collector_csv.py
    ├── intelligence/
    │   ├── quality_gate.py
    │   ├── signal_filter.py
    │   ├── issue_grouper.py
    │   ├── confidence_scorer.py
    │   └── critic.py
    ├── knowledge/
    │   ├── taxonomy.yaml
    │   ├── action_templates.yaml
    │   └── owner_mapping.yaml
    ├── delivery/
    │   ├── dashboard.py      # Streamlit
    │   ├── slack_sender.py   # Phase 2
    │   ├── email_sender.py   # Phase 2
    │   ├── notion_publisher.py  # Phase 2
    │   └── card_news.py      # Phase 2
    ├── feedback/
    │   └── feedback_store.py
    ├── workflow.py           # LangGraph 통합
    └── README.md
```

---

## 17. 데모 시나리오 (5분)

| 시간 | 액션 |
|---|---|
| 00:00 | 앱 검색 (예: "토스") |
| 00:10 | 후보 3개 표시 → 토스(viva republica) 선택 |
| 00:30 | 리뷰 500건 수집 (live progress) |
| 01:00 | Quality Report 자동 생성 → "분석 가능 / 충분도 High" |
| 01:30 | 🚨 Insight Card 등장: "본인인증 실패 반복 (confidence 0.87)" |
| 02:00 | 카드 클릭 → 근거 리뷰 3개 + 추천 액션 + 담당팀 |
| 02:30 | "+ 경쟁 앱 추가" → 카카오뱅크 등록 |
| 03:00 | 비교 모드: "인증 실패는 카카오뱅크도 동일 (시장 공통)" / "토스 고유: 송금 한도 안내 부족" |
| 04:00 | "Ask AI" 채팅: "이번 주 가장 시급한 거 알려줘" |
| 04:30 | Refusal 데모: 데이터 부족 카드 → "판단 보류" |
| 05:00 | Export → Markdown 다운로드 + (Phase 2) Notion 티켓 자동 생성 |

---

## 18. 최종 결론

> 이 프로젝트는 **리뷰 분석 도구가 아니라 리뷰 운영 인텔리전스 플랫폼이다.**
>
> 많은 리뷰를 예쁘게 요약하는 게 아니라, 서비스팀이 **오늘 봐야 할 신호만 정확하게 남기고, 팀이 일하는 도구로 자동 전달**하는 것이 목표다.
>
> 분석 엔진뿐 아니라 **전달, 티켓 생성, 피드백 루프**까지 처음부터 아키텍처에 포함시키고, 구현은 Phase 1-4로 점진적으로 한다.
