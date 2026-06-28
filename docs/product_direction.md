# Review Signal Agent 제품 방향

## 한 줄 정의

리뷰는 AI가 읽고, 서비스팀은 중요한 카드만 본다.

## 문제

앱 리뷰와 고객 피드백은 매일 쌓이지만, 제품팀이 바로 쓰기 어렵다.

- 짧은 칭찬, 감정적 불만, 무관한 경험, 중복 리뷰가 섞여 있다.
- 담당자가 모든 리뷰를 읽고 진짜 문제를 찾기에는 시간이 부족하다.
- 기존 리뷰 분석 도구는 차트와 요약을 보여주는 데서 멈추는 경우가 많다.
- 제품팀이 필요한 것은 “전체 리뷰 요약”보다 “오늘 처리할 이슈”다.

## 만들고 싶은 제품

Review Signal Agent는 고객 피드백을 실행 가능한 제품 이슈로 바꾸는
ReviewOps Agent다.

입력:

- Google Play / App Store 리뷰
- 커뮤니티 글
- CS 문의
- VOC CSV

출력:

- Issue Card
- What's Working Card
- Quality Report
- 판단 보류 / refusal summary
- Slack, Notion, Jira로 넘길 수 있는 작업 단위

## 핵심 차별점

### 1. 요약보다 액션

리뷰를 잘 요약하는 것보다, 팀이 바로 움직일 수 있는 카드로 바꾸는 데 집중한다.

예:

```yaml
title: 본인인증 실패 반복
severity: High
confidence: 0.87
evidence_count: 42
owner: App / Backend / CX
recommended_actions:
  - 최근 버전의 인증 플로우 변경사항 확인
  - 특정 OS/버전 재현 테스트
  - 고객센터 답변 템플릿 업데이트
```

### 2. confidence와 refusal

근거가 부족하면 그럴듯한 카드를 만들지 않는다.

- 리뷰 수가 너무 적으면 판단 보류
- 클러스터 응집도가 낮으면 confidence 하향
- 평점과 본문이 불일치하면 낮은 가중치
- 근거 review_id가 없는 주장은 생성하지 않음

### 3. AI를 역할별로 나눠 사용

하나의 LLM 호출로 끝내지 않고, 역할을 나눈다.

- Signal Filter: 분석할 가치가 있는 리뷰만 남김
- Issue Grouper: 반복 이슈를 묶음
- Card Writer: 사람이 볼 수 있는 카드 생성
- Critic: 근거, 숫자, 추측 여부 검증
- Delivery Agent: 업무 도구로 전달

## MVP 범위

1. Google Play 앱 검색과 리뷰 수집
2. 리뷰 품질 진단
3. 노이즈 필터링
4. 반복 이슈 그룹핑
5. Issue Card 5개 생성
6. 긍정 신호 카드 2개 생성
7. Quality Report 생성
8. Markdown / JSON export

## 아직 검증해야 할 것

- PM/CX/QA 담당자가 실제로 이슈 카드를 유용하다고 느끼는가
- 사람이 읽은 리뷰와 에이전트가 만든 카드의 precision이 충분한가
- confidence가 낮은 카드를 거절하는 기준이 맞는가
- Slack/Notion/Jira 전달이 실제 업무 시간을 줄이는가
- 앱 리뷰 외 CS/VOC 데이터까지 확장할 때 품질이 유지되는가

## 현재 ENTER-AI에서 가져갈 자산

- LangGraph 멀티에이전트 구조
- FAISS 기반 저장과 검색
- KMeans 기반 토픽 클러스터링
- 비동기 LLM 처리 최적화
- 크롤링 파이프라인
- 테스트와 벤치마크 습관
- 데이터 품질을 먼저 점검하고 방향을 바꾼 의사결정 기록
