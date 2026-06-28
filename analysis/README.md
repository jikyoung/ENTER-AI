# Analysis

이 디렉토리는 ENTER-AI를 Review Signal Agent 방향으로 다시 잡는 과정에서
사용한 데이터 품질 점검과 V0 제품 검증 스크립트를 모아둔 곳입니다.

`analysis/out/` 아래 생성 결과는 로컬 실험 데이터나 큰 CSV를 포함할 수 있어
Git에는 올리지 않습니다. 중요한 발견과 의사결정은 아래 문서에 정리했습니다.

- [`docs/project_history.md`](../docs/project_history.md)
- [`docs/product_direction.md`](../docs/product_direction.md)
- [`docs/review_signal_agent_spec.md`](../docs/review_signal_agent_spec.md)

## 주요 스크립트

- `day0_audit.py`: URL 고유성, 키워드 hit율, 짧은 본문 비율, 출처 분포,
  노이즈 카테고리를 점검합니다.
- `check_url_uniqueness.py`: 크롤링 데이터가 근거 인용에 쓸 수 있는 수준인지
  확인합니다.
- `check_googleplay_metadata.py`: Google Play 리뷰 메타데이터를 확인합니다.
- `review_action_demo.py`: LLM 호출 없이 Google Play 리뷰만으로 이슈 신호를
  뽑을 수 있는지 검증한 V0 데모입니다.
