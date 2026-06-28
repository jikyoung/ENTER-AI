# Tests and Benchmarks

핵심 회귀 테스트는 아래 명령으로 실행합니다.

```bash
uv run --extra dev python -m pytest
```

현재 로컬 검증 결과:

- 33개 테스트 수집
- 33개 통과

## 구조

- `unit/`: FAISS, 필터링, 토픽 클러스터링, 감성 분석, LangGraph 리포트
  에이전트 동작을 검증합니다.
- `integration/`: 외부 의존성을 mock 처리한 FastAPI 엔드포인트 테스트입니다.
- `benchmark_*.py`: 처리량 개선을 측정하기 위해 사용한 선택 실행 스크립트입니다.
  로컬 실험 데이터와 API 키가 필요합니다.
- `locustfile.py`: 부하 테스트 시나리오입니다.

생성된 벤치마크 로그와 JSON 결과는 Git에 올리지 않습니다.
