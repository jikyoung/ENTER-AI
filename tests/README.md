# Tests and Benchmarks

The main regression suite is:

```bash
uv run --extra dev python -m pytest
```

Current local verification:

- 33 tests collected
- 33 passed

## Structure

- `unit/`: isolated tests for FAISS, filtering, topic clustering, sentiment
  analysis, and LangGraph report-agent behavior.
- `integration/`: FastAPI endpoint tests with external dependencies mocked.
- `benchmark_*.py`: optional live benchmark scripts used to measure throughput
  improvements. These require local experiment data and API credentials.
- `locustfile.py`: optional load-test scenario.

Generated benchmark logs and JSON outputs are intentionally ignored.
