"""
SKT 데이터 LLM 필터링 벤치마크 — async only
sync 결과는 이전 실행에서 이미 확보:
  총 5073.39초 (84.56분), 건당 0.992초, p50 0.822초, p95 1.903초
"""
import sys
import json
import time
import asyncio
import statistics
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

project_root = Path(__file__).parent
project_dir = project_root.parent / 'project'
sys.path.insert(0, str(project_dir))

load_dotenv(project_root.parent / '.env')

from server.modules.set_template import SetTemplate

# ── 설정 ─────────────────────────────────────────────
USER_ID = 'user01'
CSV_PATH = project_dir / 'user_data' / 'user01' / 'crawl_data' / 'SKT' / '2026-04-07T23h10m48s' / 'merged_data.csv'
TARGET_COL = 'document'
CONCURRENCY = 50
RESULT_PATH = project_root / f'benchmark_skt_async_{datetime.now().strftime("%Y%m%dT%H%M%S")}.json'

# 이전 sync 측정 결과 (5113건 전수 실측)
SYNC_PRIOR = {
    "total_sec": 5073.39,
    "total_min": 84.56,
    "mean_sec": 0.992,
    "p50_sec": 0.822,
    "p95_sec": 1.903,
}

# ── 프롬프트/모델 로드 ───────────────────────────────
template_obj = SetTemplate(USER_ID)
prompt_str = template_obj.load_template('llama', 'crawl')
model_name = template_obj.load('chatgpt', 'params').model

llm = ChatOpenAI(model=model_name, temperature=0)
prompt = PromptTemplate(input_variables=["user_input"], template=prompt_str)
chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

# ── 데이터 로드 ──────────────────────────────────────
df = pd.read_csv(CSV_PATH).dropna(subset=[TARGET_COL])
df = df[df[TARGET_COL].str.strip() != '']
documents = df[TARGET_COL].tolist()
TOTAL = len(documents)

print(f"\n{'=' * 60}")
print(f"  SKT async-only 벤치마크 (sync 결과는 사전 실측 사용)")
print(f"{'=' * 60}")
print(f"  모델        : {model_name}")
print(f"  전체 행 수  : {TOTAL:,}건")
print(f"  동시성      : Semaphore({CONCURRENCY})")
print(f"  시작 시각   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 60}\n")


async def run_async():
    print(f"[ ASYNC ] 시작...")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    latencies = []
    counter = {'done': 0}

    async def call_one(text: str):
        async with semaphore:
            t0 = time.time()
            try:
                _ = await chain.apredict(user_input=text)
            except Exception:
                counter['done'] += 1
                return
            elapsed = time.time() - t0
            latencies.append(elapsed)
            counter['done'] += 1
            if counter['done'] % 500 == 0 or counter['done'] == TOTAL:
                print(f"  [{counter['done']:5d}/{TOTAL}]")

    start = time.time()
    await asyncio.gather(*[call_one(t) for t in documents])
    total = time.time() - start

    mean = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18]

    print(f"\n  ▶ 총 소요    : {total:.2f}초 ({total / 60:.2f}분)")
    print(f"  ▶ 건당 평균  : {mean:.3f}초")
    print(f"  ▶ p50        : {p50:.3f}초")
    print(f"  ▶ p95        : {p95:.3f}초\n")

    return total, mean, p50, p95


total, mean, p50, p95 = asyncio.run(run_async())

speedup = SYNC_PRIOR["total_sec"] / total
saved_min = (SYNC_PRIOR["total_sec"] - total) / 60

print(f"{'=' * 60}")
print(f"  최종 비교 — SKT {TOTAL:,}건 전수 실측")
print(f"{'=' * 60}")
print(f"  {'':18s} {'BEFORE(sync)':>15s}   {'AFTER(async)':>15s}")
print(f"  {'전체 처리시간':18s} {SYNC_PRIOR['total_min']:>14.2f}분  {total / 60:>14.2f}분")
print(f"  {'건당 평균':18s} {SYNC_PRIOR['mean_sec']:>14.3f}초  {mean:>14.3f}초")
print(f"  {'p50':18s} {SYNC_PRIOR['p50_sec']:>14.3f}초  {p50:>14.3f}초")
print(f"  {'p95':18s} {SYNC_PRIOR['p95_sec']:>14.3f}초  {p95:>14.3f}초")
print(f"  {'속도 향상':18s} {'':>15s}   {speedup:>13.2f}배")
print(f"  {'절약 시간':18s} {'':>15s}   {saved_min:>12.2f}분")
print(f"{'=' * 60}")

result = {
    "dataset": "SKT",
    "model": model_name,
    "total_rows": TOTAL,
    "concurrency": CONCURRENCY,
    "measured_at": datetime.now().isoformat(),
    "before_sync": SYNC_PRIOR,
    "after_async": {
        "total_sec": total,
        "total_min": total / 60,
        "mean_sec": mean,
        "p50_sec": p50,
        "p95_sec": p95,
    },
    "speedup": speedup,
    "saved_min": saved_min,
}
RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(f"\n결과 저장: {RESULT_PATH}")
