"""
SKT 데이터 LLM 필터링 성능 벤치마크 (옵션 A — 실제 옛날 패턴 vs 현재 패턴)
- Before: chain.predict() 동기 순차 (옛날 apps.py 방식 그대로)
- After:  chain.apredict() + asyncio.gather + Semaphore(50) (현재 방식)
- 5,113건 전수 실측, p50/p95까지 기록
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
CONCURRENCY = 50  # asyncio.Semaphore
RESULT_PATH = project_root / f'benchmark_skt_result_{datetime.now().strftime("%Y%m%dT%H%M%S")}.json'

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
print(f"  SKT 데이터 LLM 필터링 벤치마크 (전수 실측)")
print(f"{'=' * 60}")
print(f"  CSV         : {CSV_PATH.name}")
print(f"  모델        : {model_name}")
print(f"  전체 행 수  : {TOTAL:,}건")
print(f"  동시성      : Semaphore({CONCURRENCY}) [After only]")
print(f"  시작 시각   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 60}\n")


# ════════════════════════════════════════════════════
# BEFORE: 동기 순차 (옛날 apps.py 패턴 그대로)
# ════════════════════════════════════════════════════
print(f"[ BEFORE ] 동기 순차 처리 시작 ({TOTAL:,}건)...")
sync_latencies = []
sync_start = time.time()

for i, text in enumerate(documents):
    t0 = time.time()
    try:
        _ = chain.predict(user_input=text).strip()
    except Exception as e:
        print(f"  [{i + 1}/{TOTAL}] ERROR: {e}")
        continue
    elapsed = time.time() - t0
    sync_latencies.append(elapsed)
    if (i + 1) % 100 == 0 or (i + 1) == TOTAL:
        avg_so_far = sum(sync_latencies) / len(sync_latencies)
        eta_min = (avg_so_far * (TOTAL - i - 1)) / 60
        print(f"  [{i + 1:5d}/{TOTAL}] avg={avg_so_far:.2f}s | ETA={eta_min:.1f}분")

sync_total = time.time() - sync_start
sync_mean = statistics.mean(sync_latencies)
sync_p50 = statistics.median(sync_latencies)
sync_p95 = statistics.quantiles(sync_latencies, n=20)[18] if len(sync_latencies) >= 20 else max(sync_latencies)

print(f"\n  ▶ 총 소요    : {sync_total:.2f}초 ({sync_total / 60:.2f}분)")
print(f"  ▶ 건당 평균  : {sync_mean:.3f}초")
print(f"  ▶ p50        : {sync_p50:.3f}초")
print(f"  ▶ p95        : {sync_p95:.3f}초\n")


# ════════════════════════════════════════════════════
# AFTER: asyncio 병렬 + Semaphore (현재 방식)
# ════════════════════════════════════════════════════
async def run_async_benchmark():
    print(f"[ AFTER  ] asyncio 병렬 처리 시작 ({TOTAL:,}건, 동시성={CONCURRENCY})...")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    latencies = []
    counter = {'done': 0}

    async def call_one(text: str):
        async with semaphore:
            t0 = time.time()
            try:
                _ = await chain.apredict(user_input=text)
            except Exception as e:
                counter['done'] += 1
                return None
            elapsed = time.time() - t0
            latencies.append(elapsed)
            counter['done'] += 1
            if counter['done'] % 500 == 0 or counter['done'] == TOTAL:
                print(f"  [{counter['done']:5d}/{TOTAL}] in-flight done")
            return elapsed

    async_start = time.time()
    await asyncio.gather(*[call_one(t) for t in documents])
    async_total = time.time() - async_start

    async_mean = statistics.mean(latencies)
    async_p50 = statistics.median(latencies)
    async_p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)

    print(f"\n  ▶ 총 소요    : {async_total:.2f}초 ({async_total / 60:.2f}분)")
    print(f"  ▶ 건당 평균  : {async_mean:.3f}초 (실제 응답시간)")
    print(f"  ▶ p50        : {async_p50:.3f}초")
    print(f"  ▶ p95        : {async_p95:.3f}초\n")

    return async_total, async_mean, async_p50, async_p95


async_total, async_mean, async_p50, async_p95 = asyncio.run(run_async_benchmark())


# ════════════════════════════════════════════════════
# 최종 비교
# ════════════════════════════════════════════════════
speedup = sync_total / async_total
saved_min = (sync_total - async_total) / 60

print(f"{'=' * 60}")
print(f"  최종 비교 — SKT {TOTAL:,}건 전수 실측")
print(f"{'=' * 60}")
print(f"  {'':18s} {'BEFORE(sync)':>15s}   {'AFTER(async)':>15s}")
print(f"  {'전체 처리시간':18s} {sync_total / 60:>14.2f}분  {async_total / 60:>14.2f}분")
print(f"  {'건당 평균':18s} {sync_mean:>14.3f}초  {async_mean:>14.3f}초")
print(f"  {'p50':18s} {sync_p50:>14.3f}초  {async_p50:>14.3f}초")
print(f"  {'p95':18s} {sync_p95:>14.3f}초  {async_p95:>14.3f}초")
print(f"  {'속도 향상':18s} {'':>15s}   {speedup:>13.2f}배")
print(f"  {'절약 시간':18s} {'':>15s}   {saved_min:>12.2f}분")
print(f"{'=' * 60}")

# JSON 저장
result = {
    "dataset": "SKT",
    "csv": str(CSV_PATH),
    "model": model_name,
    "total_rows": TOTAL,
    "concurrency": CONCURRENCY,
    "measured_at": datetime.now().isoformat(),
    "before_sync": {
        "total_sec": sync_total,
        "total_min": sync_total / 60,
        "mean_sec": sync_mean,
        "p50_sec": sync_p50,
        "p95_sec": sync_p95,
    },
    "after_async": {
        "total_sec": async_total,
        "total_min": async_total / 60,
        "mean_sec": async_mean,
        "p50_sec": async_p50,
        "p95_sec": async_p95,
    },
    "speedup": speedup,
    "saved_min": saved_min,
}
RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(f"\n결과 저장: {RESULT_PATH}")
