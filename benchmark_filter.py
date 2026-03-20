"""
start_crawl LLM 필터링 성능 벤치마크
- Before: 동기 순차 처리 (현재 방식)
- After:  asyncio 병렬 처리 (개선 방식)
"""
import sys
from pathlib import Path

# project/ 디렉터리를 기준으로 경로 설정
project_root = Path(__file__).parent
project_dir  = project_root / 'project'
sys.path.insert(0, str(project_dir))

import time
import asyncio
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

load_dotenv(project_root / '.env')

from server.modules.set_template import SetTemplate

# ── 설정 ─────────────────────────────────────────────
USER_ID     = 'user01'
CSV_PATH    = project_dir / 'user_data' / 'user01' / 'crawl_data' / 'kt' / '2026-03-04T19h44m25s' / 'merged_data.csv'
SAMPLE_N    = 10   # API 비용 절약용 샘플 수
TOTAL_ROWS  = 730  # dropna 후 유효 행 수
TARGET_COL  = 'document'

# ── 공통: 프롬프트 로드 ───────────────────────────────
template_obj = SetTemplate(USER_ID)
prompt_str   = template_obj.load_template('llama', 'crawl')
model_name   = template_obj.load('chatgpt', 'params').model

llm = ChatOpenAI(model=model_name, temperature=0)
prompt = PromptTemplate(input_variables=["user_input"], template=prompt_str)
chain = LLMChain(llm=llm, prompt=prompt, verbose=False)


# ── 데이터 로드 & 샘플링 ──────────────────────────────
df = pd.read_csv(CSV_PATH).dropna(subset=[TARGET_COL])
df = df[df[TARGET_COL].str.strip() != '']
sample = df[TARGET_COL].head(SAMPLE_N).tolist()

print(f"\n{'='*55}")
print(f"  벤치마크 설정")
print(f"{'='*55}")
print(f"  모델       : {model_name}")
print(f"  전체 행 수 : {TOTAL_ROWS:,}건")
print(f"  샘플 수    : {SAMPLE_N}건")
print(f"{'='*55}\n")


# ════════════════════════════════════════════════════
# BEFORE: 동기 순차 처리 (현재 apps.py 방식)
# ════════════════════════════════════════════════════
print("[ BEFORE ] 동기 순차 처리 시작...")
sync_results = []
sync_start = time.time()

for i, text in enumerate(sample):
    t0 = time.time()
    result = chain.predict(user_input=text).strip()
    elapsed = time.time() - t0
    sync_results.append(result)
    print(f"  [{i+1:2d}/{SAMPLE_N}] {elapsed:.2f}s → {result}")

sync_total   = time.time() - sync_start
sync_per_row = sync_total / SAMPLE_N
sync_est     = sync_per_row * TOTAL_ROWS

print(f"\n  ▶ 샘플 {SAMPLE_N}건 총 소요 : {sync_total:.2f}초")
print(f"  ▶ 건당 평균 소요시간     : {sync_per_row:.2f}초")
print(f"  ▶ 전체 {TOTAL_ROWS}건 추정 소요 : {sync_est/60:.1f}분\n")


# ════════════════════════════════════════════════════
# AFTER: asyncio 병렬 처리 (개선 방식)
# ════════════════════════════════════════════════════
async def call_llm_async(text: str, idx: int) -> str:
    t0 = time.time()
    result = await chain.apredict(user_input=text)
    elapsed = time.time() - t0
    print(f"  [{idx+1:2d}/{SAMPLE_N}] {elapsed:.2f}s → {result.strip()}")
    return result.strip()

async def run_async_benchmark():
    print("[ AFTER  ] asyncio 병렬 처리 시작...")
    async_start = time.time()

    tasks = [call_llm_async(text, i) for i, text in enumerate(sample)]
    async_results = await asyncio.gather(*tasks)

    async_total = time.time() - async_start
    async_est   = (async_total / SAMPLE_N) * TOTAL_ROWS

    print(f"\n  ▶ 샘플 {SAMPLE_N}건 총 소요 : {async_total:.2f}초")
    print(f"  ▶ 건당 평균 소요시간     : {async_total/SAMPLE_N:.2f}초")
    print(f"  ▶ 전체 {TOTAL_ROWS}건 추정 소요 : {async_est/60:.1f}분\n")

    return async_total, async_est


async_total, async_est = asyncio.run(run_async_benchmark())


# ════════════════════════════════════════════════════
# 최종 비교 결과
# ════════════════════════════════════════════════════
speedup = sync_total / async_total

print(f"{'='*55}")
print(f"  최종 비교 결과")
print(f"{'='*55}")
print(f"  {'':20s} {'BEFORE':>10s}   {'AFTER':>10s}")
print(f"  {'샘플 처리시간':20s} {sync_total:>9.2f}s   {async_total:>9.2f}s")
print(f"  {'전체 추정시간':20s} {sync_est/60:>8.1f}분   {async_est/60:>8.1f}분")
print(f"  {'속도 향상':20s} {'':>10s}   {speedup:>8.1f}배 빠름")
print(f"  {'단축 시간':20s} {'':>10s}   {(sync_est-async_est)/60:>7.1f}분 절약")
print(f"{'='*55}")
