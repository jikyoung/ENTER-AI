"""
E2E 파이프라인 단계별 처리시간 측정
크롤링 → LLM 필터링 → FAISS 임베딩 → PDF 리포트 생성
"""
import sys, time, asyncio, os, shutil
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

project_dir = __import__('pathlib').Path(__file__).parent / 'project'
sys.path.insert(0, str(project_dir))

from dotenv import load_dotenv
load_dotenv(__import__('pathlib').Path(__file__).parent / '.env')

import pandas as pd
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from server.modules.vectordb_pipeline import VectorPipeline
from server.modules.chain_pipeline import ReportChainPipeline
from llm_model.llama2_answer import LangchainPipline

CSV_PATH = project_dir / 'user_data/user01/crawl_data/kt/2026-03-04T19h44m25s/merged_data.csv'
USER_ID  = 'user01'
KEYWORD  = 'kt'
SAMPLE_N = 20
results  = {}

print('=' * 55)
print('  E2E 파이프라인 단계별 처리시간 측정')
print('=' * 55)

# ── Step 1: 크롤링 ───────────────────────────────────
# 실측 시: CrawlManager(USER_ID, KEYWORD).run() 주석 해제 후 실행
# 현재는 추정값 사용 (순차: ~8분 / 병렬화 후: ~2~3분)
CRAWL_EST_BEFORE = 6.9   # 순차 실행 before (benchmark_crawl.py 실측 환산)
CRAWL_EST_AFTER  = 2.6   # 병렬 실행 after  (benchmark_crawl.py 실측 환산, 2.7x)
results['crawl'] = CRAWL_EST_AFTER * 60
print(f'\n[Step 1] 크롤링')
print(f'         Before (순차): {CRAWL_EST_BEFORE}분')
print(f'         After  (병렬): {CRAWL_EST_AFTER}분  (Popen 병렬화, 2.7x 향상)')
print(f'         근거: benchmark_crawl.py 실측 (subprocess 타이밍 직접 측정)')

# ── Step 2: CSV 로드 & 전처리 ────────────────────────
t0 = time.time()
data = pd.read_csv(CSV_PATH)
data = data.dropna(subset=['document'])
data = data[data['document'].str.strip().ne('')].reset_index(drop=True)
results['merge'] = time.time() - t0
print(f'\n[Step 2] CSV 병합 & 전처리')
print(f'         {len(data)}건 유효 데이터 로드: {results["merge"]:.3f}초')

# ── Step 3: LLM 필터링 (async, 샘플 추정) ───────────
lp = LangchainPipline(user_id=USER_ID)
sample = data['document'].head(SAMPLE_N).tolist()

async def run_filter():
    return await asyncio.gather(*[lp.async_chain(question=t) for t in sample])

t0 = time.time()
filter_res = asyncio.run(run_filter())
sample_time = time.time() - t0
total_filter = (sample_time / SAMPLE_N) * len(data)
pass_rate = sum(1 for r in filter_res if r.strip().lower().startswith('yes')) / SAMPLE_N
results['filter'] = total_filter

print(f'\n[Step 3] LLM 필터링 (async)')
print(f'         샘플 {SAMPLE_N}건: {sample_time:.2f}초')
print(f'         전체 {len(data)}건 추정: {total_filter:.1f}초 ({total_filter/60:.1f}분)')
print(f'         통과율: {pass_rate * 100:.0f}%')

# ── Step 4: FAISS 임베딩 ─────────────────────────────
est_pass = int(len(data) * pass_rate)
embed_sample = data.head(SAMPLE_N).copy()
t0 = time.time()
VectorPipeline.embedding_and_store(
    data=embed_sample, user_id='bm_tmp', keyword=KEYWORD,
    target_col='document', embedding=OpenAIEmbeddings()
)
embed_time = time.time() - t0
total_embed = (embed_time / SAMPLE_N) * est_pass
results['faiss'] = total_embed

tmp = project_dir / 'user_data/bm_tmp'
if tmp.exists():
    shutil.rmtree(tmp)

print(f'\n[Step 4] FAISS 임베딩 저장')
print(f'         샘플 {SAMPLE_N}건: {embed_time:.2f}초')
print(f'         예상 통과 {est_pass}건 추정: {total_embed:.1f}초')

# ── Step 5: PDF 리포트 생성 ──────────────────────────
print(f'\n[Step 5] PDF 리포트 생성')
t0 = time.time()
rcp = ReportChainPipeline(user_id=USER_ID, keyword=KEYWORD)
pdf_path = rcp.load_chain()
results['pdf'] = time.time() - t0
print(f'         소요시간: {results["pdf"]:.2f}초')
print(f'         저장경로: {pdf_path}')

# ── 최종 요약 ────────────────────────────────────────
total_sec = sum(results.values())
print(f'\n{"=" * 55}')
print(f'  단계별 소요시간 요약')
print(f'{"=" * 55}')
print(f'  Step 1  크롤링 (병렬화 후)  : {results["crawl"] / 60:>5.1f}분  (before: {CRAWL_EST_BEFORE:.1f}분)')
print(f'  Step 2  CSV 전처리          : {results["merge"]:>5.3f}초')
print(f'  Step 3  LLM 필터링 (async)  : {results["filter"] / 60:>5.1f}분')
print(f'  Step 4  FAISS 임베딩        : {results["faiss"]:>5.1f}초')
print(f'  Step 5  PDF 생성            : {results["pdf"]:>5.1f}초')
print(f'  {"─" * 45}')
print(f'  총 E2E 소요시간 (추정)      : {total_sec / 60:>5.1f}분')
print(f'{"=" * 55}')
