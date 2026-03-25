"""
보고서 생성 전 context 구성 확인용 테스트
GPT 보고서 생성 및 PDF 변환 없이 context만 출력
"""
import asyncio
import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator="README.md", pythonpath=True)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, format_document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from server.modules.chain_pipeline import ReportChainPipeline
from server.modules.topic_pipeline import TopicPipeline

USER_ID = "user01"
KEYWORD = "kt"


async def main():
    pipeline = ReportChainPipeline(user_id=USER_ID, keyword=KEYWORD)

    vectorstore = FAISS.load_local(
        folder_path=pipeline.database_path,
        embeddings=OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )

    print("감성 분석 + 토픽 클러스터링 병렬 실행 중...")
    sentiment_task = pipeline._analyze_sentiment_all(vectorstore)
    topic_task = asyncio.get_event_loop().run_in_executor(
        None,
        lambda: TopicPipeline(USER_ID, KEYWORD).run(n_clusters=10)
    )
    sentiment, topics = await asyncio.gather(sentiment_task, topic_task)

    sentiment_summary = f"""
[감성 분석 결과 - 전체 {sentiment['total']}개 문서]
- 긍정: {sentiment['pos']}개 ({sentiment['pos_pct']}%)
- 부정: {sentiment['neg']}개 ({sentiment['neg_pct']}%)
- 중립: {sentiment['neu']}개 ({sentiment['neu_pct']}%)

[긍정 대표 의견]
{chr(10).join(f'- {d}' for d in sentiment['top_pos'])}

[부정 대표 의견]
{chr(10).join(f'- {d}' for d in sentiment['top_neg'])}
"""

    topic_summary = TopicPipeline(USER_ID, KEYWORD).to_summary_text(topics)

    print("\n" + "="*60)
    print("【감성 분석】")
    print(sentiment_summary)
    print("="*60)
    print("【토픽 클러스터링】")
    print(topic_summary)
    print("="*60)
    print(f"\n✅ context에 감성 분석 포함: {'긍정' in sentiment_summary}")
    print(f"✅ context에 토픽 클러스터링 포함: {'토픽 클러스터링 결과' in topic_summary}")
    print(f"✅ 토픽 수: {len(topics)}개")


asyncio.run(main())
