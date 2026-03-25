import asyncio
from typing import TypedDict, Annotated
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, format_document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langgraph.graph import StateGraph, END

from server.modules.set_template import SetTemplate
from server.modules.topic_pipeline import TopicPipeline


# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────

class ReportState(TypedDict):
    keyword:           str
    user_id:           str
    sentiment:         dict          # _analyze_sentiment_all() 결과
    topics:            list          # TopicPipeline.run() 결과
    context:           str           # FAISS 검색 문서
    sentiment_insight: str           # SentimentAgent 해석
    topic_insight:     str           # TopicAgent 해석
    draft:             str           # WriterAgent 초안
    critique:          str           # CriticAgent 피드백
    iterations:        int           # 재작업 횟수
    final_report:      str           # 최종 보고서


# ──────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────

def sentiment_node(state: ReportState) -> dict:
    """감성 수치 → 핵심 인사이트 문장화"""
    s = state["sentiment"]
    keyword = state["keyword"]

    summary = (
        f"전체 {s['total']}건 중 긍정 {s['pos']}건({s['pos_pct']}%), "
        f"부정 {s['neg']}건({s['neg_pct']}%), 중립 {s['neu']}건({s['neu_pct']}%)"
    )
    top_pos = "\n".join(f"- {d[:200]}" for d in s.get("top_pos", []))
    top_neg = "\n".join(f"- {d[:200]}" for d in s.get("top_neg", []))

    prompt = (
        f"다음은 '{keyword}'에 대한 커뮤니티 감성 분석 결과입니다.\n\n"
        f"[수치]\n{summary}\n\n"
        f"[긍정 대표 의견]\n{top_pos}\n\n"
        f"[부정 대표 의견]\n{top_neg}\n\n"
        f"위 데이터를 바탕으로:\n"
        f"1. 전반적인 여론 방향과 그 근거\n"
        f"2. 긍정 여론의 핵심 원인\n"
        f"3. 부정 여론의 핵심 원인\n"
        f"을 3~5문장으로 해석하세요. 수치는 반드시 포함하세요."
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    result = llm.invoke(prompt)
    return {"sentiment_insight": result.content.strip()}


def topic_node(state: ReportState) -> dict:
    """토픽 클러스터 → 주요 이슈 해석 및 우선순위 정리"""
    topics = state["topics"]
    keyword = state["keyword"]

    topic_lines = "\n".join(
        f"- {t['topic']}: {t['count']}건({t['pct']}%)"
        for t in topics
    )

    prompt = (
        f"다음은 '{keyword}' 관련 커뮤니티 의견의 토픽 클러스터링 결과입니다.\n\n"
        f"{topic_lines}\n\n"
        f"위 토픽들을 분석하여:\n"
        f"1. 가장 많이 언급된 상위 3개 이슈와 의미\n"
        f"2. 사용자들이 가장 불만스러워하는 영역\n"
        f"3. 개선이 필요한 우선순위\n"
        f"를 구체적으로 해석하세요. 각 토픽의 비율 수치를 반드시 인용하세요."
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    result = llm.invoke(prompt)
    return {"topic_insight": result.content.strip()}


def writer_node(state: ReportState) -> dict:
    """sentiment/topic 해석 + FAISS context → 보고서 초안 작성"""
    critique = state.get("critique", "")
    retry_instruction = ""
    if critique and "RETRY" in critique:
        feedback = critique.replace("RETRY:", "").strip()
        retry_instruction = f"\n\n[이전 검토 피드백 - 반드시 반영하세요]\n{feedback}"

    prompt = (
        f"다음 분석 결과를 바탕으로 '{state['keyword']}' 온라인 여론 분석 보고서를 작성하세요.\n"
        f"반드시 제공된 데이터의 수치와 실제 의견을 인용하고, 추측하지 마세요.\n"
        f"{retry_instruction}\n\n"
        f"[감성 분석 인사이트]\n{state['sentiment_insight']}\n\n"
        f"[토픽 분석 인사이트]\n{state['topic_insight']}\n\n"
        f"[상세 의견 데이터]\n{state['context']}\n\n"
        f"보고서 섹션:\n"
        f"* Executive Summary: 감성 수치 포함, 핵심 여론 요약\n"
        f"# 데이터 수집 개요: 수집 규모 및 방법\n"
        f"# 감성 분석 결과: 수치 상세, 긍정·부정 대표 의견 인용\n"
        f"# 주요 토픽 분석: 상위 토픽별 비율과 대표 의견 인용\n"
        f"# 핵심 이슈 및 시사점: 데이터 기반 개선 제언\n\n"
        f"제목은 *, 소제목은 #, 항목은 -로 시작하세요."
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    result = llm.invoke(prompt)
    return {
        "draft": result.content.strip(),
        "iterations": state.get("iterations", 0) + 1,
    }


def critic_node(state: ReportState) -> dict:
    """초안 품질 평가 → PASS or RETRY"""
    s = state["sentiment"]
    checklist = (
        f"[평가 기준]\n"
        f"1. 감성 수치(긍정 {s['pos_pct']}%, 부정 {s['neg_pct']}%, 중립 {s['neu_pct']}%)가 보고서에 명시됐는가?\n"
        f"2. 토픽 분석 결과(비율 포함)가 인용됐는가?\n"
        f"3. 실제 사용자 의견이 구체적으로 인용됐는가?\n"
        f"4. 데이터에 없는 내용을 추측하거나 일반론으로 채우지 않았는가?\n\n"
        f"[보고서 초안]\n{state['draft']}\n\n"
        f"위 기준을 모두 충족하면 'PASS'만 답하세요.\n"
        f"미충족 항목이 있으면 'RETRY: [구체적 개선 지시]' 형식으로 답하세요."
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    result = llm.invoke(checklist)
    critique = result.content.strip()

    if "PASS" in critique or state.get("iterations", 0) >= 2:
        return {"critique": critique, "final_report": state["draft"]}
    return {"critique": critique}


# ──────────────────────────────────────────────
# Conditional edge
# ──────────────────────────────────────────────

def should_retry(state: ReportState) -> str:
    if state.get("final_report"):
        return "done"
    return "retry"


# ──────────────────────────────────────────────
# Graph 조립
# ──────────────────────────────────────────────

def build_report_graph():
    g = StateGraph(ReportState)

    g.add_node("sentiment", sentiment_node)
    g.add_node("topic",     topic_node)
    g.add_node("writer",    writer_node)
    g.add_node("critic",    critic_node)

    g.set_entry_point("sentiment")
    g.add_edge("sentiment", "topic")
    g.add_edge("topic",     "writer")
    g.add_edge("writer",    "critic")
    g.add_conditional_edges("critic", should_retry, {
        "retry": "writer",
        "done":  END,
    })

    return g.compile()


# ──────────────────────────────────────────────
# ReportAgent — 외부에서 호출하는 진입점
# ──────────────────────────────────────────────

class ReportAgent:

    def __init__(self, user_id: str, keyword: str):
        self.user_id  = user_id
        self.keyword  = keyword
        self.BASE_DIR = Path(__file__).parent.parent.parent / "user_data" / user_id
        self.config   = SetTemplate(user_id)
        self.graph    = build_report_graph()

    async def _build_initial_state(self) -> ReportState:
        database_path = self.BASE_DIR / "database" / self.keyword

        vectorstore = FAISS.load_local(
            folder_path=database_path,
            embeddings=OpenAIEmbeddings(),
            allow_dangerous_deserialization=True,
        )

        # 감성 분석 + 토픽 클러스터링 병렬 실행
        from server.modules.chain_pipeline import ReportChainPipeline
        pipeline = ReportChainPipeline(user_id=self.user_id, keyword=self.keyword)

        sentiment_task = pipeline._analyze_sentiment_all(vectorstore)
        topic_task = asyncio.get_event_loop().run_in_executor(
            None,
            lambda: TopicPipeline(self.user_id, self.keyword).run(n_clusters=10),
        )
        sentiment, topics = await asyncio.gather(sentiment_task, topic_task)

        # FAISS 검색
        config = self.config.params.load(
            self.BASE_DIR / "template" / "configs.yaml", addict=False
        )["chatgpt"]["templates"]["report"]
        document_template = config["document"] or config["document_default"]

        retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 30}),
            llm=ChatOpenAI(model=self.config.load("chatgpt", "params").model, temperature=0),
        )
        docs = retriever.invoke(self.keyword)

        DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template=document_template)
        MAX_CHARS = 1000

        def combine(docs):
            truncated = []
            for d in docs:
                if len(d.page_content) > MAX_CHARS:
                    d.page_content = d.page_content[:MAX_CHARS] + "..."
                truncated.append(d)
            return "\n".join(format_document(d, DEFAULT_DOCUMENT_PROMPT) for d in truncated)

        return ReportState(
            keyword=self.keyword,
            user_id=self.user_id,
            sentiment=sentiment,
            topics=topics,
            context=combine(docs),
            sentiment_insight="",
            topic_insight="",
            draft="",
            critique="",
            iterations=0,
            final_report="",
        )

    async def run(self) -> str:
        initial_state = await self._build_initial_state()
        result = await asyncio.get_event_loop().run_in_executor(
            None, self.graph.invoke, initial_state
        )
        return result["final_report"]
