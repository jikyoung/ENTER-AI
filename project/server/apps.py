import os
import re
import asyncio
import shutil
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from langchain_openai import OpenAIEmbeddings

from server.modules.set_template import SetTemplate
from filter_pipeline.filter_chain import FilterChain
from server.modules.crawl_pipeline import CrawlManager
from server.modules.vectordb_pipeline import VectorPipeline
from server.modules.chain_pipeline import ChainPipeline, ReportChainPipeline
from server.modules.report_agent import ReportAgent
from utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# 요청 모델
# ──────────────────────────────────────────────

class Quest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question은 빈 문자열일 수 없습니다.")
        return v.strip()

class Report(BaseModel):
    user_id: str
    keyword: str

class Template(BaseModel):
    template_config: dict[str, str]


# ──────────────────────────────────────────────
# 응답 모델
# ──────────────────────────────────────────────

class ChatListResponse(BaseModel):
    keywords: list[str]

class CrawlResponse(BaseModel):
    status: str
    added: int
    total_filtered: int

class DeleteResponse(BaseModel):
    status: str

class NewChatResponse(BaseModel):
    status: str

class CrawlDataResponse(BaseModel):
    status: bool
    data: dict | None = None


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

def validate_user_id(user_id: str) -> None:
    """user_id에 경로 문자 포함 여부 검증. 영문·숫자·언더스코어만 허용."""
    if not re.match(r"^[a-zA-Z0-9_]+$", user_id):
        raise HTTPException(status_code=400, detail="유효하지 않은 user_id입니다.")


# ──────────────────────────────────────────────
# 서버
# ──────────────────────────────────────────────

class FastApiServer:

    def __init__(self):
        self.router = APIRouter()
        self.register_routes()

    def register_routes(self):
        self.router.add_api_route("/chats/{user_id}",                               self.chat_list,      methods=["GET"])
        self.router.add_api_route("/answer/{user_id}/{keyword}/{stream}",            self.answer,         methods=["POST"])
        self.router.add_api_route("/history/{user_id}/{keyword}",                    self.history,        methods=["GET"])

        self.router.add_api_route("/crawl/{user_id}/{keyword}",                      self.start_crawl,    methods=["POST"])
        self.router.add_api_route("/crawl_data/{user_id}/{keyword}",                 self.get_crawl_data, methods=["POST"])

        self.router.add_api_route("/new_chat/{user_id}",                             self.new_chat,       methods=["GET"])
        self.router.add_api_route("/vectordb/{user_id}/{keyword}",                   self.delete_vectordb, methods=["DELETE"])
        self.router.add_api_route("/report",                                         self.report,         methods=["POST"])

        self.router.add_api_route("/load_template/{user_id}/{llm}/{template_type}",  self.load_template,  methods=["GET"])
        self.router.add_api_route("/edit_template/{user_id}/{llm}/{template_type}",  self.edit_template,  methods=["POST"])


    async def chat_list(self, user_id: str) -> ChatListResponse:
        validate_user_id(user_id)

        chat_list_path = Path(__file__).parent.parent / "user_data" / user_id / "database"
        if not chat_list_path.is_dir():
            raise HTTPException(status_code=404, detail="해당 사용자의 데이터가 없습니다.")

        keywords = [p for p in os.listdir(chat_list_path) if not p.startswith(".")]
        return ChatListResponse(keywords=keywords)


    async def answer(self,
                     user_id: str,
                     keyword: str,
                     stream: bool,
                     item: Quest):
        validate_user_id(user_id)

        chainpipe      = ChainPipeline(user_id=user_id, keyword=keyword)
        history        = chainpipe.load_history()
        chain          = chainpipe.load_chain()
        response_input = {"question": item.question}

        if stream:
            return StreamingResponse(
                content    = chainpipe.streaming(chain, response_input),
                media_type = "text/event-stream",
            )

        result = chain.invoke(response_input)
        history.save_context(response_input, {"answer": result["answer"].content})
        chainpipe.memory = history
        chainpipe.save_history()
        return result


    async def report(self, data: Report) -> FileResponse:
        validate_user_id(data.user_id)

        db_path = Path(__file__).parent.parent / "user_data" / data.user_id / "database" / data.keyword
        if not db_path.is_dir():
            raise HTTPException(status_code=404, detail="해당 키워드의 벡터DB가 없습니다. 먼저 크롤링을 실행하세요.")

        logger.info("보고서 생성 시작 | user_id=%s keyword=%s", data.user_id, data.keyword)
        agent       = ReportAgent(user_id=data.user_id, keyword=data.keyword)
        report_text = await agent.run()

        chainpipe = ReportChainPipeline(user_id=data.user_id, keyword=data.keyword)
        pdf_path  = chainpipe.to_pdf(report_text)

        logger.info("보고서 생성 완료 | user_id=%s keyword=%s", data.user_id, data.keyword)
        return FileResponse(path=pdf_path, filename="report.pdf", media_type="application/octet-stream")


    async def history(self, user_id: str, keyword: str):
        validate_user_id(user_id)

        chainpipe = ChainPipeline(user_id=user_id, keyword=keyword)
        return chainpipe.conversation_json()


    async def load_template(self, user_id: str, llm: str, template_type: str):
        validate_user_id(user_id)

        st = SetTemplate(user_id=user_id)
        return st.load(llm=llm, template_type=template_type)


    async def edit_template(self, user_id: str, llm: str, template_type: str, config: Template):
        validate_user_id(user_id)

        st = SetTemplate(user_id=user_id)
        st.edit(llm=llm, template_type=template_type, **config.template_config)
        return {"status": "ok"}


    async def delete_vectordb(self, user_id: str, keyword: str) -> DeleteResponse:
        validate_user_id(user_id)

        result = VectorPipeline.delete_store_by_keyword(user_id=user_id, keyword=keyword)
        if result["status"] == "abnormal delete request":
            raise HTTPException(status_code=404, detail="해당 키워드의 데이터가 없습니다.")
        return DeleteResponse(status=result["status"])


    async def start_crawl(self, user_id: str, keyword: str) -> CrawlResponse:
        validate_user_id(user_id)

        target_col        = "document"
        EXCLUDE_CATEGORIES = ["장터", "중고", "팝니다", "삽니다", "판매", "스포츠", "게임", "정치"]

        lp = FilterChain(user_id=user_id, keyword=keyword)
        cm = CrawlManager(user_id=user_id, keyword=keyword)
        cm.run()

        merged_csv = cm.base_dir / "merged_data.csv"
        if not merged_csv.exists():
            raise HTTPException(status_code=500, detail="크롤링 결과가 없습니다. 스파이더 실행을 확인하세요.")

        data = pd.read_csv(merged_csv)
        data = data.dropna(subset=[target_col])
        data = data[data[target_col].str.strip() != ""].reset_index(drop=True)

        for col in ["boardcategory", "documentcategory"]:
            if col in data.columns:
                mask = data[col].fillna("").apply(
                    lambda c: not any(ex in str(c) for ex in EXCLUDE_CATEGORIES)
                )
                data = data[mask].reset_index(drop=True)

        results = await asyncio.gather(*[
            lp.async_chain(question=text)
            for text in data[target_col]
        ])

        mask      = [r.strip().lower().startswith("yes") for r in results]
        result_df = data[mask].reset_index(drop=True)
        result_df.to_csv(cm.base_dir / "filtered_data.csv", index=False)

        # 이전 크롤링 폴더 삭제 (최신 1개만 유지)
        crawl_keyword_dir = cm.base_dir.parent
        old_dirs = sorted(crawl_keyword_dir.iterdir())
        for old_dir in old_dirs[:-1]:
            shutil.rmtree(str(old_dir))

        embedding = OpenAIEmbeddings()
        added = VectorPipeline.merge_into_store(
            data       = result_df,
            user_id    = user_id,
            keyword    = keyword,
            target_col = target_col,
            embedding  = embedding,
        )
        return CrawlResponse(status="ok", added=added, total_filtered=len(result_df))


    async def new_chat(self, user_id: str) -> NewChatResponse:
        validate_user_id(user_id)

        template = SetTemplate(user_id=user_id)
        template.set_initial_templates()
        return NewChatResponse(status="new_chat created!")


    async def get_crawl_data(self, user_id: str, keyword: str) -> CrawlDataResponse:
        validate_user_id(user_id)

        cm = CrawlManager(user_id=user_id, keyword=keyword)
        try:
            result = cm.get_crawl_data()
            if result.get("status") is False:
                return CrawlDataResponse(status=False)
            return CrawlDataResponse(status=True, data=result)
        except Exception as e:
            logger.error("크롤 데이터 조회 실패 | user_id=%s keyword=%s | %s: %s",
                         user_id, keyword, type(e).__name__, e, exc_info=True)
            return CrawlDataResponse(status=False)
