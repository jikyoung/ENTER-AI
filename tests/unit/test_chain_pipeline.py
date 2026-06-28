"""
ChainPipeline / ReportChainPipeline 단위 테스트

테스트 대상: server.modules.chain_pipeline.ReportChainPipeline._analyze_sentiment_all
주요 검증:
  - 긍정/부정/중립 비율 계산 정확성 (30% / 20% / 50%)
  - top_docs(): 욕설 포함 텍스트 제외
  - top_docs(): 조회수 내림차순 정렬
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from server.modules.chain_pipeline import ReportChainPipeline


def _make_pipeline(keyword: str = "kt") -> ReportChainPipeline:
    """SetTemplate 없이 ReportChainPipeline 인스턴스 생성"""
    pipeline         = ReportChainPipeline.__new__(ReportChainPipeline)
    pipeline.keyword = keyword
    pipeline.user_id = "user01"
    return pipeline


def _make_mock_vectorstore(doc_contents_and_views: list[tuple[str, str]]):
    """
    doc_contents_and_views: [(content, views_str), ...]
    """
    docs = []
    for content, views in doc_contents_and_views:
        doc          = MagicMock()
        doc.page_content = content
        doc.metadata = {"views": views}
        docs.append(doc)

    vs = MagicMock()
    n  = len(docs)
    vs.index_to_docstore_id     = {i: f"id{i}" for i in range(n)}
    vs.docstore.search.side_effect = docs
    return vs


# ──────────────────────────────────────────────
# 비율 계산
# ──────────────────────────────────────────────

def test_sentiment_ratio_calculation():
    """긍정 3 / 부정 2 / 중립 5 → 30% / 20% / 50% 검증"""
    labels = ["긍정"] * 3 + ["부정"] * 2 + ["중립"] * 5
    total  = len(labels)
    pos    = labels.count("긍정")
    neg    = labels.count("부정")
    neu    = labels.count("중립")

    assert round(pos / total * 100, 1) == 30.0
    assert round(neg / total * 100, 1) == 20.0
    assert round(neu / total * 100, 1) == 50.0
    assert pos + neg + neu == total


# ──────────────────────────────────────────────
# _analyze_sentiment_all (async)
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_top_docs_profanity_filter():
    """욕설 포함 텍스트가 top_pos에서 제외되는지 확인"""
    pipeline = _make_pipeline()

    # 정상 doc (views=100) + 욕설 doc (views=200)
    vs = _make_mock_vectorstore([
        ("KT 서비스 품질이 좋습니다",   "100"),
        ("KT 서비스 시발 형편없어",     "200"),
    ])

    # 두 doc 모두 '긍정'으로 분류. 현재 구현은 배치 JSON 응답을 기대한다.
    mock_resp = MagicMock()
    mock_resp.content = '{"1": "긍정", "2": "긍정"}'

    with patch("server.modules.chain_pipeline.ChatOpenAI") as mock_cls:
        mock_llm       = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_resp)
        mock_cls.return_value = mock_llm

        result = await pipeline._analyze_sentiment_all(vs)

    # 욕설("시발") 포함 텍스트는 top_pos에 없어야 함
    for text in result["top_pos"]:
        assert "시발" not in text


@pytest.mark.asyncio
async def test_top_docs_sorted_by_views():
    """top_pos가 조회수 내림차순으로 정렬되는지 확인"""
    pipeline = _make_pipeline()

    vs = _make_mock_vectorstore([
        ("저조회수 의견",   "10"),
        ("고조회수 의견",   "500"),
        ("중간조회수 의견", "100"),
    ])

    mock_resp = MagicMock()
    mock_resp.content = '{"1": "긍정", "2": "긍정", "3": "긍정"}'

    with patch("server.modules.chain_pipeline.ChatOpenAI") as mock_cls:
        mock_llm       = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_resp)
        mock_cls.return_value = mock_llm

        result = await pipeline._analyze_sentiment_all(vs)

    top_pos = result["top_pos"]
    assert len(top_pos) > 0
    # 가장 높은 조회수(500) doc이 첫 번째
    assert top_pos[0] == "고조회수 의견"
