"""
TopicPipeline 단위 테스트

테스트 대상: server.modules.topic_pipeline.TopicPipeline
주요 검증:
  - run(): '무관' 토픽명 반환 시 결과에서 제외
  - run(): 30자 초과 토픽명 반환 시 결과에서 제외
  - to_summary_text(): 반환 형식 (건수 / 비율 포함) 확인
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from server.modules.topic_pipeline import TopicPipeline


def _make_pipeline(keyword: str = "kt") -> TopicPipeline:
    """FAISS 로드 없이 TopicPipeline 인스턴스 생성"""
    tp               = TopicPipeline.__new__(TopicPipeline)
    tp.user_id       = "user01"
    tp.keyword       = keyword
    tp.database_path = Path("/fake/path")
    return tp


def _make_mock_vectorstore(n_docs: int = 3, vector_dim: int = 64):
    """n_docs개의 문서를 가진 mock FAISS vectorstore 생성"""
    docs = []
    for i in range(n_docs):
        doc              = MagicMock()
        doc.page_content = f"테스트 의견 {i}"
        docs.append(doc)

    vs = MagicMock()
    vs.index_to_docstore_id        = {i: f"id{i}" for i in range(n_docs)}
    vs.docstore.search.side_effect = docs
    vs.index.ntotal                = n_docs
    vs.index.reconstruct_n.return_value = np.random.rand(n_docs, vector_dim).astype(np.float32)
    return vs


# ──────────────────────────────────────────────
# run()
# ──────────────────────────────────────────────

def test_run_filters_unrelated_topics():
    """_name_cluster가 '무관' 반환 시 해당 클러스터 결과에서 제외"""
    tp      = _make_pipeline()
    mock_vs = _make_mock_vectorstore(n_docs=3)

    with patch("server.modules.topic_pipeline.FAISS.load_local", return_value=mock_vs), \
         patch("server.modules.topic_pipeline.OpenAIEmbeddings"), \
         patch.object(TopicPipeline, "_name_cluster", return_value="무관"):
        results = tp.run(n_clusters=3)

    assert results == []


def test_run_filters_long_topic_name():
    """_name_cluster가 30자 초과 이름 반환 시 해당 클러스터 결과에서 제외"""
    tp        = _make_pipeline()
    mock_vs   = _make_mock_vectorstore(n_docs=3)
    long_name = "가" * 31  # 31자

    with patch("server.modules.topic_pipeline.FAISS.load_local", return_value=mock_vs), \
         patch("server.modules.topic_pipeline.OpenAIEmbeddings"), \
         patch.object(TopicPipeline, "_name_cluster", return_value=long_name):
        results = tp.run(n_clusters=3)

    assert results == []


def test_run_includes_valid_topic():
    """_name_cluster가 정상 토픽명 반환 시 결과에 포함"""
    tp      = _make_pipeline()
    mock_vs = _make_mock_vectorstore(n_docs=3)

    with patch("server.modules.topic_pipeline.FAISS.load_local", return_value=mock_vs), \
         patch("server.modules.topic_pipeline.OpenAIEmbeddings"), \
         patch.object(TopicPipeline, "_name_cluster", return_value="요금제 불만"):
        results = tp.run(n_clusters=3)

    assert len(results) > 0
    assert all(t["topic"] == "요금제 불만" for t in results)


# ──────────────────────────────────────────────
# to_summary_text()
# ──────────────────────────────────────────────

def test_to_summary_text_format():
    """to_summary_text() 반환 텍스트에 토픽명·건수·비율 포함 확인"""
    tp     = _make_pipeline()
    topics = [
        {"topic": "요금제 불만", "count": 50, "pct": 35.5, "samples": []},
        {"topic": "속도 문제",   "count": 30, "pct": 21.4, "samples": []},
    ]

    text = tp.to_summary_text(topics)

    assert "요금제 불만" in text
    assert "50건"         in text
    assert "35.5%"        in text
    assert "속도 문제"    in text


def test_to_summary_text_empty():
    """topics가 빈 리스트일 때 오류 없이 처리"""
    tp   = _make_pipeline()
    text = tp.to_summary_text([])

    assert isinstance(text, str)
