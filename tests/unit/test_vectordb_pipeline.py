"""
VectorPipeline 단위 테스트

테스트 대상: server.modules.vectordb_pipeline.VectorPipeline
주요 검증:
  - merge_into_store: DB 없을 때 신규 생성, 기존 URL 중복 제거
  - get_existing_urls: 디렉터리 없을 때 빈 set 반환
  - delete_store_by_keyword: 정상 삭제, 디렉터리 부분 누락 시 비정상 처리
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from server.modules.vectordb_pipeline import VectorPipeline


# ──────────────────────────────────────────────
# merge_into_store
# ──────────────────────────────────────────────

def test_merge_into_store_new_data(tmp_path):
    """DB가 없을 때 embedding_and_store 호출 및 신규 행 수 반환"""
    df = pd.DataFrame({
        "document": ["text1", "text2"],
        "url": ["http://a.com", "http://b.com"],
    })

    with patch.object(VectorPipeline, "BASE_DIR", tmp_path), \
         patch.object(VectorPipeline, "embedding_and_store") as mock_embed:
        result = VectorPipeline.merge_into_store(df, "user01", "kt", MagicMock())

    mock_embed.assert_called_once()
    assert result == 2


def test_merge_into_store_no_new_data(tmp_path):
    """모든 URL이 이미 FAISS에 존재할 때 0 반환 및 DB 수정 없음"""
    df = pd.DataFrame({
        "document": ["text1"],
        "url": ["http://existing.com"],
    })

    # DB 디렉터리가 존재한다고 설정
    db_dir = tmp_path / "user01" / "database" / "kt"
    db_dir.mkdir(parents=True)

    with patch.object(VectorPipeline, "BASE_DIR", tmp_path), \
         patch.object(VectorPipeline, "get_existing_urls", return_value={"http://existing.com"}):
        result = VectorPipeline.merge_into_store(df, "user01", "kt", MagicMock())

    assert result == 0


def test_merge_into_store_filters_existing_urls(tmp_path):
    """일부 URL만 신규일 때 신규 건수만 반환"""
    df = pd.DataFrame({
        "document": ["old doc", "new doc"],
        "url": ["http://existing.com", "http://new.com"],
    })

    db_dir = tmp_path / "user01" / "database" / "kt"
    db_dir.mkdir(parents=True)

    mock_vs = MagicMock()
    mock_new_vs = MagicMock()

    with patch.object(VectorPipeline, "BASE_DIR", tmp_path), \
         patch.object(VectorPipeline, "get_existing_urls", return_value={"http://existing.com"}), \
         patch("server.modules.vectordb_pipeline.DataFrameLoader") as mock_loader, \
         patch("server.modules.vectordb_pipeline.FAISS") as mock_faiss:
        mock_faiss.load_local.return_value = mock_vs
        mock_faiss.from_documents.return_value = mock_new_vs
        mock_loader.return_value.load.return_value = [MagicMock()]

        result = VectorPipeline.merge_into_store(df, "user01", "kt", MagicMock())

    # 신규 URL 1건만 추가
    assert result == 1


# ──────────────────────────────────────────────
# get_existing_urls
# ──────────────────────────────────────────────

def test_get_existing_urls_no_dir(tmp_path):
    """FAISS 디렉터리가 없을 때 빈 set 반환"""
    with patch.object(VectorPipeline, "BASE_DIR", tmp_path):
        urls = VectorPipeline.get_existing_urls("user01", "nonexistent")

    assert urls == set()


# ──────────────────────────────────────────────
# delete_store_by_keyword
# ──────────────────────────────────────────────

def test_delete_store_both_exist(tmp_path):
    """database_path, history_path 둘 다 존재 → delete success, 디렉터리 삭제"""
    user_id, keyword = "testuser", "testkw"

    db_path   = tmp_path / user_id / "database" / keyword
    hist_path = tmp_path / user_id / "history"  / keyword
    db_path.mkdir(parents=True)
    hist_path.mkdir(parents=True)

    with patch.object(VectorPipeline, "BASE_DIR", tmp_path):
        result = VectorPipeline.delete_store_by_keyword(user_id, keyword)

    assert result["status"] == "delete success"
    assert not db_path.exists()
    assert not hist_path.exists()


def test_delete_store_partial_missing(tmp_path):
    """history_path가 없을 때 → abnormal delete request"""
    user_id, keyword = "testuser", "testkw"

    # database는 있지만 history는 없음
    db_path = tmp_path / user_id / "database" / keyword
    db_path.mkdir(parents=True)

    with patch.object(VectorPipeline, "BASE_DIR", tmp_path):
        result = VectorPipeline.delete_store_by_keyword(user_id, keyword)

    assert result["status"] == "abnormal delete request"
