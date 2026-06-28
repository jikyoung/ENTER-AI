"""
FastAPI 통합 테스트 (TestClient)

테스트 대상: project/main.py app
주요 검증:
  - user_id 유효성 검사 (400)
  - 존재하지 않는 리소스 (404)
  - Pydantic 입력 검증 (422)
  - 정상 응답 (200)
  - VectorPipeline DELETE 결과 분기

실행 방법:
    uv run pytest tests/integration/ -v
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


# ──────────────────────────────────────────────
# 입력 검증
# ──────────────────────────────────────────────

def test_invalid_user_id_returns_400():
    """user_id에 하이픈 등 특수문자 포함 시 400 반환"""
    response = client.get("/chats/invalid-user")

    assert response.status_code == 400
    assert "유효하지 않은" in response.json()["detail"]


def test_invalid_user_id_dot_returns_400():
    """user_id에 점('.') 포함 시 400 반환"""
    response = client.get("/chats/user.name")

    assert response.status_code == 400


def test_empty_question_returns_422():
    """빈 question 전송 → Pydantic validator 422"""
    response = client.post(
        "/answer/user01/kt/false",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_whitespace_question_returns_422():
    """공백만 있는 question 전송 → 422"""
    response = client.post(
        "/answer/user01/kt/false",
        json={"question": "   "},
    )

    assert response.status_code == 422


# ──────────────────────────────────────────────
# 404 처리
# ──────────────────────────────────────────────

def test_chat_list_not_found():
    """존재하지 않는 user_id로 채팅 목록 조회 → 404"""
    response = client.get("/chats/nonexistent_user_xyz9999")

    assert response.status_code == 404


def test_report_missing_vectordb_returns_404():
    """FAISS DB 없는 상태에서 보고서 요청 → 404"""
    response = client.post(
        "/report",
        json={"user_id": "nonexistent9999", "keyword": "nosuchkeyword"},
    )

    assert response.status_code == 404
    assert "크롤링" in response.json()["detail"]


# ──────────────────────────────────────────────
# 정상 응답
# ──────────────────────────────────────────────

def test_new_chat_success():
    """새 채팅 생성 → 200, status 확인"""
    with patch("server.apps.SetTemplate") as mock_st:
        mock_st.return_value.set_initial_templates.return_value = None
        response = client.get("/new_chat/testuser01")

    assert response.status_code == 200
    assert response.json()["status"] == "new_chat created!"


def test_delete_vectordb_success():
    """존재하는 keyword 삭제 → 200, delete success"""
    with patch("server.apps.VectorPipeline.delete_store_by_keyword",
               return_value={"status": "delete success"}):
        response = client.delete("/vectordb/user01/kt")

    assert response.status_code == 200
    assert response.json()["status"] == "delete success"


def test_delete_vectordb_not_found():
    """존재하지 않는 keyword 삭제 → 404"""
    with patch("server.apps.VectorPipeline.delete_store_by_keyword",
               return_value={"status": "abnormal delete request"}):
        response = client.delete("/vectordb/user01/nosuchkeyword")

    assert response.status_code == 404


def test_crawl_data_no_data_returns_false():
    """크롤링 데이터 없는 상태 → {"status": false}"""
    with patch("server.apps.CrawlManager") as mock_cm:
        mock_cm.return_value.get_crawl_data.return_value = {"status": False}
        response = client.post("/crawl_data/user01/nonexistent")

    assert response.status_code == 200
    assert response.json()["status"] is False
