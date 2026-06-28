"""
ReportAgent / LangGraph 노드 단위 테스트

테스트 대상: server.modules.report_agent
  - should_retry(): final_report 유무에 따라 "done" / "retry" 분기
  - critic_node(): LLM mock → PASS/RETRY 처리, 최대 반복 강제 PASS
"""
import pytest
from unittest.mock import patch, MagicMock

from server.modules.report_agent import should_retry, critic_node


def _make_state(**overrides) -> dict:
    """테스트용 ReportState 딕셔너리 생성"""
    base = {
        "keyword":           "kt",
        "user_id":           "user01",
        "sentiment":         {"pos_pct": 30.0, "neg_pct": 20.0, "neu_pct": 50.0},
        "topics":            [],
        "context":           "",
        "sentiment_insight": "",
        "topic_insight":     "",
        "draft":             "보고서 초안 내용",
        "critique":          "",
        "iterations":        1,
        "final_report":      "",
    }
    base.update(overrides)
    return base


# ──────────────────────────────────────────────
# should_retry()
# ──────────────────────────────────────────────

def test_should_retry_with_final_report():
    """final_report가 있으면 'done' 반환"""
    state = _make_state(final_report="완성된 보고서")
    assert should_retry(state) == "done"


def test_should_retry_without_final_report():
    """final_report가 비어 있으면 'retry' 반환"""
    state = _make_state(final_report="")
    assert should_retry(state) == "retry"


# ──────────────────────────────────────────────
# critic_node()
# ──────────────────────────────────────────────

def test_critic_node_pass():
    """LLM이 'PASS' 반환 → final_report = draft"""
    state = _make_state(iterations=1)
    mock_result = MagicMock()
    mock_result.content = "PASS"

    with patch("server.modules.report_agent.ChatOpenAI") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_result
        mock_llm_cls.return_value = mock_llm

        result = critic_node(state)

    assert result.get("final_report") == "보고서 초안 내용"
    assert "critique" in result


def test_critic_node_retry():
    """LLM이 'RETRY: ...' 반환 → final_report 미포함"""
    state = _make_state(iterations=1)
    mock_result = MagicMock()
    mock_result.content = "RETRY: 감성 수치가 Executive Summary에 명시되지 않음"

    with patch("server.modules.report_agent.ChatOpenAI") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_result
        mock_llm_cls.return_value = mock_llm

        result = critic_node(state)

    # RETRY 시에는 final_report 키가 없어야 함
    assert "final_report" not in result
    assert "critique" in result


def test_critic_node_force_pass_at_max_iterations():
    """iterations=2에서 RETRY 응답이어도 강제 PASS → final_report = draft"""
    state = _make_state(iterations=2)
    mock_result = MagicMock()
    mock_result.content = "RETRY: 토픽 비율 누락"

    with patch("server.modules.report_agent.ChatOpenAI") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_result
        mock_llm_cls.return_value = mock_llm

        result = critic_node(state)

    # 최대 반복 도달 → 강제 PASS
    assert result.get("final_report") == "보고서 초안 내용"
