"""
FilterChain 단위 테스트

테스트 대상: filter_pipeline.filter_chain.FilterChain
주요 검증:
  - chain(): LLM mock → yes/no 응답 처리
  - _build_prompt(): keyword가 프롬프트에 포함되는지 확인
"""
import pytest
from unittest.mock import patch, MagicMock

from filter_pipeline.filter_chain import FilterChain


def _make_filter_chain(user_id: str = "user01", keyword: str = "kt") -> FilterChain:
    """테스트용 FilterChain 인스턴스 생성 (외부 의존성 없이)"""
    fc = FilterChain.__new__(FilterChain)
    fc.user_id = user_id
    fc.keyword = keyword
    fc._chain  = MagicMock()
    return fc


# ──────────────────────────────────────────────
# chain()
# ──────────────────────────────────────────────

def test_chain_returns_yes():
    """LLM이 'yes'로 시작하는 답변 반환 시 chain() 결과에 'yes' 포함"""
    fc = _make_filter_chain()
    fc._chain.predict.return_value = "yes, KT 서비스 관련 의견입니다."

    result = fc.chain("KT 인터넷 품질이 좋아요")

    assert result.strip().lower().startswith("yes")


def test_chain_returns_no():
    """LLM이 'no'로 시작하는 답변 반환 시 chain() 결과에 'no' 포함"""
    fc = _make_filter_chain()
    fc._chain.predict.return_value = "no"

    result = fc.chain("중고 노트북 팝니다")

    assert result.strip().lower().startswith("no")


# ──────────────────────────────────────────────
# _build_prompt()
# ──────────────────────────────────────────────

def test_build_prompt_contains_keyword():
    """_build_prompt() 결과에 keyword가 포함되는지 확인"""
    prompt = FilterChain._build_prompt("kt")

    assert "kt" in prompt


def test_build_prompt_contains_yes_no_instruction():
    """프롬프트에 'yes'/'no' 판단 지시문 포함 여부 확인"""
    prompt = FilterChain._build_prompt("kt")

    assert "yes" in prompt.lower()
    assert "no"  in prompt.lower()
