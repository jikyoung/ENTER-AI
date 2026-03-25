"""
LangGraph Multi-Agent 보고서 생성 테스트
PDF 변환 없이 최종 보고서 텍스트만 확인
"""
import asyncio
from datetime import datetime
from pathlib import Path

import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator="README.md", pythonpath=True)

from server.modules.report_agent import ReportAgent

USER_ID = "user01"
KEYWORD = "kt"


async def main():
    print("LangGraph Multi-Agent 보고서 생성 시작...\n")
    agent = ReportAgent(user_id=USER_ID, keyword=KEYWORD)
    report = await agent.run()

    print("=" * 60)
    print(report)
    print("=" * 60)

    out_dir = Path(__file__).parent / "test_results"
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"report_agent_{KEYWORD}_{timestamp}.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n결과 저장: {out_path}")


asyncio.run(main())
