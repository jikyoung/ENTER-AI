import json
from datetime import datetime
from pathlib import Path

import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator="README.md", pythonpath=True)

from server.modules.topic_pipeline import TopicPipeline

user_id = "user01"
keyword = "kt"

tp = TopicPipeline(user_id=user_id, keyword=keyword)
topics = tp.run(n_clusters=10)

summary = tp.to_summary_text(topics)

print("\n" + "="*50)
print(summary)
print("="*50)

print("\n[클러스터별 샘플 의견]")
for t in topics:
    print(f"\n▶ {t['topic']} ({t['count']}건, {t['pct']}%)")
    for s in t['samples']:
        print(f"  - {s[:80]}...")

# 결과 저장
out_dir = Path(__file__).parent / "test_results"
out_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = out_dir / f"topic_{keyword}_{timestamp}.json"
out_path.write_text(json.dumps(topics, ensure_ascii=False, indent=2))
print(f"\n결과 저장: {out_path}")
