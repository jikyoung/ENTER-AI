import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator="README.md", pythonpath=True)

from server.modules.topic_pipeline import TopicPipeline

user_id = "user01"
keyword = "kt"

tp = TopicPipeline(user_id=user_id, keyword=keyword)
topics = tp.run(n_clusters=10)

print("\n" + "="*50)
print(tp.to_summary_text(topics))
print("="*50)

print("\n[클러스터별 샘플 의견]")
for t in topics:
    print(f"\n▶ {t['topic']} ({t['count']}건, {t['pct']}%)")
    for s in t['samples']:
        print(f"  - {s[:80]}...")
