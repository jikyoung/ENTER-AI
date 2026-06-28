"""ENTER Day 0 Data Audit.

크롤링 데이터의 품질을 진단하고 라벨링용 샘플을 추출한다.
- URL 진단 (unique 수, 검색 URL 비율, TOP 5)
- 키워드 hit율 (본문 = document 컬럼)
- 본문 길이 분포 / 50자 미만 비율
- 사이트, 보드 카테고리 분포
- 100건 랜덤 샘플 -> 수기 라벨링용 CSV
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # ENTER-AI/
DATA = {
    "SKT": ROOT / "project/user_data/user01/crawl_data/SKT/2026-04-07T23h10m48s/merged_data.csv",
    "KT_v1": ROOT / "project/user_data/user01/crawl_data/kt/2026-03-04T18h51m05s/merged_data.csv",
    "KT_v2": ROOT / "project/user_data/user01/crawl_data/kt/2026-03-04T19h44m25s/merged_data.csv",
}
KEYWORDS = {
    "SKT": ["SKT", "skt", "에스케이텔레콤", "SK텔레콤", "에스케이"],
    "KT_v1": ["KT", "kt", "케이티"],
    "KT_v2": ["KT", "kt", "케이티"],
}
TEXT_COL = "document"
OUT_DIR = ROOT / "analysis" / "out"


def audit_one(name: str, path: Path) -> dict:
    df = pd.read_csv(path)
    total = len(df)

    url_series = df["url"].astype(str)
    unique_url = url_series.nunique()
    is_search = url_series.str.contains(r"search|query|q=", case=False, na=False, regex=True)
    top_urls = url_series.value_counts().head(5).to_dict()

    pat = "|".join(KEYWORDS[name])
    doc = df[TEXT_COL].astype(str)
    hit = doc.str.contains(pat, case=False, na=False, regex=True)

    doc_len = doc.str.len()

    site_dist = df["site"].value_counts().to_dict()
    board_dist = df["boardcategory"].value_counts().head(10).to_dict()

    noise_pat = "|".join(["중고", "거래", "장터", "판매"])
    board_str = df["boardcategory"].astype(str)
    noise_mask = board_str.str.contains(noise_pat, na=False, regex=True)

    return {
        "name": name,
        "total_rows": total,
        "unique_url": int(unique_url),
        "search_url_ratio": float(is_search.mean()),
        "top_5_urls": top_urls,
        "keyword_hit_ratio": float(hit.mean()),
        "doc_len_describe": {k: float(v) for k, v in doc_len.describe().to_dict().items()},
        "short_doc_ratio_lt50": float((doc_len < 50).mean()),
        "site_distribution": site_dist,
        "board_category_top10": board_dist,
        "noise_category_ratio": float(noise_mask.mean()),
    }


def print_report(r: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  DATASET: {r['name']}")
    print(f"{'=' * 60}")
    print(f"전체 행            : {r['total_rows']:,}")
    print(f"unique URL         : {r['unique_url']}")
    print(f"검색 URL 비율      : {r['search_url_ratio']:.1%}")
    print(f"키워드 hit율(본문) : {r['keyword_hit_ratio']:.1%}")
    print(f"본문 50자 미만     : {r['short_doc_ratio_lt50']:.1%}")
    print(f"중고/거래 카테고리 : {r['noise_category_ratio']:.1%}")

    print("\n[TOP 5 URL]")
    for u, c in r["top_5_urls"].items():
        print(f"  {c:>5}건  {u[:80]}")

    print("\n[사이트 분포]")
    for s, c in r["site_distribution"].items():
        print(f"  {c:>5}건  {s}")

    print("\n[보드 카테고리 TOP 10]")
    for b, c in r["board_category_top10"].items():
        print(f"  {c:>5}건  {b}")


def make_label_sample(name: str, path: Path, n: int = 100) -> Path:
    df = pd.read_csv(path)
    cols = ["url", "site", "document", "boardcategory"]
    sample = df.sample(min(n, len(df)), random_state=42)[cols].copy()
    sample["human_label"] = ""        # 고객경험 / 중고거래 / 스포츠 / 단순언급 / 타사이슈
    sample["exclude_reason"] = ""
    out = OUT_DIR / f"label_sample_{name}.csv"
    sample.to_csv(out, index=False)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name, path in DATA.items():
        if not path.exists():
            print(f"[skip] {name}: {path} not found")
            continue
        r = audit_one(name, path)
        results.append(r)
        print_report(r)
        sample_path = make_label_sample(name, path, n=100)
        print(f"\n라벨링 샘플 -> {sample_path}")

    summary_path = OUT_DIR / "audit_summary.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n전체 요약 JSON -> {summary_path}")


if __name__ == "__main__":
    main()
