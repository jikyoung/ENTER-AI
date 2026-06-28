"""Small review-to-action demo for Google Play apps.

This script intentionally avoids LLM calls. It checks whether a thin MVP is
viable by collecting recent reviews, ranking useful review candidates, and
summarizing issue signals for a main app and a competitor.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
from google_play_scraper import Sort, app, reviews


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "out" / "review_action_demo"

DEFAULT_APPS = [
    ("baemin", "com.sampleapp"),
    ("coupangeats", "com.coupang.mobile.eats"),
]

ISSUE_TAXONOMY = {
    "delivery_delay": [
        "배달",
        "지연",
        "늦",
        "시간",
        "도착",
        "라이더",
        "기사",
        "배차",
        "한시간",
        "식음",
        "식었",
    ],
    "customer_support": [
        "고객센터",
        "상담",
        "문의",
        "환불",
        "보상",
        "대응",
        "전화",
        "답변",
        "cs",
        "컴플레인",
    ],
    "coupon_price": [
        "쿠폰",
        "할인",
        "배달비",
        "가격",
        "와우",
        "클럽",
        "수수료",
        "무료",
        "포인트",
        "멤버십",
    ],
    "payment_order": [
        "결제",
        "주문",
        "취소",
        "카드",
        "페이",
        "계좌",
        "승인",
        "결재",
    ],
    "app_usability": [
        "앱",
        "오류",
        "버그",
        "렉",
        "튕",
        "로그인",
        "로딩",
        "화면",
        "업데이트",
        "ui",
        "알림",
    ],
    "ads_popup": [
        "광고",
        "팝업",
        "배너",
        "구독",
        "푸시",
        "알림",
    ],
    "food_quality": [
        "맛",
        "음식",
        "포장",
        "누락",
        "차갑",
        "식당",
        "가게",
        "메뉴",
    ],
    "address_location": [
        "주소",
        "위치",
        "지도",
        "gps",
        "현관",
        "배달지",
    ],
}


@dataclass(frozen=True)
class AppTarget:
    alias: str
    app_id: str


def parse_app_id(value: str) -> str:
    """Accept a Play URL or a package id."""
    if value.startswith("http"):
        query = parse_qs(urlparse(value).query)
        app_id = query.get("id", [""])[0]
        if not app_id:
            raise ValueError(f"Cannot find id= in URL: {value}")
        return app_id
    return value.strip()


def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def category_hits(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {
        category: sum(1 for token in keywords if token.lower() in lowered)
        for category, keywords in ISSUE_TAXONOMY.items()
    }


def useful_score(row: pd.Series, now: datetime) -> float:
    content = normalize_text(row["content"])
    length = len(content)
    score = int(row.get("score") or 0)
    thumbs_up = int(row.get("thumbs_up") or 0)
    at = row.get("at")

    rating_weight = {1: 3.0, 2: 2.5, 3: 1.2, 4: 0.4, 5: 0.2}.get(score, 0.5)
    length_score = min(length / 120, 1.5)
    short_penalty = -1.0 if length < 15 else 0.0
    helpful_score = min(math.log1p(thumbs_up), 2.5)

    recency_score = 0.0
    if isinstance(at, pd.Timestamp) and not pd.isna(at):
        days = max((now - at.to_pydatetime()).days, 0)
        recency_score = max(0.0, 1.0 - (days / 90))

    issue_hit_score = min(sum(category_hits(content).values()) * 0.25, 1.5)
    return rating_weight + length_score + short_penalty + helpful_score + recency_score + issue_hit_score


def dominant_category(text: str) -> str:
    hits = category_hits(text)
    category, count = max(hits.items(), key=lambda item: item[1])
    return category if count > 0 else "uncategorized"


def collect_app_reviews(target: AppTarget, count: int) -> tuple[dict, pd.DataFrame]:
    metadata = app(target.app_id, lang="ko", country="kr")
    rows, _ = reviews(
        target.app_id,
        lang="ko",
        country="kr",
        sort=Sort.NEWEST,
        count=count,
    )

    normalized = []
    for row in rows:
        normalized.append(
            {
                "alias": target.alias,
                "app_id": target.app_id,
                "app_title": metadata.get("title", target.alias),
                "developer": metadata.get("developer"),
                "review_id": row.get("reviewId"),
                "content": normalize_text(row.get("content")),
                "score": row.get("score"),
                "thumbs_up": row.get("thumbsUpCount", 0),
                "at": row.get("at"),
                "app_version": row.get("reviewCreatedVersion") or row.get("appVersion"),
                "user_name": row.get("userName"),
            }
        )

    return metadata, pd.DataFrame(normalized)


def summarize_app(df: pd.DataFrame) -> dict:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    df = df.copy()
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df["content_len"] = df["content"].str.len()
    df["issue_category"] = df["content"].apply(dominant_category)
    df["useful_score"] = df.apply(lambda row: useful_score(row, now), axis=1)

    issue_rows = df[df["issue_category"] != "uncategorized"]
    category_counts = issue_rows["issue_category"].value_counts().to_dict()
    negative = df[df["score"].isin([1, 2])]
    top_candidates = (
        df.sort_values(["useful_score", "thumbs_up"], ascending=False)
        .head(10)[
            [
                "review_id",
                "score",
                "thumbs_up",
                "at",
                "app_version",
                "issue_category",
                "useful_score",
                "content",
            ]
        ]
        .to_dict(orient="records")
    )

    version_counts = (
        df["app_version"].fillna("__MISSING__").value_counts().head(10).to_dict()
    )

    return {
        "app_title": df["app_title"].iloc[0] if not df.empty else None,
        "developer": df["developer"].iloc[0] if not df.empty else None,
        "app_id": df["app_id"].iloc[0] if not df.empty else None,
        "review_count": int(len(df)),
        "avg_rating": round(float(df["score"].mean()), 2) if not df.empty else None,
        "negative_review_count": int(len(negative)),
        "negative_review_ratio": round(float(len(negative) / len(df)), 3) if len(df) else 0,
        "short_review_ratio_lt15": round(float((df["content_len"] < 15).mean()), 3)
        if len(df)
        else 0,
        "category_counts": category_counts,
        "top_versions": version_counts,
        "top_useful_reviews": top_candidates,
    }


def build_comparison(summaries: dict[str, dict]) -> dict:
    categories = sorted(
        {
            category
            for summary in summaries.values()
            for category in summary["category_counts"].keys()
        }
    )
    rows = []
    for category in categories:
        row = {"issue_category": category}
        for alias, summary in summaries.items():
            count = summary["category_counts"].get(category, 0)
            row[f"{alias}_count"] = count
            row[f"{alias}_share"] = round(count / max(summary["review_count"], 1), 3)
        rows.append(row)
    return {"issue_comparison": rows}


def parse_targets(values: list[str]) -> list[AppTarget]:
    if not values:
        return [AppTarget(alias, app_id) for alias, app_id in DEFAULT_APPS]

    targets = []
    for raw in values:
        if "=" in raw:
            alias, value = raw.split("=", 1)
        else:
            value = raw
            alias = parse_app_id(value).split(".")[-1]
        targets.append(AppTarget(alias.strip(), parse_app_id(value)))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--app",
        action="append",
        default=[],
        help="App target as alias=package_id or alias=Google Play URL. Repeatable.",
    )
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = parse_targets(args.app)

    frames = []
    summaries = {}
    for target in targets:
        print(f"[collect] {target.alias}: {target.app_id} ({args.count} reviews)")
        _, df = collect_app_reviews(target, args.count)
        frames.append(df)
        summaries[target.alias] = summarize_app(df)

    all_reviews = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    raw_path = OUT_DIR / "reviews_raw.csv"
    summary_path = OUT_DIR / "summary.json"

    all_reviews.to_csv(raw_path, index=False)
    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "count_per_app_requested": args.count,
        "apps": summaries,
        "comparison": build_comparison(summaries),
    }
    summary_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str))

    print(f"\n[done] raw reviews -> {raw_path}")
    print(f"[done] summary     -> {summary_path}")
    print("\n[quick view]")
    for alias, summary in summaries.items():
        print(
            f"- {alias}: {summary['app_title']} / "
            f"{summary['review_count']} reviews / "
            f"avg {summary['avg_rating']} / "
            f"1-2★ {summary['negative_review_ratio']:.1%}"
        )
        print(f"  top issues: {Counter(summary['category_counts']).most_common(5)}")


if __name__ == "__main__":
    main()
