"""Check which Google Play review fields are available from the scraper.

This script does not modify project data. It fetches a few live reviews and
prints the keys/values that can be preserved by the crawler.
"""
from __future__ import annotations

import argparse
from pprint import pprint

from google_play_scraper import Sort, reviews, search


def resolve_app_id(keyword: str, app_id: str | None) -> str:
    if app_id:
        return app_id

    results = search(keyword, lang="ko", country="kr")
    if not results:
        raise SystemExit(f"No Google Play app found for keyword: {keyword}")

    first = results[0]
    print(f"[app] {first.get('title')} ({first.get('appId')})")
    return first["appId"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", default="SKT")
    parser.add_argument("--app-id", default=None)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    app_id = resolve_app_id(args.keyword, args.app_id)
    rows, _ = reviews(
        app_id,
        lang="ko",
        country="kr",
        sort=Sort.NEWEST,
        count=args.count,
    )

    if not rows:
        raise SystemExit("No reviews returned.")

    print("\n[available fields]")
    print(sorted(rows[0].keys()))

    wanted = [
        "reviewId",
        "userName",
        "content",
        "score",
        "thumbsUpCount",
        "reviewCreatedVersion",
        "at",
        "appVersion",
    ]
    print("\n[sample reviews]")
    for row in rows:
        pprint({key: row.get(key) for key in wanted})


if __name__ == "__main__":
    main()
