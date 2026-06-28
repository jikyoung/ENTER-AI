"""Inspect whether crawl output preserves per-post URLs.

Run this against an existing or newly crawled CSV. If a community source has
many rows but only one unique URL, the crawler is still saving the search/list
URL instead of the individual post URL.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--sample", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if "url" not in df.columns:
        raise SystemExit("CSV has no 'url' column.")

    print(f"[file] {args.csv_path}")
    print(f"rows: {len(df):,}")
    print(f"unique_url: {df['url'].astype(str).nunique():,}")

    if "site" in df.columns:
        print("\n[by site]")
        grouped = df.groupby("site", dropna=False)["url"].agg(["count", "nunique"])
        print(grouped.to_string())

    url_series = df["url"].astype(str)
    search_like = url_series.str.contains(
        r"search|groupSearches|\?_filter=search|q=|keyword=",
        case=False,
        na=False,
        regex=True,
    )
    print(f"\nsearch/list URL ratio: {search_like.mean():.1%}")

    print("\n[top URLs]")
    print(url_series.value_counts().head(args.sample).to_string())

    cols = [col for col in ["site", "url", "document", "postdate", "boardcategory"] if col in df.columns]
    print("\n[samples]")
    print(df[cols].head(args.sample).to_string(max_colwidth=120))


if __name__ == "__main__":
    main()
