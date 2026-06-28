import pyrootutils
project_root = pyrootutils.setup_root(search_from = __file__,
                                      indicator   = "README.md",
                                      pythonpath  = True)

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from google_play_scraper import Sort, reviews_all, search


class GooglePlay():
    def __init__(self, user_id: str, keyword: str):
        self.site     = 'google_play'
        self.keyword  = keyword
        self.base_dir = project_root / 'project' / 'user_data' / user_id / 'crawl_data' / keyword / datetime.today().strftime('%Y-%m-%dT%Hh%Mm%Ss')
        self.data     = pd.DataFrame(columns=[
            'url', 'site', 'document', 'documenttype',
            'postdate', 'likes', 'dislike', 'comment_cnt',
            'views', 'boardcategory', 'documentcategory',
            'source_id', 'rating', 'app_version', 'author'
        ])

    def crawl(self, output_path: str = None):
        # 키워드로 앱 검색 → 첫 번째 앱 ID 사용
        results = search(self.keyword, lang='ko', country='kr')
        if not results:
            print(f"[GooglePlay] '{self.keyword}' 검색 결과 없음")
            return self.data

        app_id  = results[0]['appId']
        app_url = f"https://play.google.com/store/apps/details?id={app_id}"
        print(f"[GooglePlay] 앱: {results[0]['title']} ({app_id})")

        google_reviews = reviews_all(
            app_id,
            sleep_milliseconds=0,
            lang='ko',
            country='kr',
            sort=Sort.NEWEST
        )

        df_google = pd.DataFrame(np.array(google_reviews), columns=['review'])
        df_google = df_google.join(pd.DataFrame(df_google.pop('review').tolist()))

        reviews_df = df_google[[
            'reviewId', 'userName', 'content', 'score',
            'thumbsUpCount', 'at', 'reviewCreatedVersion', 'appVersion'
        ]].copy()
        reviews_df.rename(
            columns={
                'reviewId': 'source_id',
                'userName': 'author',
                'content': 'document',
                'score': 'rating',
                'thumbsUpCount': 'likes',
                'at': 'postdate',
                'appVersion': 'app_version',
            },
            inplace=True,
        )
        reviews_df['app_version'] = reviews_df['app_version'].fillna(reviews_df['reviewCreatedVersion'])
        reviews_df.drop(columns=['reviewCreatedVersion'], inplace=True)

        result = pd.concat([self.data, reviews_df], axis=0)
        result['url']  = app_url
        result['site'] = self.site

        save_path = Path(output_path) if output_path else self.base_dir / 'A_googleplay.csv'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False)
        print(f"[GooglePlay] {len(reviews_df)}건 저장 → {save_path}")

        return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_id', required=True)
    parser.add_argument('--keyword', required=True)
    parser.add_argument('--output',  required=True)
    args = parser.parse_args()

    google = GooglePlay(user_id=args.user_id, keyword=args.keyword)
    google.crawl(output_path=args.output)
