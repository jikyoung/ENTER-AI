import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import pyrootutils
project_root = pyrootutils.setup_root(search_from = __file__,
                                      indicator   = "README.md",
                                      pythonpath  = True)
import re
import scrapy
import numpy as np

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
from pathlib import Path
from datetime import datetime
from utils import CrawlerSettings
from w3lib.html import remove_tags
from scrapy_splash import SplashRequest


# 스파이더의 작업 디렉토리를 설정
dir_spiders = Path(__file__).parent.absolute()

class MiniGigiKoreaSpider(scrapy.Spider):
    name = "MiniGigiKoreaSpider"
    custom_settings = {
        **CrawlerSettings.get("SPLASH_LOCAL"),
        "CONCURRENT_REQUESTS": 2,
        "DOWNLOAD_DELAY": 1.5,
        "RANDOMIZE_DOWNLOAD_DELAY": True,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1,
        "AUTOTHROTTLE_MAX_DELAY": 10,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,
        "RETRY_HTTP_CODES": [429, 500, 502, 503, 504],
        "RETRY_TIMES": 3,
    }

    def __init__(self, user_id:str, keyword:str, max_pages: int | str = 3, since_date: str | None = None):
        super().__init__()
        self.site       = '미니기기코리아'
        self.keyword    = keyword
        self.max_pages  = int(max_pages)
        self.since_date = since_date
        self.start_urls = [f"https://meeco.kr/?_filter=search&act=&vid=&mid=ITplus&category=&search_target=title_content&search_keyword={self.keyword}"]
        self.base_dir = project_root / 'project' / 'user_data' / user_id / 'crawl_data' / keyword /datetime.today().strftime('%Y-%m-%dT%H:%M:%S')


    # Splash Lua 스크립트를 읽어옴
    lua_source = (
        dir_spiders / "MiniGigiKorea_main.lua"
    ).open("r", encoding='UTF-8').read()


    # 시작 요청을 생성하는 함수를 정의
    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url=url, headers=HEADERS, callback=self.parse)


    def parse(self, response):
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

        next_page_url = response.xpath('//div[@class="paging bBt"]/a[@class="pageNext"]/@href').get()

        if next_page_url:
            full_next_url = response.urljoin(next_page_url)
            parsed = urlparse(full_next_url)
            params = parse_qs(parsed.query)

            # pageNext URL에서 마지막 페이지 번호와 division 파라미터 동적 추출
            last_page = int(params.get('page', [1])[0])

            for i in range(1, min(last_page, self.max_pages) + 1):
                params['page'] = [str(i)]
                new_query = urlencode({k: v[0] for k, v in params.items()})
                new_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', new_query, ''))
                yield scrapy.Request(url=new_url, headers=HEADERS, callback=self.parse_info)
        else:
            # 페이지가 1개뿐인 경우
            yield scrapy.Request(url=response.url, headers=HEADERS, callback=self.parse_info,
                                 dont_filter=True)


    def parse_info(self, response):

        for href in response.xpath('//td[@class="title"]/a[@class="title_a title_moa"]/@href'):
            detail_url = href.get()
            post_url = response.urljoin(detail_url)

            yield scrapy.Request(url=post_url, headers=HEADERS, callback=self.parse_detail)


    def parse_detail(self, response):
        # 게시글 가져오기
        contents_elements = response.xpath('//*[@id="bBd"]/article/div[1]/div[2]')
        contents_text_list = contents_elements.xpath('.//text()').getall()
        # 각 텍스트를 공백을 제거하고 빈 문자열은 제외합니다.
        contents_text_list = [text.strip() for text in contents_text_list if text.strip()]
        # 정규표현식을 사용하여 공백, 개행 등을 모두 공백 하나로 치환합니다.
        document = re.sub(r'\s+', ' ', ' '.join(contents_text_list))

        # 게시 날짜
        date = response.xpath('//*[@id="bBd"]/article/header/ul[1]/li[3]').get()
        date = remove_tags(date)

        # 좋아요
        like = response.xpath('//div[@class="atc-vote-bts"]//span[@class="num"]').get()
        likes = remove_tags(like)

        #댓글수 가져오기
        comment_cnt = response.xpath('//span[@class="ptCl num cmt-cnt-ori"]/text()').get()

        # 조회수 가져오기
        view = response.xpath('//ul[@class="ldd-title-under"]/li/span[@class="num"]').get()
        views = remove_tags(view)

        #게시판 카테고리
        boardcategory = response.xpath('//header[@class="bBd-hd"]//a/text()').get()

        #게시글 카테고리
        documentcategory = response.xpath('//span[@class="atc-ctg"]//a/text()').get()

        MiniGigiKorea_data = dict(url              = response.url,
                                  site             = self.site,
                                  document         = document,
                                  documenttype     = np.nan,
                                  postdate         = date,
                                  likes            = likes,
                                  dislike          = np.nan,
                                  comment_cnt      = comment_cnt,
                                  views            = views,
                                  boardcategory    = boardcategory,
                                  documentcategory = documentcategory
                                  )

        yield MiniGigiKorea_data


if __name__ == '__main__':
    from scrapy.crawler import CrawlerProcess
    process = CrawlerProcess()
    process.crawl(MiniGigiKoreaSpider, keyword='기가지니', user_id='asdf1234')
    process.start()
