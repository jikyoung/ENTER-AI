import pyrootutils
project_root = pyrootutils.setup_root(search_from = __file__,
                                      indicator   = "README.md",
                                      pythonpath  = True)

import os
import time
import scrapy
import inspect
import subprocess
import pandas as pd
import importlib.util
from datetime import datetime


class CrawlManager():
    def __init__(self,
                 user_id: str,
                 keyword: str) -> None:

        self.user_id     = user_id
        self.keyword     = keyword.replace(" ","_")
        self.base_dir    = project_root / 'project' / 'user_data' / user_id / 'crawl_data' / self.keyword /datetime.today().strftime('%Y-%m-%dT%Hh%Mm%Ss')
        self.module_path = project_root / 'crawler' / 'crawler' / 'spiders'


    def run(self, except_spider:list[str]=[]):

        container_id = self.run_docker_splash()

        if self.base_dir.is_dir() == False:
            os.makedirs(self.base_dir, exist_ok=True)

        spider_commands = self.get_spider_commands(except_spider)
        self.run_scrapy(spider_commands)
        self.remove_docker_container(container_id)
        self.merge_csv_files()


    @staticmethod
    def run_docker_splash():

        # 이미 실행 중인 splash 컨테이너 확인
        check_command = "docker ps --filter ancestor=scrapinghub/splash --format '{{.ID}}'"
        existing = subprocess.check_output(check_command, shell=True).strip().decode("utf-8")

        if existing:
            return existing

        docker_command = "docker run -d -p 8050:8050 scrapinghub/splash"
        container_id = subprocess.check_output(docker_command, shell=True).strip().decode("utf-8")
        time.sleep(2)

        return container_id


    def run_scrapy(self, spider_commands):
        cwd = str(self.module_path.parent.parent)
        procs = [subprocess.Popen(cmd, shell=True, cwd=cwd) for cmd in spider_commands]
        for p in procs:
            p.wait()


    def remove_docker_container(self, container_id):

        stop_command = f"docker stop {container_id}"
        remove_command = f"docker rm {container_id}"

        subprocess.run(stop_command, shell=True)
        subprocess.run(remove_command, shell=True)


    def get_spider_commands(self, except_spider):

        spiders = self._get_spider_name(except_spider)
        commands = []
        for spider in spiders:
            cmd = f"scrapy crawl {spider} -a user_id={self.user_id} -a keyword={self.keyword} -o {self.base_dir}/A_{spider.lower().split('spider')[0]}.csv"
            commands.append(cmd)

        return commands


    def _get_spider_name(self, except_spider):
        import sys
        spiders_list = []

        crawler_root = str(self.module_path.parent.parent)

        # sys.modules에서 충돌하는 캐시를 임시 제거 후 복원
        cached = {k: v for k, v in sys.modules.items() if k == 'utils' or k.startswith('utils.')}
        for k in cached:
            del sys.modules[k]

        if crawler_root not in sys.path:
            sys.path.insert(0, crawler_root)

        for modules in self.module_path.rglob('./*.py'):
            if (modules.name == '__init__.py') or (modules.name == None):
                continue

            spec = importlib.util.spec_from_file_location(modules.name, self.module_path / modules.stem / modules.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and 'Spider' in name and issubclass(obj, scrapy.Spider):
                    spiders_list.append(name)

        # 캐시 복원
        sys.modules.update(cached)

        return [item for item in spiders_list if item not in except_spider]


    def get_crawl_data(self):
        crawl_keyword_dir = self.base_dir.parent

        if not crawl_keyword_dir.is_dir():
            return {'status': False}

        result = {}
        for crawl_dir in sorted(crawl_keyword_dir.iterdir()):
            merged_csv = crawl_dir / 'merged_data.csv'
            if merged_csv.is_file():
                try:
                    df = pd.read_csv(merged_csv)
                    result[crawl_dir.name] = len(df)
                except Exception:
                    continue

        if not result:
            return {'status': False}

        return result


    def merge_csv_files(self):
        crawl_dir_list = list(self.base_dir.parent.iterdir())

        for crawl_dir in crawl_dir_list:
            csv_files = [file for file in crawl_dir.glob('A_*.csv') if file.is_file()]

            if not csv_files:
                continue

            dataframes_list = []
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    if not df.empty:
                        dataframes_list.append(df)

                except pd.errors.EmptyDataError:
                    continue

            if not dataframes_list:
                continue

            merged_df = pd.concat(dataframes_list)
            print(f"{crawl_dir}에서 {len(dataframes_list)}개의 데이터프레임을 병합 중입니다.")

            if not merged_df.empty:
                merged_df.to_csv(crawl_dir / 'merged_data.csv', index=False)
                print(f"병합된 데이터를 {crawl_dir / 'merged_data.csv'}에 저장했습니다.")

                for csv_file in csv_files:
                    os.remove(csv_file)
                    
            else:
                print(f"병합된 데이터프레임이 비어 있어 {crawl_dir}에 CSV로 저장되지 않았습니다.")


if __name__ == "__main__":
    cm = CrawlManager('asdf1234', '에어팟')
    cm.run()

