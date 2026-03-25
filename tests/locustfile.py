"""
FastAPI /answer 엔드포인트 부하테스트
- non-streaming vs streaming 응답시간 비교
- 동시 사용자 처리 성능 측정
"""
from locust import HttpUser, task, between

USER_ID = "user01"
KEYWORD = "kt"
PAYLOAD = {"question": "KT 인터넷 속도가 느린데 해결 방법이 있나요?"}


class AnswerUser(HttpUser):
    wait_time = between(1, 2)  # 요청 사이 1~2초 대기

    @task(3)
    def answer_non_stream(self):
        """non-streaming 응답 (일반 JSON)"""
        with self.client.post(
            f"/answer/{USER_ID}/{KEYWORD}/false",
            json=PAYLOAD,
            name="/answer [non-stream]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"status={resp.status_code}")

    @task(1)
    def answer_stream(self):
        """streaming 응답 (text/event-stream) — 첫 청크까지 시간 측정"""
        with self.client.post(
            f"/answer/{USER_ID}/{KEYWORD}/true",
            json=PAYLOAD,
            name="/answer [stream]",
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                # 첫 청크만 받고 종료 (체감 응답속도 측정)
                for _ in resp.iter_content(chunk_size=1):
                    break
                resp.success()
            else:
                resp.failure(f"status={resp.status_code}")
