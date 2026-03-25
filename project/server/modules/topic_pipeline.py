import numpy as np
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class TopicPipeline:

    def __init__(self, user_id: str, keyword: str):
        self.database_path = Path(__file__).parent.parent.parent / 'user_data' / user_id / 'database' / keyword
        self.user_id = user_id
        self.keyword = keyword

    def _load_docs(self, vectorstore):
        index_to_id = vectorstore.index_to_docstore_id
        docs = [vectorstore.docstore.search(index_to_id[i]) for i in range(len(index_to_id))]
        return docs

    def _extract_vectors(self, vectorstore):
        return vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)

    def _name_cluster(self, docs: list[str]) -> str:
        sample = "\n".join(f"- {d[:150]}" for d in docs[:5])
        prompt = f"다음 사용자 의견들의 공통 주제를 한국어로 5~10자 이내로 간결하게 표현하세요. 예: '가격/요금 불만', '보안 우려', 'CS 서비스 품질'\n\n{sample}"
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        result = llm.invoke(prompt)
        return result.content.strip()

    def run(self, n_clusters: int = 10) -> list[dict]:
        vectorstore = FAISS.load_local(
            folder_path=self.database_path,
            embeddings=OpenAIEmbeddings(),
            allow_dangerous_deserialization=True,
        )

        docs = self._load_docs(vectorstore)
        vectors = self._extract_vectors(vectorstore)
        vectors_norm = normalize(vectors)

        n_clusters = min(n_clusters, len(docs))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(vectors_norm)

        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(docs[idx])

        results = []
        for label, cluster_docs in sorted(clusters.items(), key=lambda x: -len(x[1])):
            contents = [d.page_content.strip() for d in cluster_docs if d.page_content.strip()]
            if not contents:
                continue
            topic_name = self._name_cluster(contents)
            if '의견이 없' in topic_name or len(topic_name) > 30:
                continue
            results.append({
                'topic': topic_name,
                'count': len(cluster_docs),
                'pct': round(len(cluster_docs) / len(docs) * 100, 1),
                'samples': contents[:3],
            })

        return results

    def to_summary_text(self, topics: list[dict]) -> str:
        lines = [f"[토픽 클러스터링 결과 - 상위 {len(topics)}개 주제]"]
        for i, t in enumerate(topics, 1):
            lines.append(f"{i}. {t['topic']}: {t['count']}건 ({t['pct']}%)")
        return "\n".join(lines)
