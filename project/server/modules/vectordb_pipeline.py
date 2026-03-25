import shutil
import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader

class VectorPipeline():
    BASE_DIR = Path(__file__).parent.parent.parent / 'user_data'

    @classmethod
    def get_existing_urls(cls, user_id: str, keyword: str) -> set:
        """기존 FAISS에 저장된 문서의 url 집합 반환"""
        kwd_db_path = cls.BASE_DIR / user_id / 'database' / keyword
        if not kwd_db_path.is_dir():
            return set()

        from langchain_openai import OpenAIEmbeddings
        vectorstore = FAISS.load_local(
            folder_path=kwd_db_path,
            embeddings=OpenAIEmbeddings(),
            allow_dangerous_deserialization=True,
        )
        index_to_id = vectorstore.index_to_docstore_id
        urls = set()
        for i in range(len(index_to_id)):
            doc = vectorstore.docstore.search(index_to_id[i])
            url = doc.metadata.get('url', '')
            if url:
                urls.add(url)
        return urls

    @classmethod
    def embedding_and_store(cls,
                            data:pd.DataFrame,
                            user_id:str,
                            keyword:str,
                            embedding,
                            target_col:str='document'):

        kwd_db_path = cls.BASE_DIR / user_id / 'database' / f'{keyword}'

        data = data.dropna(subset=[target_col])
        loader = DataFrameLoader(data_frame          = data,
                                 page_content_column = target_col)
        docs = loader.load()
        vectorstore = FAISS.from_documents(documents = docs,
                                           embedding = embedding)

        if kwd_db_path.is_dir():
            shutil.rmtree(str(kwd_db_path))

        vectorstore.save_local(folder_path=kwd_db_path)

    @classmethod
    def merge_into_store(cls,
                         data: pd.DataFrame,
                         user_id: str,
                         keyword: str,
                         embedding,
                         target_col: str = 'document'):
        """기존 FAISS에 새 문서만 추가 (증분 업데이트)"""
        kwd_db_path = cls.BASE_DIR / user_id / 'database' / keyword

        data = data.dropna(subset=[target_col])
        data = data[data[target_col].str.strip() != '']

        if not kwd_db_path.is_dir():
            cls.embedding_and_store(data, user_id, keyword, embedding, target_col)
            return len(data)

        existing_urls = cls.get_existing_urls(user_id, keyword)
        new_data = data[~data['url'].isin(existing_urls)].reset_index(drop=True)

        if new_data.empty:
            return 0

        loader = DataFrameLoader(data_frame=new_data, page_content_column=target_col)
        new_docs = loader.load()

        vectorstore = FAISS.load_local(
            folder_path=kwd_db_path,
            embeddings=embedding,
            allow_dangerous_deserialization=True,
        )
        new_vectorstore = FAISS.from_documents(documents=new_docs, embedding=embedding)
        vectorstore.merge_from(new_vectorstore)
        vectorstore.save_local(folder_path=kwd_db_path)

        return len(new_data)
            
            
    @classmethod
    def delete_store_by_keyword(cls, 
                                user_id:str, 
                                keyword:str):
        
        database_path = cls.BASE_DIR / user_id / 'database' / keyword
        history_path  = cls.BASE_DIR / user_id / 'history' / keyword
            
        if (history_path.is_dir() == False) or (database_path.is_dir() == False):
            
            return {"status" : "abnormal delete request"}
        
        else:
            shutil.rmtree(str(database_path))
            shutil.rmtree(str(history_path))
            
            return {"status" : "delete success"}
        
        
if __name__ == "__main__":
    # VectorPipeline.delete_store_by_keyword('asdf1234', 'cafecopy')
    print(VectorPipeline.BASE_DIR)