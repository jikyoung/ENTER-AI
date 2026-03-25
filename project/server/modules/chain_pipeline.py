import pyrootutils
pyrootutils.setup_root(search_from = __file__,
                       indicator   = "README.md",
                       pythonpath  = True)

import os
import asyncio
import pickle
from pathlib import Path
from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_core.prompts import format_document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from langchain_openai import OpenAIEmbeddings
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus.flowables import HRFlowable
from reportlab.platypus import BaseDocTemplate, PageTemplate, KeepTogether, Frame

from utils.mermaid_utils import *
from server.modules.set_template import SetTemplate
from server.modules.topic_pipeline import TopicPipeline


class ChainPipeline():
    
    def __init__(self, 
                 user_id:str, 
                 keyword:str):
        self.BASE_DIR       = Path(__file__).parent.parent.parent / 'user_data' / user_id 
        self.history_path   = self.BASE_DIR / 'history' / keyword / f'{keyword}.pkl'
        self.database_path  = self.BASE_DIR / 'database' / keyword
        self.memory         = None
        self.user_id        = user_id
        self.keyword        = keyword
        self.stream_history = None
        self.config         = SetTemplate(user_id).load('chatgpt','conversation')
        self.params         = SetTemplate(user_id)
    
    
    def load_history(self):
        if self.history_path.is_file():
            with open(self.history_path,'rb') as f:
                memory = pickle.load(f)
                
        else:
            memory = ConversationBufferMemory(
                return_messages = True, 
                output_key      = "answer", 
                input_key       = "question"
                )
            
        self.memory = memory
        
        return memory
    
    
    def save_history(self):
        if self.history_path.is_file() == False:
            os.makedirs(self.history_path.parent, exist_ok=True)
            
        with open(self.history_path,'wb') as f:
            pickle.dump(self.memory,f)


    def load_chain(self):
        
        if not self.memory:
            self.memory = self.load_history()

        memory_k = self.memory_load_k(5)
        
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory_k.load_memory_variables) | itemgetter("history"),
        )
        
        if self.config.system == '':
            answer_prompt = self.config.system_default
        else:
            answer_prompt = self.config.system

        #3. 벡터DB 존재 여부 확인
        if self.database_path.is_dir():
            vectorstore = FAISS.load_local(folder_path = self.database_path,
                                           embeddings  = OpenAIEmbeddings(),
                                           allow_dangerous_deserialization = True)
            retriever = vectorstore.as_retriever()

            retriever_from_llm = MultiQueryRetriever.from_llm(retriever = retriever,
                                                              llm       = ChatOpenAI(model       = self.params.load('chatgpt','params').model,
                                                                                     temperature = 0))

            retrieved_documents = {
                "docs": itemgetter("question") | retriever_from_llm,
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }

            if self.config.document == '':
                document_prompt = self.config.document_default
            else:
                document_prompt = self.config.document

            DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template=document_prompt)

            def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
                doc_strings = [format_document(doc, document_prompt) for doc in docs]
                return document_separator.join(doc_strings)

            ANSWER_PROMPT = ChatPromptTemplate.from_messages([
                ("system", answer_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ])

            final_inputs = {
                "context": lambda x: _combine_documents(x["docs"]),
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }
            answer = {
                "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(model=self.params.load('chatgpt','params').model),
            }

            final_chain = loaded_memory | retrieved_documents | answer

        else:
            # 벡터DB 없을 때 일반 ChatGPT로 폴백
            FALLBACK_PROMPT = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer in Korean."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ])
            answer = {
                "answer": FALLBACK_PROMPT | ChatOpenAI(model=self.params.load('chatgpt','params').model),
            }

            final_chain = loaded_memory | answer
        
        return final_chain
    
    
    def conversation_json(self):
        if not self.memory:
            self.memory = self.load_history()
            
        temp = self.memory.load_memory_variables({})['history']
        n = len(temp)//2
        conversation = {'n': n, 'conversation':[]}
        
        for i in range(n):
            conversation['conversation'].append({'history_id': f'{self.user_id}_{self.keyword}_{i}',
                                                 'question':temp[2*i].content,
                                                 'answer': temp[2*i+1].content
                                                 })
            
        return conversation


    def memory_load_k(self, k:int):
        if not self.memory:
            self.memory = self.load_history()
            
        temp = self.memory.load_memory_variables({})['history']
        #print(temp)
        N_con = len(temp)//2
        
        if k >= N_con:
            return self.memory
        else:
            memory_k = ConversationBufferMemory(return_messages = True, 
                                                output_key      = "answer", 
                                                input_key       = "question")
            for i in range(N_con-k, N_con):
                memory_k.save_context({"question": temp[2 * i].content},
                                      {"answer": temp[2 * i+1].content})
            
            return memory_k
        
        
    async def streaming(self, chain, query):
        self.stream_history=''
        
        async for stream in chain.astream(query):
            self.stream_history += stream['answer'].content
          
            yield stream['answer'].content
            
        self.memory.save_context({"question" : query['question']}, {"answer" : self.stream_history})
        self.save_history()

    
class ReportChainPipeline():

    def __init__(self,
                user_id:str,
                keyword:str,
                ):
        self.BASE_DIR          = Path(__file__).parent.parent.parent / 'user_data' / user_id
        self.database_path     = self.BASE_DIR / 'database' / keyword
        self.user_id           = user_id
        self.keyword           = keyword
        self.config            = SetTemplate(user_id)
        self.report_template   = ''
        self.document_template = ''

    async def _analyze_sentiment_all(self, vectorstore) -> dict:
        # 전체 문서 추출
        index_to_id = vectorstore.index_to_docstore_id
        all_docs = [vectorstore.docstore.search(index_to_id[i]) for i in range(len(index_to_id))]

        mini_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        async def classify(doc):
            text = doc.page_content[:300].replace('\x00', '').replace('\r', ' ').strip()
            prompt = f"다음 텍스트의 감성을 '긍정', '부정', '중립' 중 하나로만 답하세요.\n\n{text}"
            try:
                result = await mini_llm.ainvoke(prompt)
                return result.content.strip()
            except Exception:
                return '중립'

        labels = await asyncio.gather(*[classify(doc) for doc in all_docs])

        pos = labels.count('긍정')
        neg = labels.count('부정')
        neu = labels.count('중립')
        total = len(labels)

        # 감성별 대표 의견 (조회수 높은 순 상위 3개씩)
        tagged = list(zip(labels, all_docs))

        PROFANITY_PATTERNS = ['시발', '씨발', '개새', '병신', '좆', '보지', '자지', '쌍년', '새끼야', '미친놈', '존나', '지랄']

        def top_docs(sentiment, n=3):
            docs = [(d.metadata.get('views', 0), d.page_content[:200]) for l, d in tagged if l == sentiment]
            docs.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0, reverse=True)
            filtered = [content for _, content in docs if not any(p in content for p in PROFANITY_PATTERNS)]
            return filtered[:n]

        return {
            'total': total,
            'pos': pos, 'pos_pct': round(pos / total * 100, 1),
            'neg': neg, 'neg_pct': round(neg / total * 100, 1),
            'neu': neu, 'neu_pct': round(neu / total * 100, 1),
            'top_pos': top_docs('긍정'),
            'top_neg': top_docs('부정'),
        }

    async def load_chain(self):

        vectorstore = FAISS.load_local(folder_path = self.database_path,
                                       embeddings  = OpenAIEmbeddings(),
                                       allow_dangerous_deserialization = True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 30})

        retriever_from_llm = MultiQueryRetriever.from_llm(
                                                          retriever = retriever,
                                                          llm       = ChatOpenAI(model       = self.config.load('chatgpt','params').model,
                                                                                 temperature = 0,
                                                                                 ))

        config = self.config.params.load(self.BASE_DIR / 'template' / 'configs.yaml' ,addict=False)['chatgpt']['templates']['report']

        if config['prompt'] == '':
            self.report_template = config['prompt_default']
        else:
            self.report_template = config['prompt']

        if config['document'] == '':
            self.document_template = config['document_default']
        else:
            self.document_template = config['document']

        # 전체 문서 감성 분석 + 토픽 클러스터링 병렬 실행
        sentiment_task = self._analyze_sentiment_all(vectorstore)
        topic_task = asyncio.get_event_loop().run_in_executor(
            None,
            lambda: TopicPipeline(self.user_id, self.keyword).run(n_clusters=10)
        )
        sentiment, topics = await asyncio.gather(sentiment_task, topic_task)

        sentiment_summary = f"""
[감성 분석 결과 - 전체 {sentiment['total']}개 문서]
- 긍정: {sentiment['pos']}개 ({sentiment['pos_pct']}%)
- 부정: {sentiment['neg']}개 ({sentiment['neg_pct']}%)
- 중립: {sentiment['neu']}개 ({sentiment['neu_pct']}%)

[긍정 대표 의견]
{chr(10).join(f'- {d}' for d in sentiment['top_pos'])}

[부정 대표 의견]
{chr(10).join(f'- {d}' for d in sentiment['top_neg'])}
"""

        topic_summary = TopicPipeline(self.user_id, self.keyword).to_summary_text(topics)

        retrieved_documents = retriever_from_llm.invoke(self.report_template)

        DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template=self.document_template)

        MAX_DOC_CHARS = 1000

        def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n"):
            truncated_docs = []
            for doc in docs:
                if len(doc.page_content) > MAX_DOC_CHARS:
                    doc.page_content = doc.page_content[:MAX_DOC_CHARS] + '...'
                truncated_docs.append(doc)
            doc_strings = [format_document(doc, document_prompt) for doc in truncated_docs]
            return document_separator.join(doc_strings)

        context = sentiment_summary + "\n\n" + topic_summary + "\n\n[상세 의견 데이터]\n" + _combine_documents(retrieved_documents)
        ANSWER_PROMPT = self.report_template.format(context=context)
        answer_prompt = ChatPromptTemplate.from_messages([('system',"당신은 한국어로 보고서를 최대한 자세히 써야합니다"),
                                                          ('system',ANSWER_PROMPT),
                                                          ('human',"제목은 *, 소제목은 #, 하위 항목은 -로 시작하게 해줘")])

        result = ChatOpenAI(model=self.config.load('chatgpt','params').model).invoke(answer_prompt.format_prompt().to_messages()).content

        return self.to_pdf(result)
    
    
    def to_pdf(self,content):
        import platform
        if platform.system() == 'Windows':
            regular_font = "malgun.ttf"
            bold_font    = "Malgunbd.ttf"
        else:
            regular_font = str(Path.home() / "Library/Fonts/NanumGothic-Regular.ttf")
            bold_font    = str(Path.home() / "Library/Fonts/NanumGothic-Bold.ttf")
        pdfmetrics.registerFont(TTFont("맑은고딕", regular_font))
        pdfmetrics.registerFont(TTFont("맑은고딕B", bold_font))
        # text_frame = Frame(
        #     x1=2.54 * cm ,  # From left
        #     y1=2.54 * cm ,  # From bottom
        #     height=24.16 * cm,
        #     width=15.92 * cm,
        #     leftPadding=0 * cm,
        #     bottomPadding=0 * cm,
        #     rightPadding=0 * cm,
        #     topPadding=0 * cm,
        #     showBoundary=0,
        #     id='text_frame'
        # )

        content = convert_mm(content)
        # **텍스트** → <b>텍스트</b> 변환
        import re
        content = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', content)
        lines = content.split('\n')
        L=[]

        for line in lines:
            if line == '':
                continue
            
            if "mermaid;" in line:
                image_mm(line,L)
                continue
            
            if line[0]=='*':
                L.append(Paragraph(line.replace('*','')+'<br/><br/>',ParagraphStyle(name='fd',fontName='맑은고딕B',fontSize=21,leading=40)))
                L.append(HRFlowable(width='100%', thickness=0.2))
                continue
            
            elif line[0]=='#':
                L.append(Paragraph(line.replace('#',''),ParagraphStyle(name='fd',fontName='맑은고딕B',fontSize=15,leading=30)))
                
            else:
                L.append(Paragraph(line+'<br/><br/>',ParagraphStyle(name='fd',fontName='맑은고딕',fontSize=12,leading=20)))
                
        #L.append(KeepTogether([]))
        
        self.mermaid(content,L)
        
        # doc = BaseDocTemplate(str(self.BASE_DIR / 'Report.pdf'), pagesize=A4)
        # frontpage = PageTemplate(id='FrontPage',
        #                      frames=[text_frame]
        #             )
        # doc.addPageTemplates(frontpage)
        # doc.build(L)
        
        return str(self.BASE_DIR / 'Report.pdf')
        
    
    def mermaid(self,content,L):
        answer_prompt = ChatPromptTemplate.from_messages([('system',"다음 보고서에서 Review of Statistics의 각 항목의 내용을 기반으로 충분히 mermaid 코드를 만듭니다. "),
                                                          ('human',content)])
        result = ChatOpenAI(model=self.config.load('chatgpt','params').model).invoke(answer_prompt.format_prompt().to_messages()).content
            
        result = convert_mm(result)
        lines = result.split('\n')

        for line in lines:
            if "mermaid;" in line:
                image_mm(line,L)
                    
        text_frame = Frame(
            x1=2.54 * cm ,  # From left
            y1=2.54 * cm ,  # From bottom
            height=24.16 * cm,
            width=15.92 * cm,
            leftPadding=0 * cm,
            bottomPadding=0 * cm,
            rightPadding=0 * cm,
            topPadding=0 * cm,
            showBoundary=0,
            id='text_frame'
        )
        L.append(KeepTogether([]))
        
        doc = BaseDocTemplate(str(self.BASE_DIR / 'Report.pdf'), pagesize=A4)
        frontpage = PageTemplate(id     = 'FrontPage',
                                 frames = [text_frame]
                    )
        
        doc.addPageTemplates(frontpage)
        doc.build(L)