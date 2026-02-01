---
title: "RAG 시스템 구축: 완벽한 가이드"
description: "검색 증강 생성(RAG) 시스템을 구축하는 방법"
date: 2026-01-31
draft: false
tags: ["RAG", "임베딩", "벡터-데이터베이스"]
categories: ["ai-tools"]
---

## RAG란 무엇인가요?

**RAG (검색 증강 생성)** = 검색 + AI 생성

**왜 RAG가 필요한가요?**
- ✅ LLM은 최신/비공개 데이터를 모름
- ✅ RAG는 당신의 데이터를 컨텍스트에 추가
- ✅ 더 정확하고 사실적인 응답
- ✅ 출처 인용 가능

**작동 방식:**
1. **인덱싱**: 문서 분할 → 임베딩 → 벡터 DB에 저장
2. **쿼리**: 사용자 질문 → 관련 문서 찾기
3. **생성**: LLM이 검색된 문서를 사용해 답변

## 빠른 시작 (5분)

```bash
pip install langchain langchain-openai chromadb
```

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. 문서 준비
documents = ["Python은 프로그래밍 언어입니다", "JavaScript는 웹 개발용입니다"]

# 2. 텍스트 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = splitter.create_documents(documents)

# 3. 벡터 저장소 생성
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# 4. QA 체인 생성
llm = ChatOpenAI()
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# 5. 질문하기
print(qa.run("Python이 뭐야?"))
```

## 문서 처리

### 1. 텍스트 분할

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 청크당 최대 문자 수
    chunk_overlap=200,    # 청크 간 중첩
    separators=["\\n\\n", "\\n", " ", ""]  # 분할 우선순위
)

chunks = splitter.split_text(long_text)
print(f"{len(chunks)}개의 청크 생성")
```

**청킹 전략:**
```python
# 작은 청크 (정확도 향상)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 큰 청크 (더 많은 컨텍스트)
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# 의미적 분할 (문단 단위)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    separators=["\\n\\n", "\\n", ". ", " "]
)
```

### 2. 문서 로딩

```python
from langchain.document_loaders import (
    TextLoader, PyPDFLoader, UnstructuredMarkdownLoader, 
    WebBaseLoader, DirectoryLoader
)

# 텍스트 파일 로드
loader = TextLoader("data.txt")
docs = loader.load()

# PDF 로드
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 웹에서 로드
loader = WebBaseLoader("https://example.com/article")
docs = loader.load()

# 디렉토리의 모든 파일 로드
loader = DirectoryLoader("./docs", glob="**/*.md")
docs = loader.load()
```

### 3. 메타데이터 추출

```python
from langchain.schema import Document

# 메타데이터 추가
docs_with_metadata = []
for i, chunk in enumerate(chunks):
    doc = Document(
        page_content=chunk,
        metadata={
            "source": "manual.pdf",
            "page": i // 5,  # 페이지 추정
            "chunk_id": i
        }
    )
    docs_with_metadata.append(doc)
```

## 임베딩

### OpenAI 임베딩

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # 또는 "text-embedding-3-large"
)

# 단일 텍스트 임베딩
vector = embeddings.embed_query("안녕하세요")
print(f"벡터 차원: {len(vector)}")  # 1536

# 여러 텍스트 임베딩
vectors = embeddings.embed_documents(["텍스트 1", "텍스트 2"])
```

### 다른 임베딩 모델

```python
# HuggingFace
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Cohere
from langchain.embeddings import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")
```

## 벡터 데이터베이스

### Chroma (로컬)

```python
from langchain.vectorstores import Chroma

# 생성
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # 디스크에 저장
)

# 기존 것 로드
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 검색
results = vectorstore.similarity_search("쿼리", k=3)
```

### FAISS (빠름)

```python
from langchain.vectorstores import FAISS

# 생성
vectorstore = FAISS.from_documents(chunks, embeddings)

# 저장/로드
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings)

# 점수와 함께 검색
results = vectorstore.similarity_search_with_score("쿼리", k=3)
for doc, score in results:
    print(f"점수: {score}, 텍스트: {doc.page_content[:100]}")
```

### Pinecone (클라우드)

```python
from langchain.vectorstores import Pinecone
import pinecone

# 초기화
pinecone.init(api_key="your-key", environment="us-west1-gcp")
index_name = "my-index"

# 생성
vectorstore = Pinecone.from_documents(chunks, embeddings, index_name=index_name)

# 검색
results = vectorstore.similarity_search("쿼리", k=3)
```

## 검색 전략

### 1. 유사도 검색

```python
# 기본 유사도
results = vectorstore.similarity_search("Python 프로그래밍", k=3)

# 점수와 함께
results = vectorstore.similarity_search_with_score("Python", k=3)
for doc, score in results:
    print(f"점수: {score:.3f}")
```

### 2. MMR (최대 한계 관련성)

```python
# 다양한 결과 (중복 감소)
results = vectorstore.max_marginal_relevance_search(
    "Python 프로그래밍",
    k=5,
    fetch_k=20,  # 더 많이 가져온 다음 필터링
    lambda_mult=0.5  # 0=다양함, 1=유사함
)
```

### 3. 메타데이터 필터링

```python
# 메타데이터로 필터링
results = vectorstore.similarity_search(
    "Python",
    k=3,
    filter={"source": "tutorial.pdf", "page": 1}
)
```

### 4. 다중 쿼리 검색

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# 자동으로 여러 쿼리 생성
results = retriever.get_relevant_documents("Python 설명해줘")
```

## 생성

### 기본 QA 체인

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # stuff, map_reduce, refine, map_rerank
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

answer = qa_chain.run("Python은 무엇에 사용되나요?")
```

### 대화형 RAG

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# 다중 턴 대화
print(qa({"question": "문서에 무엇이 있나요?"})["answer"])
print(qa({"question": "그것에 대해 더 말해줘"})["answer"])  # 메모리 사용!
```

### 커스텀 프롬프트

```python
from langchain.prompts import PromptTemplate

template = \"\"\"다음 컨텍스트를 사용하여 질문에 답하세요.
모르면 "모르겠습니다"라고 말하세요.

컨텍스트: {context}

질문: {question}

답변:\"\"\"

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)
```

## 완전한 RAG 시스템

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader, TextLoader

class RAGSystem:
    def __init__(self, docs_path):
        # 문서 로드
        loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        # 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        
        # 벡터 저장소 생성
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(chunks, embeddings)
        
        # LLM 생성
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # 메모리 생성
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 체인 생성
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            return_source_documents=True
        )
    
    def ask(self, question):
        result = self.qa({"question": question})
        
        # 출처 출력
        print("\\n출처:")
        for doc in result["source_documents"]:
            print(f"- {doc.metadata.get('source', '알 수 없음')}")
        
        return result["answer"]
    
    def reset(self):
        self.memory.clear()

# 사용법
rag = RAGSystem("./docs")
print(rag.ask("주요 주제가 뭐야?"))
print(rag.ask("자세히 설명해줘"))
rag.reset()
```

## 모범 사례

### 1. 청킹

```python
# ✅ 좋은 청크 크기
chunk_size = 1000  # 1-2 문단

# ✅ 중첩으로 컨텍스트 손실 방지
chunk_overlap = 200  # chunk_size의 20%

# ❌ 너무 작음: 컨텍스트 손실
# ❌ 너무 큼: 관련성 희석
```

### 2. 검색

```python
# ✅ 필요한 것보다 더 많이 검색
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ✅ 다양성을 위해 MMR 사용
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)
```

### 3. 프롬프트 엔지니어링

```python
# ✅ 명확한 지시사항
prompt = \"\"\"컨텍스트만을 기반으로 답변하세요.
확실하지 않으면 "충분한 정보가 없습니다"라고 말하세요.

컨텍스트: {context}
질문: {question}
답변:\"\"\"

# ❌ 모호함
prompt = "답변: {question}"
```

### 4. 평가

```python
# 테스트 질문
test_cases = [
    {"question": "X가 뭐야?", "expected": "X는..."},
    {"question": "Y를 어떻게 해?", "expected": "Y를 하려면..."}
]

for case in test_cases:
    answer = qa_chain.run(case["question"])
    print(f"Q: {case['question']}")
    print(f"A: {answer}")
    print(f"기대값: {case['expected']}\\n")
```

## 일반적인 문제

**낮은 관련성:**
```python
# 해결책: 더 나은 청킹, 더 많은 문서, k 조정
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

**느린 성능:**
```python
# 해결책: Chroma 대신 FAISS 사용
vectorstore = FAISS.from_documents(chunks, embeddings)
```

**높은 비용:**
```python
# 해결책: 더 작은 임베딩, 더 적은 청크 사용
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 더 저렴
```

## 고급: 재순위

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 기본 검색기
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 압축기 (재순위 지정자)
compressor = LLMChainExtractor.from_llm(llm)

# 압축된 검색기
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 재순위 후 가장 관련성 높은 것 반환
docs = retriever.get_relevant_documents("쿼리")
```
