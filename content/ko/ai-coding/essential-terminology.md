---
title: "필수 AI 코딩 용어"
description: "모든 AI 개발자가 알아야 할 핵심 용어"
date: 2026-01-31
draft: false
tags: ["AI", "terminology", "fundamentals"]
categories: ["ai-coding"]
---

## 소개

AI 코딩의 세계에 오신 것을 환영합니다! 이제 막 시작하시는 분이라면 수많은 전문 용어에 압도당하실 수 있습니다. 걱정하지 마세요 - 이 포괄적인 가이드는 알아야 할 모든 필수 용어를 간단하고 초보자 친화적인 언어로 실용적인 예제와 함께 설명합니다.

이러한 기본 개념을 이해하면 다음과 같은 이점이 있습니다:
- 다른 AI 개발자들과 효과적으로 소통하기
- AI 애플리케이션을 구축할 때 정보에 입각한 결정 내리기
- 문서와 튜토리얼 이해하기
- 문제를 더 효율적으로 해결하기

시작해볼까요!

## 핵심 개념

### 거대 언어 모델 (LLM)

**정의:** 거대 언어 모델(LLM)은 방대한 양의 텍스트 데이터로 학습된 AI 시스템으로, 인간과 유사한 텍스트를 이해하고 생성할 수 있습니다. 매우 정교한 자동완성 기능으로 문맥, 의미를 이해하고 심지어 추론까지 할 수 있다고 생각하시면 됩니다.

**작동 원리:**
LLM은 수십억 개의 단어를 담은 책, 웹사이트, 기사 및 기타 텍스트 소스로 학습된 신경망(특히 트랜스포머 아키텍처)을 기반으로 합니다. 학습 과정에서 모델은 언어의 패턴, 문법, 사실 지식, 심지어 일부 추론 능력도 학습합니다.

**주요 예시:**
- **GPT-4** (OpenAI): 가장 강력한 모델 중 하나, 복잡한 작업에 적합
- **Claude** (Anthropic): 도움이 되고, 해롭지 않고, 정직한 것으로 유명
- **Gemini** (Google): 텍스트와 이미지를 포함한 멀티모달 기능
- **Llama** (Meta): 자체 하드웨어에서 실행할 수 있는 오픈소스 모델

**실제 사용 사례:**
```python
# 예제: LLM을 사용한 텍스트 요약
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

long_article = """
[긴 기사 텍스트...]
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "당신은 기사를 요약하는 도움이 되는 어시스턴트입니다."},
        {"role": "user", "content": f"이 기사를 요약해주세요:\n\n{long_article}"}
    ]
)

summary = response.choices[0].message.content
print(summary)
```

### 프롬프팅 (Prompting)

**정의:** 프롬프팅은 AI 모델에서 최상의 결과를 얻기 위해 지시사항("프롬프트"라고 함)을 작성하는 기술이자 과학입니다. 가장 효과적인 방법으로 질문하는 법을 배우는 것과 같습니다.

**중요한 이유:** 같은 AI 모델이라도 요청을 어떻게 표현하느냐에 따라 크게 다른 결과를 제공할 수 있습니다. 좋은 프롬프팅은 평범한 결과와 놀라운 결과의 차이를 만듭니다.

**프롬프트 엔지니어링 모범 사례:**

1. **구체적이고 명확하게**
   ```
   ❌ 나쁨: "코드 작성"
   ✅ 좋음: "숫자 리스트를 받아 평균을 반환하는 Python 함수 작성"
   ```

2. **맥락 제공**
   ```
   ❌ 나쁨: "이 버그 수정"
   ✅ 좋음: "상태가 변경될 때 업데이트되지 않는 React 컴포넌트가 있습니다. 
            코드는 다음과 같습니다: [코드]. 예상 동작은 [X]인데 
            [Y]를 수행하고 있습니다."
   ```

3. **예제 사용 (Few-Shot Learning)**
   ```
   프롬프트: "이 문장들을 질문으로 바꾸세요:
   
   예제 1:
   입력: 하늘은 파랗다.
   출력: 하늘은 무슨 색인가요?
   
   예제 2:
   입력: 개는 짖는다.
   출력: 개는 어떤 소리를 내나요?
   
   이제 이것을 바꿔보세요:
   입력: Python은 프로그래밍 언어입니다."
   ```

4. **역할 설정**
   ```python
   messages = [
       {"role": "system", "content": "당신은 10년 경력의 전문 Python 개발자입니다."},
       {"role": "user", "content": "이 데이터베이스 쿼리를 어떻게 최적화하나요?"}
   ]
   ```

**일반적인 프롬프트 패턴:**

- **사고의 연쇄(Chain of Thought):** 모델에게 단계별로 생각하도록 요청
  ```
  "이것을 단계별로 해결해봅시다:
  1. 먼저 문제를 식별합니다
  2. 그 다음 가능한 해결책을 고려합니다
  3. 마지막으로 최선의 해결책을 구현합니다"
  ```

- **템플릿 채우기:** 따라야 할 구조 제공
  ```
  "이 템플릿을 사용하여 제품 설명을 생성하세요:
  제품명: [X]
  주요 기능: [Y]
  대상 고객: [Z]
  독특한 판매 포인트: [W]"
  ```

### 컨텍스트 윈도우 (Context Window)

**정의:** 컨텍스트 윈도우는 AI 모델이 한 번에 "기억"하거나 처리할 수 있는 최대 텍스트 양(토큰으로 측정)입니다. 모델의 단기 기억이라고 생각하시면 됩니다.

**중요한 이유:** 모델에 보내는 모든 것(이전 메시지, 시스템 프롬프트, 모델의 응답)이 이 제한에 포함됩니다. 제한을 초과하면 가장 오래된 메시지가 잊혀집니다.

**컨텍스트 윈도우 크기 (2026년 기준):**
- GPT-4 Turbo: 128,000 토큰 (~300페이지)
- Claude 3: 200,000 토큰 (~500페이지)
- GPT-3.5: 16,385 토큰 (~50페이지)
- Gemini 1.5 Pro: 1,000,000 토큰 (~2,800페이지)

**컨텍스트 효과적으로 관리하기:**

1. **중요한 정보 우선순위 지정**
   ```python
   # 가장 최근 메시지만 유지
   conversation_history = messages[-10:]  # 최근 10개 메시지만
   ```

2. **오래된 컨텍스트 요약**
   ```python
   def manage_context(messages, max_messages=10):
       if len(messages) > max_messages:
           # 오래된 메시지 요약
           old_messages = messages[:-max_messages]
           summary = summarize_conversation(old_messages)
           
           # 요약 + 최근 메시지 유지
           return [
               {"role": "system", "content": f"이전 컨텍스트: {summary}"},
               *messages[-max_messages:]
           ]
       return messages
   ```

3. **외부 메모리 사용**
   ```python
   # 전체 히스토리를 데이터베이스에 저장
   # AI에는 최근 메시지만 전송
   db.save_conversation(full_history)
   recent_messages = full_history[-5:]
   response = client.chat.completions.create(messages=recent_messages)
   ```

**실용적인 예제:**
```python
# 토큰 사용량 모니터링
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages
)

tokens_used = response.usage.total_tokens
print(f"사용된 토큰: {tokens_used} / 128000")

if tokens_used > 100000:
    print("경고: 컨텍스트 제한에 접근하고 있습니다!")
```

### 토큰 (Tokens)

**정의:** 토큰은 AI 모델이 텍스트를 처리하는 데 사용하는 기본 단위입니다. 정확히 단어도 아니고 문자도 아닌, 그 중간쯤입니다.

**토큰 예제:**
- "안녕하세요" = 약 2-3 토큰
- "Hello, world!" = 4 토큰 ["Hello", ",", " world", "!"]
- "ChatGPT" = 2 토큰 ["Chat", "GPT"]
- "🚀" = 1-3 토큰 (이모티콘은 비쌀 수 있습니다!)

**중요한 이유:**
1. **비용:** API 제공업체는 토큰당 요금을 부과
   - 입력 토큰: 보내는 것
   - 출력 토큰: 모델이 생성하는 것
   
2. **속도:** 토큰이 많을수록 응답이 느림

3. **제한:** 컨텍스트 윈도우는 토큰으로 측정됨

**토큰 계산:**
```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

text = "안녕하세요, 오늘 어떻게 지내세요?"
tokens = count_tokens(text)
print(f"이 텍스트는 {tokens} 토큰을 사용합니다")  # 출력: ~10-12 토큰
```

**토큰 최적화 전략:**

1. **간결하게**
   ```
   ❌ "제가 정중하게 정보를 제공해주시길 요청드립니다"
   ✅ "정보를 제공해주세요"
   (약 5 토큰 절약)
   ```

2. **약어를 현명하게 사용**
   ```python
   # 때때로 약어는 더 많은 토큰 사용
   "AI" = 1-2 토큰
   "인공지능" = 2-3 토큰
   ```

3. **불필요한 서식 제거**
   ```
   ❌ "**중요:** 다음 사항에 유의하세요..."
   ✅ "중요: 다음 사항에 유의하세요..."
   ```

4. **가능한 경우 요청 배치 처리**
   ```python
   # 3번의 개별 호출 대신:
   ❌ 
   response1 = ask("'안녕'을 스페인어로 번역")
   response2 = ask("'안녕히'를 스페인어로 번역")
   response3 = ask("'감사합니다'를 스페인어로 번역")
   
   # 한 번의 호출로:
   ✅
   response = ask("""다음을 스페인어로 번역하세요:
   1. 안녕
   2. 안녕히
   3. 감사합니다""")
   ```

**비용 계산:**
```python
def estimate_cost(input_tokens, output_tokens, model="gpt-4"):
    # 2026년 기준 가격 (예시)
    prices = {
        "gpt-4": {"input": 0.03, "output": 0.06},  # 1K 토큰당
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
    }
    
    price = prices[model]
    input_cost = (input_tokens / 1000) * price["input"]
    output_cost = (output_tokens / 1000) * price["output"]
    
    return input_cost + output_cost

# 예제
cost = estimate_cost(input_tokens=500, output_tokens=1000, model="gpt-4")
print(f"예상 비용: ${cost:.4f}")  # $0.0750
```

## 고급 용어

### 파인튜닝 (Fine-tuning)

**정의:** 파인튜닝은 사전 학습된 AI 모델을 가져와 특정 데이터로 추가 학습시켜 특정 사용 사례에 더 적합하게 만드는 과정입니다.

**사용 시기:**
- 전문 도메인이 있는 경우 (의료, 법률, 기술)
- 일관된 형식이나 톤이 필요한 경우
- 프롬프트 길이를 줄이고 싶은 경우
- 특정 작업에서 더 나은 성능이 필요한 경우

**작동 방식:**
1. 기본 모델로 시작 (예: GPT-3.5)
2. 학습 데이터 준비 (입력과 원하는 출력의 예제)
3. 데이터로 모델 학습
4. 테스트 및 반복

**사용 사례 예제:**
```python
# 파인튜닝 데이터 준비
training_data = [
    {
        "messages": [
            {"role": "system", "content": "당신은 AcmeCorp의 고객 지원 상담원입니다."},
            {"role": "user", "content": "비밀번호를 재설정하려면 어떻게 하나요?"},
            {"role": "assistant", "content": "비밀번호를 재설정하려면:\n1. acmecorp.com/reset으로 이동\n2. 이메일 입력\n3. 받은 편지함에서 재설정 링크 확인\n4. 링크를 클릭하고 새 비밀번호 생성\n\n추가 도움이 필요하신가요? support@acmecorp.com으로 문의하세요"}
        ]
    },
    # ... 더 많은 예제
]

# 모델 파인튜닝 (간단화)
from openai import OpenAI
client = OpenAI()

# 학습 파일 업로드
file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

# 파인튜닝 작업 생성
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-3.5-turbo"
)
```

**비용 대비 이점:**
- **장점:** 더 나은 성능, 짧은 프롬프트, 대규모 비용 절감
- **단점:** 초기 비용, 데이터 준비 시간, 유지보수

### RAG (검색 증강 생성)

**정의:** RAG는 정보 검색(검색)과 텍스트 생성을 결합하는 기술입니다. AI의 학습 데이터만 의존하는 대신, RAG 시스템은 먼저 관련 정보를 검색한 다음 해당 정보를 사용하여 응답을 생성합니다.

**RAG를 사용하는 이유:**
- 최신 정보에 액세스 (모델은 학습 데이터만 알고 있음)
- 환각(거짓 정보 생성) 감소
- 비공개/독점 데이터 작업
- 출처 인용

**작동 방식:**
```
사용자 질문 → 문서 검색 → 관련 정보 찾기 → 
질문과 함께 AI에 전송 → AI가 검색된 정보를 기반으로 답변 생성
```

**간단한 RAG 예제:**
```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1단계: 문서 로드
loader = TextLoader("company_docs.txt")
documents = loader.load()

# 2단계: 청크로 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3단계: 임베딩 생성 및 벡터 데이터베이스에 저장
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4단계: 검색 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 5단계: 질문하기!
answer = qa_chain.run("반품 정책은 무엇인가요?")
print(answer)
```

**RAG 아키텍처 다이어그램:**
```
┌─────────────┐
│  귀하의 문서 │
└──────┬──────┘
       │ 1. 청크로 분할
       ▼
┌─────────────┐
│    청크     │
└──────┬──────┘
       │ 2. 임베딩 생성
       ▼
┌─────────────┐
│    벡터     │◄─── 3. 사용자 질문
│ 데이터베이스 │
└──────┬──────┘
       │ 4. 관련 청크 검색
       ▼
┌─────────────┐
│ 검색된 컨텍스트│
└──────┬──────┘
       │ 5. 질문과 결합
       ▼
┌─────────────┐
│     LLM     │
└──────┬──────┘
       │ 6. 답변 생성
       ▼
┌─────────────┐
│    답변     │
└─────────────┘
```

### 임베딩 (Embeddings)

**정의:** 임베딩은 의미를 포착하는 텍스트(또는 이미지, 오디오 등)의 숫자 표현입니다. 유사한 항목이 가까이 있는 다차원 공간의 좌표라고 생각하시면 됩니다.

**시각화 (2D로 단순화):**
```
        반려동물
         │
    고양이 •  • 강아지
         │
─────────┼─────────
         │
    사과 •  • 바나나
         │
        과일
```

**작동 방식:**
```python
from openai import OpenAI

client = OpenAI()

# 임베딩 생성
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="빠른 갈색 여우가 점프합니다"
)

embedding = response.data[0].embedding
print(len(embedding))  # 1536개의 숫자!
print(embedding[:5])   # [0.123, -0.456, 0.789, ...]
```

**사용 사례:**

1. **의미론적 검색**
   ```python
   def find_similar(query, documents):
       # 쿼리를 임베딩으로 변환
       query_embedding = create_embedding(query)
       
       # 모든 문서 임베딩과 비교
       similarities = []
       for doc in documents:
           doc_embedding = create_embedding(doc)
           similarity = cosine_similarity(query_embedding, doc_embedding)
           similarities.append((doc, similarity))
       
       # 가장 유사한 것 반환
       return sorted(similarities, key=lambda x: x[1], reverse=True)
   
   results = find_similar(
       query="파스타 요리 방법",
       documents=["파스타 레시피", "자동차 관리", "파스타 조리 팁"]
   )
   # 반환: "파스타 레시피"와 "파스타 조리 팁"
   ```

2. **군집화 및 분류**
   ```python
   # 유사한 고객 피드백 그룹화
   feedbacks = ["훌륭한 제품!", "배송이 느렸어요", "정말 좋아요!", "배송 지연"]
   embeddings = [create_embedding(f) for f in feedbacks]
   clusters = cluster(embeddings, n_clusters=2)
   # 클러스터 1: 긍정적 피드백
   # 클러스터 2: 배송 문제
   ```

3. **추천 시스템**
   ```python
   # 유사한 기사 찾기
   user_liked = "Python 프로그래밍 튜토리얼"
   all_articles = ["Java 튜토리얼", "Python 가이드", "요리 레시피"]
   
   recommendations = find_similar(user_liked, all_articles)
   # 추천: "Python 가이드", "Java 튜토리얼"
   ```

### 벡터 데이터베이스 (Vector Databases)

**정의:** 임베딩(벡터)을 저장하고 효율적으로 검색하도록 설계된 특수 데이터베이스입니다. 일반 데이터베이스는 정확한 일치에 적합하지만, 벡터 데이터베이스는 "유사한" 항목을 찾는 데 탁월합니다.

**인기 있는 벡터 데이터베이스:**

1. **Pinecone** - 관리형, 클라우드 네이티브
   ```python
   import pinecone
   
   pinecone.init(api_key="your-key")
   index = pinecone.Index("my-index")
   
   # 벡터 저장
   index.upsert([
       ("id1", [0.1, 0.2, 0.3, ...], {"text": "안녕하세요"}),
       ("id2", [0.4, 0.5, 0.6, ...], {"text": "안녕히 가세요"})
   ])
   
   # 검색
   results = index.query(
       vector=[0.15, 0.25, 0.35, ...],
       top_k=5
   )
   ```

2. **Chroma** - 오픈소스, 사용하기 쉬움
   ```python
   import chromadb
   
   client = chromadb.Client()
   collection = client.create_collection("my_docs")
   
   # 문서 추가 (임베딩 자동 생성)
   collection.add(
       documents=["이것은 문서 1입니다", "이것은 문서 2입니다"],
       ids=["id1", "id2"]
   )
   
   # 검색
   results = collection.query(
       query_texts=["유사한 문서 찾기"],
       n_results=2
   )
   ```

3. **FAISS** - Facebook의 라이브러리, 로컬 실행
   ```python
   import faiss
   import numpy as np
   
   # 인덱스 생성
   dimension = 1536  # 임베딩 크기
   index = faiss.IndexFlatL2(dimension)
   
   # 벡터 추가
   vectors = np.random.random((100, dimension)).astype('float32')
   index.add(vectors)
   
   # 검색
   query = np.random.random((1, dimension)).astype('float32')
   distances, indices = index.search(query, k=5)
   ```

4. **Qdrant** - 고급 필터링 기능을 갖춘 오픈소스
   ```python
   from qdrant_client import QdrantClient
   
   client = QdrantClient("localhost", port=6333)
   
   # 컬렉션 생성
   client.create_collection(
       collection_name="my_collection",
       vectors_config={"size": 1536, "distance": "Cosine"}
   )
   
   # 메타데이터와 함께 벡터 추가
   client.upsert(
       collection_name="my_collection",
       points=[
           {
               "id": 1,
               "vector": [0.1, 0.2, ...],
               "payload": {"category": "tech", "date": "2026-01-31"}
           }
       ]
   )
   
   # 필터를 사용한 검색
   results = client.search(
       collection_name="my_collection",
       query_vector=[0.1, 0.2, ...],
       query_filter={"category": "tech"},
       limit=5
   )
   ```

**벡터 데이터베이스 선택:**

| 데이터베이스 | 최적 용도 | 장점 | 단점 |
|----------|----------|------|------|
| **Pinecone** | 프로덕션 앱 | 완전 관리형, 확장 가능 | 비용, 벤더 종속 |
| **Chroma** | 프로토타입, 소규모 프로젝트 | 사용하기 쉬움, 무료 | 제한된 규모 |
| **FAISS** | 로컬 개발 | 빠름, API 불필요 | 기본적으로 영속성 없음 |
| **Qdrant** | 고급 필터링 요구사항 | 오픈소스, 유연함 | 자체 호스팅 필요 |
| **Weaviate** | 멀티모달 데이터 | 내장 ML, GraphQL | 복잡성 |

**성능 고려사항:**
```python
# 검색 성능 측정
import time

def benchmark_search(index, queries, k=5):
    start = time.time()
    
    for query in queries:
        results = index.search(query, k=k)
    
    elapsed = time.time() - start
    qps = len(queries) / elapsed
    
    print(f"초당 쿼리 수: {qps:.2f}")
    print(f"평균 지연 시간: {(elapsed/len(queries)*1000):.2f}ms")

# 예제 출력:
# 초당 쿼리 수: 1250.00
# 평균 지연 시간: 0.80ms
```

## 추가 중요 용어

### 온도 (Temperature)

**정의:** AI 응답의 무작위성을 제어하는 매개변수입니다. 낮은 온도 = 더 집중적이고 결정론적. 높은 온도 = 더 창의적이고 무작위적.

**범위:** 일반적으로 0에서 2 (때때로 0에서 1)

**예제:**
```python
# 온도 = 0 (결정론적, 매번 같은 답변)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "2+2는 무엇인가요?"}],
    temperature=0
)
# 출력: "2+2는 4입니다."

# 온도 = 0.7 (균형잡힌)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "창의적인 이야기 시작 부분을 작성하세요."}],
    temperature=0.7
)
# 출력: 다양하고, 창의적이지만 일관성 있음

# 온도 = 1.5 (매우 창의적/무작위적)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "창의적인 이야기 시작 부분을 작성하세요."}],
    temperature=1.5
)
# 출력: 매우 창의적, 잠재적으로 특이함
```

**다른 온도 사용 시기:**
- **0.0-0.3:** 사실적 질문, 코드 생성, 데이터 추출
- **0.5-0.8:** 창의적 글쓰기, 브레인스토밍, 일반 대화
- **0.9-2.0:** 매우 창의적인 작업, 이야기 쓰기, 시

### Top-p (핵 샘플링)

**정의:** 가장 가능성 있는 토큰으로 응답을 제한하는 온도의 대안입니다. 일관성을 희생하지 않고 다양성을 제어합니다.

**예제:**
```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "완성하세요: 고양이가 앉았다..."}],
    top_p=0.9  # 상위 90% 가능성이 있는 토큰만 고려
)
```

**모범 사례:** 온도 또는 top-p 중 하나만 사용하세요, 둘 다 사용하지 마세요!

### 시스템 프롬프트 (System Prompt)

**정의:** 대화 전반에 걸쳐 AI 모델의 동작, 개성, 제약을 설정하는 특수 메시지입니다.

**예제:**
```python
messages = [
    {
        "role": "system",
        "content": """당신은 초보자를 위한 도움이 되는 Python 튜터입니다.
        
        지침:
        - 개념을 간단한 용어로 설명하세요
        - 많은 예제를 사용하세요
        - 격려하고 인내심을 가지세요
        - 복잡한 주제를 단계로 나누세요
        - 계속 진행하기 전에 학생이 이해했는지 물어보세요"""
    },
    {"role": "user", "content": "변수란 무엇인가요?"}
]
```

### 환각 (Hallucination)

**정의:** AI 모델이 자신 있게 거짓 또는 무의미한 정보를 생성하는 경우입니다. AI의 가장 큰 과제 중 하나입니다.

**예제:**
```
사용자: "스티븐 킹이 2025년에 출판한 책은 무엇인가요?"
AI: "스티븐 킹은 2025년에 'Dark Shadows'와 'The Haunting Hour'를 출판했습니다."
[환각 - 이 책들은 존재하지 않습니다!]
```

**환각을 줄이는 방법:**
1. RAG를 사용하여 사실에 근거한 응답 생성
2. 출처/인용 요청
3. 사실적 작업에 대해 낮은 온도 사용
4. 시스템 프롬프트를 사용하여 정확성 강조
5. 검증 단계 구현

```python
# 예제: 출처 요청
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "항상 출처를 인용하세요. 확실하지 않으면 그렇게 말하세요."},
        {"role": "user", "content": "프랑스의 수도는 어디인가요?"}
    ]
)
```

## 결론

이제 AI 코딩을 위한 필수 용어를 배웠습니다! 간단한 요약은 다음과 같습니다:

**기본 개념:**
- **LLM:** 텍스트를 이해하고 생성하는 AI 두뇌
- **프롬프팅:** AI와 소통하는 방법
- **컨텍스트 윈도우:** AI의 단기 기억 제한
- **토큰:** 텍스트를 측정하는 데 사용되는 단위

**고급 기술:**
- **파인튜닝:** 필요에 맞게 AI 모델 사용자 정의
- **RAG:** 정확한 답변을 위해 검색과 생성 결합
- **임베딩:** 의미의 숫자 표현
- **벡터 데이터베이스:** 임베딩을 위한 특수 저장소

**주요 매개변수:**
- **온도:** 창의성 대 일관성 제어
- **Top-p:** 다양성을 제어하는 대안적 방법
- **시스템 프롬프트:** AI의 동작 설정

**일반적인 과제:**
- **환각:** AI가 거짓 정보를 만들어낼 때

## 다음 단계

이제 용어를 이해했으므로 다음을 수행할 준비가 되었습니다:
1. 첫 번째 AI 애플리케이션 구축 시작
2. 다양한 모델과 매개변수 실험
3. 사용 사례에 RAG 구현
4. 고급 프롬프트 엔지니어링 기술 학습

기억하세요: 배우는 가장 좋은 방법은 실천하는 것입니다. 간단한 프로젝트로 시작하여 이러한 개념에 더 익숙해지면서 점차 복잡성을 높이세요.

## 추가 리소스

- OpenAI 문서: https://platform.openai.com/docs
- Anthropic Claude 문서: https://docs.anthropic.com
- LangChain 가이드: https://python.langchain.com
- 프롬프트 엔지니어링 가이드: https://www.promptingguide.ai

즐거운 코딩 되세요! 🚀
