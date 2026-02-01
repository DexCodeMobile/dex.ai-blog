---
title: "AI 코드 예제 모음"
description: "AI 개발을 위한 재사용 가능한 코드 스니펫과 패턴"
date: 2026-01-31
draft: false
tags: ["코드", "예제", "스니펫"]
categories: ["tutorials"]
---

## 빠른 참조

일반적인 AI 작업을 위한 복사-붙여넣기 가능한 코드입니다.

## API 기본

### OpenAI 간단한 채팅

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat(message):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content

print(chat("async/await 설명해줘"))
```

### Claude 간단한 채팅

```python
import anthropic
import os

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def chat(message):
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": message}]
    )
    return response.content[0].text

print(chat("데코레이터 설명해줘"))
```

### 스트리밍 응답

```python
# OpenAI 스트리밍
def stream_openai(prompt):
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

# Claude 스트리밍
def stream_claude(prompt):
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="")
```

## 임베딩 & 검색

### 임베딩 생성

```python
from openai import OpenAI

client = OpenAI()

def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

vector = embed("머신러닝 기초")
print(f"차원: {len(vector)}")  # 1536
```

### 코사인 유사도

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

vec1 = embed("Python 프로그래밍")
vec2 = embed("Python으로 코딩")
vec3 = embed("이탈리아 요리")

print(cosine_similarity(vec1, vec2))  # ~0.9 (유사)
print(cosine_similarity(vec1, vec3))  # ~0.3 (다름)
```

### 간단한 시맨틱 검색

```python
documents = [
    "Python은 ML에 좋습니다",
    "JavaScript는 웹 개발용",
    "SQL은 데이터베이스용"
]

# 모든 문서 임베딩
doc_vectors = [embed(doc) for doc in documents]

def search(query, k=2):
    query_vec = embed(query)
    scores = [cosine_similarity(query_vec, doc_vec) for doc_vec in doc_vectors]
    top_indices = np.argsort(scores)[-k:][::-1]
    return [(documents[i], scores[i]) for i in top_indices]

results = search("머신러닝")
for doc, score in results:
    print(f"{score:.3f}: {doc}")
```

## RAG 패턴

### 최소 RAG (프레임워크 없음)

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

class SimpleRAG:
    def __init__(self, documents):
        self.documents = documents
        self.vectors = [self._embed(doc) for doc in documents]
    
    def _embed(self, text):
        return client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
    
    def search(self, query, k=3):
        query_vec = self._embed(query)
        scores = [
            np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            for doc_vec in self.vectors
        ]
        top_k = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top_k]
    
    def ask(self, question):
        context = self.search(question)
        prompt = f\"\"\"컨텍스트:
{chr(10).join(context)}

질문: {question}
컨텍스트 기반 답변:\"\"\"
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# 사용법
rag = SimpleRAG([
    "Python은 고급 프로그래밍 언어입니다.",
    "AI와 데이터 과학에 인기가 많습니다.",
    "Python은 간단하고 읽기 쉬운 문법을 가지고 있습니다."
])

print(rag.ask("Python이 뭐에 좋아?"))
```

### LangChain RAG

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 로드 및 분할
text = "여기에 긴 문서..."
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

# 벡터 저장소 생성
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(chunks, embeddings)

# QA 체인 생성
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 질문하기
answer = qa.run("주요 주제가 뭐야?")
print(answer)
```

## 오류 처리

### 지수 백오프로 재시도

```python
import time
from openai import OpenAI, RateLimitError, APIError

client = OpenAI()

def api_call_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"요청 제한. {wait}초 대기...")
            time.sleep(wait)
        except APIError as e:
            print(f"API 오류: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)

# 사용법
result = api_call_with_retry(
    lambda: client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "안녕"}]
    )
)
```

### 타임아웃 핸들러

```python
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("작업 시간 초과")

def with_timeout(func, timeout_sec=30):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    try:
        result = func()
        signal.alarm(0)
        return result
    except TimeoutError:
        return None

# 사용법
result = with_timeout(lambda: slow_api_call(), timeout_sec=10)
```

## 프롬프트 템플릿

### 퓨샷 학습

```python
def few_shot_prompt(examples, new_input):
    prompt = "이 예제들로부터 배우세요:\\n\\n"
    for ex in examples:
        prompt += f"입력: {ex['input']}\\n출력: {ex['output']}\\n\\n"
    prompt += f"이제 이것을 하세요:\\n입력: {new_input}\\n출력:"
    return prompt

examples = [
    {"input": "개", "output": "동물"},
    {"input": "자동차", "output": "교통수단"},
]

prompt = few_shot_prompt(examples, "자전거")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
print(response.choices[0].message.content)  # "교통수단"
```

### 사고의 연쇄

```python
def cot_prompt(question):
    return f\"\"\"{question}

단계별로 생각해봅시다:
1. 먼저, 핵심 정보 파악
2. 그런 다음, 문제를 논리적으로 추론
3. 마지막으로, 답변 제공

생각하기:\"\"\"

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": cot_prompt("80의 25%는 얼마야?")}]
)
print(response.choices[0].message.content)
```

## 대화 메모리

### 간단한 메모리

```python
class SimpleChatbot:
    def __init__(self):
        self.history = []
    
    def chat(self, message):
        self.history.append({"role": "user", "content": message})
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=self.history
        )
        
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        
        return reply
    
    def reset(self):
        self.history = []

bot = SimpleChatbot()
print(bot.chat("내 이름은 철수야"))
print(bot.chat("내 이름이 뭐야?"))  # 기억함!
```

### 슬라이딩 윈도우 메모리

```python
class SlidingWindowChat:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = []
    
    def chat(self, message):
        self.history.append({"role": "user", "content": message})
        
        # 최근 N개 메시지만 유지
        recent = self.history[-self.window_size:]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=recent
        )
        
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        
        return reply
```

## 함수 호출

### OpenAI 함수 호출

```python
import json

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "현재 날씨 가져오기",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "파리 날씨 어때?"}],
    tools=tools
)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    print(f"함수: {tool_call.function.name}")
    print(f"인자: {args}")
```

## 비동기/배치 처리

### 비동기 요청

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def async_chat(message):
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content

async def process_batch(messages):
    tasks = [async_chat(msg) for msg in messages]
    return await asyncio.gather(*tasks)

# 사용법
messages = ["Python이 뭐야?", "JavaScript가 뭐야?", "Rust가 뭐야?"]
results = asyncio.run(process_batch(messages))
for msg, result in zip(messages, results):
    print(f"{msg}: {result[:50]}...")
```

## 캐싱

### 간단한 캐시

```python
import hashlib
import json

cache = {}

def cached_api_call(func):
    def wrapper(*args, **kwargs):
        # 캐시 키 생성
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        key = hashlib.md5(key_data.encode()).hexdigest()
        
        if key in cache:
            print("캐시 히트!")
            return cache[key]
        
        result = func(*args, **kwargs)
        cache[key] = result
        return result
    return wrapper

@cached_api_call
def chat(message):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content

# 첫 번째 호출: API 요청
print(chat("안녕"))
# 두 번째 호출: 캐시 히트!
print(chat("안녕"))
```
