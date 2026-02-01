---
title: "OpenAI API 사용 가이드"
description: "애플리케이션에 OpenAI API를 사용하는 완벽한 가이드"
date: 2026-01-31
draft: false
tags: ["OpenAI", "API", "tutorial"]
categories: ["ai-tools"]
---

## 시작하기

**배울 내용:**
- 5분 안에 OpenAI API 설정
- GPT-4, 임베딩, DALL-E 사용
- 모범 사례 및 비용 최적화

### 1. API 키 받기

1. [platform.openai.com](https://platform.openai.com) 방문
2. 가입 / 로그인
3. API Keys로 이동
4. 새 시크릿 키 생성
5. **저장** (다시 볼 수 없습니다!)

### 2. SDK 설치

```bash
pip install openai
```

### 3. 첫 API 호출

```python
from openai import OpenAI

client = OpenAI(api_key="sk-...your-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "안녕!"}]
)

print(response.choices[0].message.content)
# 출력: 안녕하세요! 무엇을 도와드릴까요?
```

## Chat Completions (GPT-4)

### 기본 채팅

```python
def chat(user_message):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "당신은 도움이 되는 어시스턴트입니다."},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content

print(chat("양자 컴퓨팅을 쉽게 설명해줘"))
```

### 스트리밍 응답

```python
def stream_chat(prompt):
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

stream_chat("코딩에 관한 시를 써줘")
```

### 다중 턴 대화

```python
conversation = [
    {"role": "system", "content": "당신은 Python 튜터입니다."}
]

def chat_with_history(user_msg):
    conversation.append({"role": "user", "content": user_msg})
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=conversation
    )
    
    assistant_msg = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": assistant_msg})
    
    return assistant_msg

# 사용법
print(chat_with_history("Python 리스트가 뭐야?"))
print(chat_with_history("예제를 보여줘"))  # 맥락을 기억합니다!
```

### 함수 호출

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
    messages=[{"role": "user", "content": "서울 날씨 어때?"}],
    tools=tools,
    tool_choice="auto"
)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    print(f"함수: {tool_call.function.name}")
    print(f"인자: {args}")
    # 실제 날씨 함수 호출
```

## 임베딩

**사용 사례:** 시맨틱 검색, 유사도, 클러스터링

```python
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# 임베딩 가져오기
text1 = "Python 프로그래밍"
text2 = "Python으로 코딩하기"
text3 = "요리 레시피"

emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)

# 유사도 계산
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

print(f"text1 vs text2: {cosine_similarity(emb1, emb2):.3f}")  # 높음 (유사)
print(f"text1 vs text3: {cosine_similarity(emb1, emb3):.3f}")  # 낮음 (다름)
```

### 시맨틱 검색

```python
documents = [
    "Python은 프로그래밍 언어입니다",
    "JavaScript는 웹 개발에 사용됩니다",
    "머신러닝은 알고리즘을 사용합니다",
    "요리는 레시피와 재료가 필요합니다"
]

# 모든 문서 임베딩
doc_embeddings = [get_embedding(doc) for doc in documents]

def search(query):
    query_emb = get_embedding(query)
    
    # 가장 유사한 것 찾기
    similarities = [cosine_similarity(query_emb, doc_emb) 
                   for doc_emb in doc_embeddings]
    
    best_idx = similarities.index(max(similarities))
    return documents[best_idx]

print(search("코딩 언어"))  # 반환: "Python은 프로그래밍 언어입니다"
```

## 이미지 생성 (DALL-E)

```python
def generate_image(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )
    return response.data[0].url

image_url = generate_image("노트북으로 코딩하는 고양이, 디지털 아트")
print(f"이미지 URL: {image_url}")

# 이미지 다운로드
import requests
img_data = requests.get(image_url).content
with open('generated.png', 'wb') as f:
    f.write(img_data)
```

## 비전 (GPT-4 Vision)

```python
def analyze_image(image_url, question):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }],
        max_tokens=300
    )
    return response.choices[0].message.content

result = analyze_image(
    "https://example.com/chart.png",
    "이 차트가 보여주는 것은 무엇인가요?"
)
print(result)
```

## 모범 사례

### 1. 환경 변수

```python
import os
from openai import OpenAI

# API 키를 하드코딩하지 마세요!
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

```bash
# .env 파일
OPENAI_API_KEY=sk-your-key-here
```

### 2. 오류 처리

```python
from openai import OpenAI, OpenAIError

def safe_chat(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        return f"오류: {e}"
```

### 3. 비용 최적화

```python
# 간단한 작업에는 GPT-3.5 사용 (20배 저렴)
model = "gpt-3.5-turbo" if simple_task else "gpt-4"

# 토큰 제한
response = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=100  # 비용 폭증 방지
)

# 사용량 추적
print(f"사용된 토큰: {response.usage.total_tokens}")
```

### 4. 프롬프트 엔지니어링

```python
# ❌ 모호함
prompt = "개에 대해 써줘"

# ✅ 구체적
prompt = """
골든 리트리버에 대한 100단어 기사를 작성하세요.
포함 사항: 기질, 크기, 관리 요구사항.
톤: 정보성이지만 친근하게.
"""

# ✅ 일관된 동작을 위해 시스템 메시지 사용
system = "당신은 기술 작가입니다. 간결하고 정확하게 작성하세요."
```

## 완전한 예제: 챗봇

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class Chatbot:
    def __init__(self, system_prompt):
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]
    
    def chat(self, user_message):
        self.messages.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=self.messages,
                temperature=0.7
            )
            
            assistant_message = response.choices[0].message.content
            self.messages.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
        except Exception as e:
            return f"오류: {e}"
    
    def reset(self):
        self.messages = self.messages[:1]  # 시스템 메시지 유지

# 사용법
bot = Chatbot("당신은 도움이 되는 Python 튜터입니다.")

print(bot.chat("데코레이터가 뭐야?"))
print(bot.chat("예제를 보여줘"))
bot.reset()
```

## 가격 (2026년 기준)

| 모델 | 입력 (1K 토큰당) | 출력 (1K 토큰당) |
|-------|-----------------|----------------|
| GPT-4 | $0.03 | $0.06 |
| GPT-3.5-turbo | $0.0015 | $0.002 |
| 임베딩 (small) | $0.00002 | - |
| DALL-E 3 | 이미지당 $0.04 | - |

**팁:** GPT-3.5-turbo로 시작하고, 필요할 때만 GPT-4로 업그레이드하세요.

## 일반적인 문제

**요청 제한:**
```python
import time
from openai import RateLimitError

def chat_with_retry(prompt, max_retries=3):
    for i in range(max_retries):
        try:
            return chat(prompt)
        except RateLimitError:
            if i < max_retries - 1:
                time.sleep(2 ** i)  # 지수 백오프
            else:
                raise
```

**컨텍스트 길이:**
```python
# GPT-4: 8K 토큰 (일부 모델 32K/128K)
# 너무 길면 요약하거나 청크로 나누기

if len(text) > 6000:  # 대략적인 추정
    # 먼저 요약
    summary = chat(f"요약: {text}")
    response = chat(f"기반: {summary}, 답변: {question}")
```
