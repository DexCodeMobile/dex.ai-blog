---
title: "Claude API 사용 가이드"
description: "지능형 애플리케이션을 위한 Anthropic의 Claude API 마스터하기"
date: 2026-01-31
draft: false
tags: ["Claude", "Anthropic", "API"]
categories: ["ai-tools"]
---

## 왜 Claude인가?

**Claude의 장점:**
- ✅ 200K 컨텍스트 윈도우 (엄청 큼!)
- ✅ 강력한 안전성 & 거부 메커니즘
- ✅ 지시사항 따르기 우수
- ✅ 비전 기능 (이미지 분석)
- ✅ 도구 사용 (함수 호출)

## 빠른 시작

### 1. API 키 받기

1. [console.anthropic.com](https://console.anthropic.com) 방문
2. 가입 / 로그인
3. API Keys로 이동
4. 새 키 생성
5. 안전하게 저장

### 2. SDK 설치

```bash
pip install anthropic
```

### 3. 첫 요청

```python
import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "양자 컴퓨팅을 간단히 설명해줘"}
    ]
)

print(message.content[0].text)
```

## Messages API

### 기본 채팅

```python
def chat(prompt):
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

response = chat("코딩에 관한 하이쿠를 써줘")
print(response)
```

### 시스템 메시지

```python
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="당신은 간결하고 실용적인 조언을 제공하는 Python 전문가입니다.",
    messages=[{
        "role": "user",
        "content": "CSV 파일을 어떻게 읽어?"
    }]
)

print(message.content[0].text)
```

### 다중 턴 대화

```python
conversation_history = []

def chat_with_history(user_message):
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=conversation_history
    )
    
    assistant_message = message.content[0].text
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    return assistant_message

# 사용법
print(chat_with_history("Python 데코레이터가 뭐야?"))
print(chat_with_history("실용적인 예제를 보여줘"))  # 맥락을 기억함
```

## 스트리밍 응답

```python
def stream_chat(prompt):
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

stream_chat("AI에 관한 이야기를 써줘")
```

### 완전한 스트리밍 예제

```python
from anthropic import Anthropic

def advanced_stream(prompt):
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        # 전체 응답 가져오기
        full_text = ""
        for text in stream.text_stream:
            full_text += text
            print(text, end="", flush=True)
        
        # 최종 메시지 접근
        final_message = stream.get_final_message()
        print(f"\n\n사용된 토큰: {final_message.usage.input_tokens + final_message.usage.output_tokens}")
        
        return full_text

result = advanced_stream("Python의 async/await를 설명해줘")
```

## 비전 (이미지 분석)

### 이미지 분석

```python
import base64

def analyze_image(image_path, question):
    # 이미지 읽기 및 인코딩
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }]
    )
    
    return message.content[0].text

# 사용법
result = analyze_image("chart.jpg", "이 차트에서 어떤 트렌드를 볼 수 있나요?")
print(result)
```

### 여러 이미지

```python
def compare_images(img1_path, img2_path):
    with open(img1_path, "rb") as f:
        img1_data = base64.b64encode(f.read()).decode()
    with open(img2_path, "rb") as f:
        img2_data = base64.b64encode(f.read()).decode()
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img1_data}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img2_data}},
                {"type": "text", "text": "이 두 이미지를 비교해줘"}
            ]
        }]
    )
    return message.content[0].text
```

## 도구 사용 (함수 호출)

```python
import json

tools = [{
    "name": "get_weather",
    "description": "위치의 현재 날씨 가져오기",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "도시 이름"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "온도 단위"
            }
        },
        "required": ["location"]
    }
}]

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "도쿄 날씨 어때?"}]
)

# Claude가 도구를 사용하려는지 확인
if message.stop_reason == "tool_use":
    tool_use = next(block for block in message.content if block.type == "tool_use")
    print(f"도구: {tool_use.name}")
    print(f"입력: {tool_use.input}")
    
    # 실제 함수 호출
    weather_data = {"temp": 22, "condition": "sunny"}
    
    # 결과를 Claude에게 다시 보내기
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "도쿄 날씨 어때?"},
            {"role": "assistant", "content": message.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(weather_data)
                }]
            }
        ]
    )
    print(response.content[0].text)
```

### 완전한 에이전트 예제

```python
def run_agent(user_query):
    tools = [{
        "name": "calculator",
        "description": "계산 수행",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "수학 표현식"}
            },
            "required": ["expression"]
        }
    }]
    
    messages = [{"role": "user", "content": user_query}]
    
    while True:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        if response.stop_reason == "end_turn":
            return response.content[0].text
        
        if response.stop_reason == "tool_use":
            # 도구 사용 찾기
            tool_use = next(block for block in response.content if block.type == "tool_use")
            
            # 도구 실행
            if tool_use.name == "calculator":
                result = eval(tool_use.input["expression"])  # 프로덕션에선 safe eval 사용!
                
                # 대화 계속
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(result)
                    }]
                })

print(run_agent("234 * 567은 얼마야?"))
```

## 모범 사례

### 1. 최신 모델 사용

```python
# ✅ 최고의 균형을 위해 Sonnet 사용
model = "claude-3-5-sonnet-20241022"  # 빠름 + 똑똑함

# 복잡한 작업에는 Opus 사용
model = "claude-3-opus-20240229"  # 가장 똑똑함, 느림

# 간단한 작업에는 Haiku 사용
model = "claude-3-haiku-20240307"  # 가장 빠름, 저렴
```

### 2. 프롬프트 엔지니어링

```python
# ❌ 모호함
prompt = "Python에 대해 알려줘"

# ✅ 구조화된 구체적 내용
prompt = """
Python 리스트 컴프리헨션을 설명해주세요.

포함사항:
1. 기본 문법
2. 간단한 예제 하나
3. 고급 예제 하나
4. 일반적인 함정

200단어 이내로 작성.
"""
```

### 3. 오류 처리

```python
from anthropic import APIError, RateLimitError
import time

def safe_chat(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 지수 백오프
            else:
                raise
        except APIError as e:
            return f"API 오류: {e}"
```

### 4. 토큰 관리

```python
def count_tokens_estimate(text):
    # 대략적인 추정: 1 토큰 ≈ 4 문자
    return len(text) // 4

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=500,  # 출력 제한
    messages=[{"role": "user", "content": prompt}]
)

# 실제 사용량 확인
print(f"입력: {message.usage.input_tokens}")
print(f"출력: {message.usage.output_tokens}")
```

## 완전한 챗봇

```python
from anthropic import Anthropic
import os

class ClaudeChatbot:
    def __init__(self, system_prompt=""):
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.system = system_prompt
        self.history = []
    
    def chat(self, user_message):
        self.history.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system=self.system,
                messages=self.history
            )
            
            response = message.content[0].text
            self.history.append({
                "role": "assistant",
                "content": response
            })
            
            return response
        except Exception as e:
            return f"오류: {e}"
    
    def reset(self):
        self.history = []

# 사용법
bot = ClaudeChatbot(system_prompt="당신은 도움이 되는 Python 튜터입니다.")

print(bot.chat("제너레이터가 뭐야?"))
print(bot.chat("예제를 보여줘"))
bot.reset()
```

## 가격 (2026년)

| 모델 | 입력 (백만 토큰당) | 출력 (백만 토큰당) |
|-------|------------------|------------------|
| Claude 3 Opus | $15.00 | $75.00 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Claude 3 Haiku | $0.25 | $1.25 |

**팁:** Sonnet으로 시작. 간단한 작업은 Haiku, 복잡한 추론은 Opus 사용.

## Claude vs OpenAI

| 기능 | Claude | GPT-4 |
|------|--------|-------|
| 컨텍스트 윈도우 | 200K | 128K |
| 지시사항 따르기 | 우수 | 매우 좋음 |
| 코딩 | 매우 좋음 | 우수 |
| 안전성 | 우수 | 매우 좋음 |
| 속도 (Sonnet vs GPT-4) | 더 빠름 | 느림 |

**Claude를 사용해야 할 때:**
- 큰 문서 (200K 컨텍스트)
- 안전이 중요한 애플리케이션
- 정확한 지시사항 따르기
- 이미지 분석

**GPT-4를 사용해야 할 때:**
- 복잡한 코딩 작업
- 수학 & 추론
- 더 넓은 도구 생태계
