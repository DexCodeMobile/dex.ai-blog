---
title: "Claude API Usage Guide"
description: "Master Anthropic's Claude API for intelligent applications"
date: 2026-01-31
draft: false
tags: ["Claude", "Anthropic", "API"]
categories: ["ai-tools"]
---

## Why Claude?

**Claude advantages:**
- ✅ 200K context window (huge!)
- ✅ Strong safety & refusal mechanisms
- ✅ Better at following instructions
- ✅ Vision capabilities (analyze images)
- ✅ Tool use (function calling)

## Quick Start

### 1. Get API Key

1. Visit [console.anthropic.com](https://console.anthropic.com)
2. Sign up / Log in
3. Go to API Keys
4. Generate new key
5. Save it securely

### 2. Install SDK

```bash
pip install anthropic
```

### 3. First Request

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
        {"role": "user", "content": "Explain quantum computing simply"}
    ]
)

print(message.content[0].text)
```

## Messages API

### Basic Chat

```python
def chat(prompt):
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

response = chat("Write a haiku about coding")
print(response)
```

### System Messages

```python
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="You are a Python expert who gives concise, practical advice.",
    messages=[{
        "role": "user",
        "content": "How do I read a CSV file?"
    }]
)

print(message.content[0].text)
```

### Multi-turn Conversations

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

# Usage
print(chat_with_history("What are Python decorators?"))
print(chat_with_history("Show me a practical example"))  # Remembers context
```

## Streaming Responses

```python
def stream_chat(prompt):
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

stream_chat("Write a story about AI")
```

### Complete Streaming Example

```python
from anthropic import Anthropic

def advanced_stream(prompt):
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        # Get full response
        full_text = ""
        for text in stream.text_stream:
            full_text += text
            print(text, end="", flush=True)
        
        # Access final message
        final_message = stream.get_final_message()
        print(f"\n\nTokens used: {final_message.usage.input_tokens + final_message.usage.output_tokens}")
        
        return full_text

result = advanced_stream("Explain async/await in Python")
```

## Vision (Image Analysis)

### Analyze Images

```python
import base64

def analyze_image(image_path, question):
    # Read and encode image
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

# Usage
result = analyze_image("chart.jpg", "What trends do you see in this chart?")
print(result)
```

### Multiple Images

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
                {"type": "text", "text": "Compare these two images"}
            ]
        }]
    )
    return message.content[0].text
```

## Tool Use (Function Calling)

```python
import json

tools = [{
    "name": "get_weather",
    "description": "Get current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        },
        "required": ["location"]
    }
}]

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}]
)

# Check if Claude wants to use a tool
if message.stop_reason == "tool_use":
    tool_use = next(block for block in message.content if block.type == "tool_use")
    print(f"Tool: {tool_use.name}")
    print(f"Input: {tool_use.input}")
    
    # Call your actual function here
    weather_data = {"temp": 22, "condition": "sunny"}
    
    # Send result back to Claude
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo?"},
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

### Complete Agent Example

```python
def run_agent(user_query):
    tools = [{
        "name": "calculator",
        "description": "Perform calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
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
            # Find tool use
            tool_use = next(block for block in response.content if block.type == "tool_use")
            
            # Execute tool
            if tool_use.name == "calculator":
                result = eval(tool_use.input["expression"])  # In production, use safe eval!
                
                # Continue conversation
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(result)
                    }]
                })

print(run_agent("What's 234 * 567?"))
```

## Best Practices

### 1. Use Latest Model

```python
# ✅ Use Sonnet for best balance
model = "claude-3-5-sonnet-20241022"  # Fast + Smart

# Use Opus for complex tasks
model = "claude-3-opus-20240229"  # Smartest, slower

# Use Haiku for simple tasks  
model = "claude-3-haiku-20240307"  # Fastest, cheapest
```

### 2. Prompt Engineering

```python
# ❌ Vague
prompt = "Tell me about Python"

# ✅ Specific with structure
prompt = """
Explain Python list comprehensions.

Include:
1. Basic syntax
2. One simple example
3. One advanced example
4. Common pitfalls

Keep it under 200 words.
"""
```

### 3. Error Handling

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
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
        except APIError as e:
            return f"API Error: {e}"
```

### 4. Token Management

```python
def count_tokens_estimate(text):
    # Rough estimate: 1 token ≈ 4 characters
    return len(text) // 4

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=500,  # Limit output
    messages=[{"role": "user", "content": prompt}]
)

# Check actual usage
print(f"Input: {message.usage.input_tokens}")
print(f"Output: {message.usage.output_tokens}")
```

## Complete Chatbot

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
            return f"Error: {e}"
    
    def reset(self):
        self.history = []

# Usage
bot = ClaudeChatbot(system_prompt="You are a helpful Python tutor.")

print(bot.chat("What are generators?"))
print(bot.chat("Show me an example"))
bot.reset()
```

## Pricing (2026)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Claude 3 Opus | $15.00 | $75.00 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Claude 3 Haiku | $0.25 | $1.25 |

**Tip:** Start with Sonnet. Use Haiku for simple tasks, Opus for complex reasoning.

## Claude vs OpenAI

| Feature | Claude | GPT-4 |
|---------|--------|-------|
| Context window | 200K | 128K |
| Instruction following | Excellent | Very Good |
| Coding | Very Good | Excellent |
| Safety | Excellent | Very Good |
| Speed (Sonnet vs GPT-4) | Faster | Slower |

**When to use Claude:**
- Large documents (200K context)
- Safety-critical applications
- Precise instruction following
- Image analysis

**When to use GPT-4:**
- Complex coding tasks
- Math & reasoning
- Broader tool ecosystem
