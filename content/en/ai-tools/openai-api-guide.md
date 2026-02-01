---
title: "OpenAI API Usage Guide"
description: "Complete guide to using OpenAI's API for your applications"
date: 2026-01-31
draft: false
tags: ["OpenAI", "API", "tutorial"]
categories: ["ai-tools"]
---

## Getting Started

**What you'll learn:**
- Setup OpenAI API in 5 minutes
- Use GPT-4, embeddings, and DALL-E
- Best practices and cost optimization

### 1. Get API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up / Log in
3. Navigate to API Keys
4. Create new secret key
5. **Save it** (you can't see it again!)

### 2. Install SDK

```bash
pip install openai
```

### 3. First API Call

```python
from openai import OpenAI

client = OpenAI(api_key="sk-...your-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
# Output: Hello! How can I assist you today?
```

## Chat Completions (GPT-4)

### Basic Chat

```python
def chat(user_message):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content

print(chat("Explain quantum computing in simple terms"))
```

### Streaming Responses

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

stream_chat("Write a poem about coding")
```

### Multi-turn Conversations

```python
conversation = [
    {"role": "system", "content": "You are a Python tutor."}
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

# Usage
print(chat_with_history("What are Python lists?"))
print(chat_with_history("Show me an example"))  # Remembers context!
```

### Function Calling

```python
import json

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
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
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
    tools=tools,
    tool_choice="auto"
)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    print(f"Function: {tool_call.function.name}")
    print(f"Arguments: {args}")
    # Call your actual weather function here
```

## Embeddings

**Use case:** Semantic search, similarity, clustering

```python
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Get embeddings
text1 = "Python programming"
text2 = "Coding in Python"
text3 = "Cooking recipes"

emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)

# Calculate similarity
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

print(f"text1 vs text2: {cosine_similarity(emb1, emb2):.3f}")  # High (similar)
print(f"text1 vs text3: {cosine_similarity(emb1, emb3):.3f}")  # Low (different)
```

### Semantic Search

```python
documents = [
    "Python is a programming language",
    "JavaScript is used for web development",
    "Machine learning uses algorithms",
    "Cooking requires recipes and ingredients"
]

# Embed all documents
doc_embeddings = [get_embedding(doc) for doc in documents]

def search(query):
    query_emb = get_embedding(query)
    
    # Find most similar
    similarities = [cosine_similarity(query_emb, doc_emb) 
                   for doc_emb in doc_embeddings]
    
    best_idx = similarities.index(max(similarities))
    return documents[best_idx]

print(search("coding language"))  # Returns: "Python is a programming language"
```

## Image Generation (DALL-E)

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

image_url = generate_image("A cat coding on a laptop, digital art")
print(f"Image URL: {image_url}")

# Download image
import requests
img_data = requests.get(image_url).content
with open('generated.png', 'wb') as f:
    f.write(img_data)
```

## Vision (GPT-4 Vision)

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
    "What does this chart show?"
)
print(result)
```

## Best Practices

### 1. Environment Variables

```python
import os
from openai import OpenAI

# Never hardcode API keys!
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

```bash
# .env file
OPENAI_API_KEY=sk-your-key-here
```

### 2. Error Handling

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
        return f"Error: {e}"
```

### 3. Cost Optimization

```python
# Use GPT-3.5 for simple tasks (20x cheaper)
model = "gpt-3.5-turbo" if simple_task else "gpt-4"

# Limit tokens
response = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=100  # Prevents runaway costs
)

# Track usage
print(f"Tokens used: {response.usage.total_tokens}")
```

### 4. Prompt Engineering

```python
# ❌ Vague
prompt = "Write about dogs"

# ✅ Specific
prompt = """
Write a 100-word article about Golden Retrievers.
Include: temperament, size, and care requirements.
Tone: Informative but friendly.
"""

# ✅ Use system message for consistent behavior
system = "You are a technical writer. Be concise and accurate."
```

## Complete Example: Chatbot

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
            return f"Error: {e}"
    
    def reset(self):
        self.messages = self.messages[:1]  # Keep system message

# Usage
bot = Chatbot("You are a helpful Python tutor.")

print(bot.chat("What are decorators?"))
print(bot.chat("Show me an example"))
bot.reset()
```

## Pricing (as of 2026)

| Model | Input (per 1K tokens) | Output (per 1K tokens) |
|-------|----------------------|------------------------|
| GPT-4 | $0.03 | $0.06 |
| GPT-3.5-turbo | $0.0015 | $0.002 |
| Embeddings (small) | $0.00002 | - |
| DALL-E 3 | $0.04 per image | - |

**Tip:** Start with GPT-3.5-turbo, upgrade to GPT-4 only when needed.

## Common Issues

**Rate limits:**
```python
import time
from openai import RateLimitError

def chat_with_retry(prompt, max_retries=3):
    for i in range(max_retries):
        try:
            return chat(prompt)
        except RateLimitError:
            if i < max_retries - 1:
                time.sleep(2 ** i)  # Exponential backoff
            else:
                raise
```

**Context length:**
```python
# GPT-4: 8K tokens (some models 32K/128K)
# If too long, summarize or chunk

if len(text) > 6000:  # rough estimate
    # Summarize first
    summary = chat(f"Summarize: {text}")
    response = chat(f"Based on: {summary}, answer: {question}")
```

### Rate Limiting
### Error Handling
### Cost Optimization

## Advanced Features

*To be completed*
