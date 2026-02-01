---
title: "AI Code Examples Collection"
description: "Reusable code snippets and patterns for AI development"
date: 2026-01-31
draft: false
tags: ["code", "examples", "snippets"]
categories: ["tutorials"]
---

## Quick Reference

Copy-paste ready code for common AI tasks.

## API Basics

### OpenAI Simple Chat

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

print(chat("Explain async/await"))
```

### Claude Simple Chat

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

print(chat("Explain decorators"))
```

### Streaming Responses

```python
# OpenAI streaming
def stream_openai(prompt):
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

# Claude streaming
def stream_claude(prompt):
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="")
```

## Embeddings & Search

### Generate Embeddings

```python
from openai import OpenAI

client = OpenAI()

def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

vector = embed("Machine learning basics")
print(f"Dimension: {len(vector)}")  # 1536
```

### Cosine Similarity

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

vec1 = embed("Python programming")
vec2 = embed("Coding in Python")
vec3 = embed("Italian cuisine")

print(cosine_similarity(vec1, vec2))  # ~0.9 (similar)
print(cosine_similarity(vec1, vec3))  # ~0.3 (different)
```

### Simple Semantic Search

```python
documents = [
    "Python is great for ML",
    "JavaScript for web dev",
    "SQL for databases"
]

# Embed all docs
doc_vectors = [embed(doc) for doc in documents]

def search(query, k=2):
    query_vec = embed(query)
    scores = [cosine_similarity(query_vec, doc_vec) for doc_vec in doc_vectors]
    top_indices = np.argsort(scores)[-k:][::-1]
    return [(documents[i], scores[i]) for i in top_indices]

results = search("machine learning")
for doc, score in results:
    print(f"{score:.3f}: {doc}")
```

## RAG Patterns

### Minimal RAG (No Framework)

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
        prompt = f"""Context:
{chr(10).join(context)}

Question: {question}
Answer based on context:"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# Usage
rag = SimpleRAG([
    "Python is a high-level programming language.",
    "It's popular for AI and data science.",
    "Python has simple, readable syntax."
])

print(rag.ask("What is Python good for?"))
```

### LangChain RAG

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and split
text = "Your long document here..."
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(chunks, embeddings)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Ask questions
answer = qa.run("What is the main topic?")
print(answer)
```

## Error Handling

### Retry with Exponential Backoff

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
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
        except APIError as e:
            print(f"API Error: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)

# Usage
result = api_call_with_retry(
    lambda: client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
)
```

### Timeout Handler

```python
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def with_timeout(func, timeout_sec=30):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    try:
        result = func()
        signal.alarm(0)
        return result
    except TimeoutError:
        return None

# Usage
result = with_timeout(lambda: slow_api_call(), timeout_sec=10)
```

## Prompt Templates

### Few-Shot Learning

```python
def few_shot_prompt(examples, new_input):
    prompt = "Learn from these examples:\n\n"
    for ex in examples:
        prompt += f"Input: {ex['input']}\nOutput: {ex['output']}\n\n"
    prompt += f"Now do this:\nInput: {new_input}\nOutput:"
    return prompt

examples = [
    {"input": "dog", "output": "animal"},
    {"input": "car", "output": "vehicle"},
]

prompt = few_shot_prompt(examples, "bicycle")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
print(response.choices[0].message.content)  # "vehicle"
```

### Chain of Thought

```python
def cot_prompt(question):
    return f"""{question}

Let's think step by step:
1. First, identify the key information
2. Then, reason through the problem
3. Finally, provide the answer

Thinking:"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": cot_prompt("What is 25% of 80?")}]
)
print(response.choices[0].message.content)
```

## Conversation Memory

### Simple Memory

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
print(bot.chat("My name is Alice"))
print(bot.chat("What's my name?"))  # Remembers!
```

### Sliding Window Memory

```python
class SlidingWindowChat:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = []
    
    def chat(self, message):
        self.history.append({"role": "user", "content": message})
        
        # Keep only last N messages
        recent = self.history[-self.window_size:]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=recent
        )
        
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        
        return reply
```

## Function Calling

### OpenAI Function Calling

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
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    print(f"Function: {tool_call.function.name}")
    print(f"Args: {args}")
```

## Async/Batch Processing

### Async Requests

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

# Usage
messages = ["What is Python?", "What is JavaScript?", "What is Rust?"]
results = asyncio.run(process_batch(messages))
for msg, result in zip(messages, results):
    print(f"{msg}: {result[:50]}...")
```

## Caching

### Simple Cache

```python
import hashlib
import json

cache = {}

def cached_api_call(func):
    def wrapper(*args, **kwargs):
        # Create cache key
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        key = hashlib.md5(key_data.encode()).hexdigest()
        
        if key in cache:
            print("Cache hit!")
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

# First call: API request
print(chat("Hello"))
# Second call: Cache hit!
print(chat("Hello"))
```
