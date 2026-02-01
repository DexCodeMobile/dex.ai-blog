---
title: "Hands-On AI Projects"
description: "Practical projects to master AI development"
date: 2026-01-31
draft: false
tags: ["projects", "tutorial", "practice"]
categories: ["tutorials"]
---

## Learn By Building

Build real AI apps from scratch. Each project includes complete code and explanations.

## Project 1: AI Chatbot (Beginner)

**What you'll build:** Conversational AI with memory

**Time:** 1-2 hours

### Setup

```bash
pip install openai streamlit
```

### Complete Code

```python
# chatbot.py
import streamlit as st
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("AI Chatbot 🤖")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Say something..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=st.session_state.messages
        )
        reply = response.choices[0].message.content
        st.write(reply)
    
    # Add to history
    st.session_state.messages.append({"role": "assistant", "content": reply})
```

### Run It

```bash
streamlit run chatbot.py
```

**Learn more:** Context management, Streamlit UI, chat history

---

## Project 2: Document Q&A (Intermediate)

**What you'll build:** Upload PDFs, ask questions about them

**Time:** 2-3 hours

### Setup

```bash
pip install langchain langchain-openai chromadb pypdf
```

### Complete Code

```python
# doc_qa.py
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import tempfile
import os

st.title("Document Q&A 📄")

# File upload
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    # Process document
    with st.spinner("Processing document..."):
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(chunks, embeddings)
        
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(),
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )
    
    os.unlink(tmp_path)
    st.success("Document processed!")
    
    # Q&A
    question = st.text_input("Ask a question about your document:")
    if question:
        with st.spinner("Thinking..."):
            answer = qa.run(question)
        st.write("**Answer:**", answer)
```

### Run It

```bash
export OPENAI_API_KEY=sk-your-key
streamlit run doc_qa.py
```

**Learn more:** RAG, PDF processing, vector databases

---

## Project 3: Image Caption Generator (Intermediate)

**What you'll build:** AI that describes images

**Time:** 1-2 hours

### Setup

```bash
pip install openai pillow streamlit
```

### Complete Code

```python
# image_caption.py
import streamlit as st
from openai import OpenAI
import base64
from io import BytesIO

client = OpenAI()

st.title("Image Caption Generator 🖼️")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Generate caption button
    if st.button("Generate Caption"):
        with st.spinner("Analyzing image..."):
            # Convert to base64
            image_bytes = uploaded_file.read()
            base64_image = base64.b64encode(image_bytes).decode()
            
            # Call GPT-4 Vision
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }],
                max_tokens=300
            )
            
            caption = response.choices[0].message.content
            st.write("**Caption:**", caption)
```

**Learn more:** Vision AI, image processing, base64 encoding

---

## Project 4: AI Content Generator (Advanced)

**What you'll build:** Blog post generator with SEO

**Time:** 3-4 hours

### Setup

```bash
pip install openai streamlit
```

### Complete Code

```python
# content_generator.py
import streamlit as st
from openai import OpenAI

client = OpenAI()

st.title("AI Content Generator ✍️")

# Input form
topic = st.text_input("Topic:")
keywords = st.text_input("Keywords (comma-separated):")
tone = st.selectbox("Tone:", ["Professional", "Casual", "Technical", "Friendly"])

if st.button("Generate Blog Post"):
    with st.spinner("Writing..."):
        # Generate outline
        outline_prompt = f\"\"\"Create a blog post outline about {topic}.
Include:
- Title
- 5 main sections with subheadings
- SEO-optimized for: {keywords}
Tone: {tone}\"\"\"
        
        outline = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": outline_prompt}]
        ).choices[0].message.content
        
        st.write("## Outline")
        st.write(outline)
        
        # Generate full post
        post_prompt = f\"\"\"Write a complete 1000-word blog post based on this outline:

{outline}

Topic: {topic}
Keywords to include: {keywords}
Tone: {tone}

Make it engaging, SEO-optimized, and include examples.\"\"\"
        
        with st.spinner("Writing full post..."):
            post = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": post_prompt}],
                temperature=0.7
            ).choices[0].message.content
        
        st.write("## Full Post")
        st.write(post)
        
        # Generate meta description
        meta_prompt = f"Write a 155-character SEO meta description for this blog post:\\n{post[:500]}"
        meta = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": meta_prompt}]
        ).choices[0].message.content
        
        st.write("## Meta Description")
        st.code(meta)
```

**Learn more:** Multi-step AI workflows, SEO, content generation

---

## Project 5: AI Agent with Tools (Advanced)

**What you'll build:** Autonomous agent that uses tools

**Time:** 3-4 hours

### Complete Code

```python
# ai_agent.py
import streamlit as st
from openai import OpenAI
import json
import requests

client = OpenAI()

# Define tools
tools = [{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    }
}]

def search_web(query):
    # Placeholder - integrate real search API
    return f"Search results for: {query}"

def calculate(expression):
    try:
        return str(eval(expression))
    except:
        return "Error in calculation"

# Agent loop
def run_agent(user_query):
    messages = [{"role": "user", "content": user_query}]
    
    for _ in range(5):  # Max 5 iterations
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            # Execute tools
            messages.append(message)
            
            for tool_call in message.tool_calls:
                if tool_call.function.name == "search_web":
                    args = json.loads(tool_call.function.arguments)
                    result = search_web(args["query"])
                elif tool_call.function.name == "calculate":
                    args = json.loads(tool_call.function.arguments)
                    result = calculate(args["expression"])
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            return message.content

st.title("AI Agent 🤖")

query = st.text_input("What do you want the agent to do?")
if query:
    with st.spinner("Agent working..."):
        result = run_agent(query)
    st.write(result)
```

**Learn more:** Function calling, agents, tool use

---

## Project 6: Full-Stack AI App (Expert)

**What you'll build:** Complete app with FastAPI backend + React frontend

**Time:** 8-10 hours

### Backend (FastAPI)

```python
# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        messages = request.history + [
            {"role": "user", "content": request.message}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        return {
            "reply": response.choices[0].message.content,
            "usage": dict(response.usage)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Frontend (React)

```jsx
// frontend/src/App.jsx
import { useState } from 'react';

function App() {
  const [messages, setMessages] = [useState([]);
  const [input, setInput] = useState('');
  
  const sendMessage = async () => {
    const newMessages = [...messages, { role: 'user', content: input }];
    setMessages(newMessages);
    setInput('');
    
    const response = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: input,
        history: messages
      })
    });
    
    const data = await response.json();
    setMessages([...newMessages, { role: 'assistant', content: data.reply }]);
  };
  
  return (
    <div className=\"chat-app\">
      <div className=\"messages\">
        {messages.map((msg, i) => (
          <div key={i} className={msg.role}>
            {msg.content}
          </div>
        ))}
      </div>
      <input 
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
      />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
}
```

### Deploy

```bash
# Backend
cd backend
uvicorn main:app --reload

# Frontend
cd frontend
npm start
```

**Learn more:** API design, full-stack development, production deployment

---

## Next Steps

After completing these projects:
1. Add features (authentication, database, file uploads)
2. Deploy to production (Vercel, Railway, AWS)
3. Build your own AI product!

**Resources:**
- OpenAI Documentation
- LangChain Docs
- Streamlit Gallery
