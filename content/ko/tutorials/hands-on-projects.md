---
title: "실습 AI 프로젝트"
description: "AI 개발을 마스터하기 위한 실용 프로젝트"
date: 2026-01-31
draft: false
tags: ["프로젝트", "튜토리얼", "실습"]
categories: ["tutorials"]
---

## 만들면서 배우기

처음부터 실제 AI 앱을 구축하세요. 각 프로젝트에는 완전한 코드와 설명이 포함되어 있습니다.

## 프로젝트 1: AI 챗봇 (초보자)

**만들 것:** 메모리가 있는 대화형 AI

**시간:** 1-2시간

### 설정

```bash
pip install openai streamlit
```

### 완전한 코드

```python
# chatbot.py
import streamlit as st
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 채팅 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("AI 챗봇 🤖")

# 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 채팅 입력
if prompt := st.chat_input("말해보세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # AI 응답 받기
    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=st.session_state.messages
        )
        reply = response.choices[0].message.content
        st.write(reply)
    
    # 히스토리에 추가
    st.session_state.messages.append({"role": "assistant", "content": reply})
```

### 실행하기

```bash
streamlit run chatbot.py
```

**배울 것:** 컨텍스트 관리, Streamlit UI, 채팅 히스토리

---

## 프로젝트 2: 문서 Q&A (중급)

**만들 것:** PDF 업로드하고 질문하기

**시간:** 2-3시간

### 설정

```bash
pip install langchain langchain-openai chromadb pypdf
```

### 완전한 코드

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

st.title("문서 Q&A 📄")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 업로드", type="pdf")

if uploaded_file:
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    # 문서 처리
    with st.spinner("문서 처리 중..."):
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
    st.success("문서 처리 완료!")
    
    # Q&A
    question = st.text_input("문서에 대해 질문하세요:")
    if question:
        with st.spinner("생각 중..."):
            answer = qa.run(question)
        st.write("**답변:**", answer)
```

### 실행하기

```bash
export OPENAI_API_KEY=sk-your-key
streamlit run doc_qa.py
```

**배울 것:** RAG, PDF 처리, 벡터 데이터베이스

---

## 프로젝트 3: 이미지 캡션 생성기 (중급)

**만들 것:** 이미지를 설명하는 AI

**시간:** 1-2시간

### 설정

```bash
pip install openai pillow streamlit
```

### 완전한 코드

```python
# image_caption.py
import streamlit as st
from openai import OpenAI
import base64
from io import BytesIO

client = OpenAI()

st.title("이미지 캡션 생성기 🖼️")

uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 이미지 표시
    st.image(uploaded_file, caption="업로드된 이미지", use_column_width=True)
    
    # 캡션 생성 버튼
    if st.button("캡션 생성"):
        with st.spinner("이미지 분석 중..."):
            # base64로 변환
            image_bytes = uploaded_file.read()
            base64_image = base64.b64encode(image_bytes).decode()
            
            # GPT-4 Vision 호출
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "이 이미지를 자세히 설명해주세요."},
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
            st.write("**캡션:**", caption)
```

**배울 것:** Vision AI, 이미지 처리, base64 인코딩

---

## 프로젝트 4: AI 콘텐츠 생성기 (고급)

**만들 것:** SEO가 적용된 블로그 포스트 생성기

**시간:** 3-4시간

### 설정

```bash
pip install openai streamlit
```

### 완전한 코드

```python
# content_generator.py
import streamlit as st
from openai import OpenAI

client = OpenAI()

st.title("AI 콘텐츠 생성기 ✍️")

# 입력 폼
topic = st.text_input("주제:")
keywords = st.text_input("키워드 (쉼표로 구분):")
tone = st.selectbox("톤:", ["전문적", "캐주얼", "기술적", "친근한"])

if st.button("블로그 포스트 생성"):
    with st.spinner("작성 중..."):
        # 개요 생성
        outline_prompt = f\"\"\"{topic}에 대한 블로그 포스트 개요를 만드세요.
포함사항:
- 제목
- 5개 주요 섹션과 부제목
- SEO 최적화 대상: {keywords}
톤: {tone}\"\"\"
        
        outline = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": outline_prompt}]
        ).choices[0].message.content
        
        st.write("## 개요")
        st.write(outline)
        
        # 전체 포스트 생성
        post_prompt = f\"\"\"이 개요를 바탕으로 완전한 1000단어 블로그 포스트를 작성하세요:

{outline}

주제: {topic}
포함할 키워드: {keywords}
톤: {tone}

흥미롭고, SEO에 최적화되며, 예제를 포함하세요.\"\"\"
        
        with st.spinner("전체 포스트 작성 중..."):
            post = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": post_prompt}],
                temperature=0.7
            ).choices[0].message.content
        
        st.write("## 전체 포스트")
        st.write(post)
        
        # 메타 설명 생성
        meta_prompt = f"이 블로그 포스트를 위한 155자 SEO 메타 설명을 작성하세요:\\n{post[:500]}"
        meta = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": meta_prompt}]
        ).choices[0].message.content
        
        st.write("## 메타 설명")
        st.code(meta)
```

**배울 것:** 다단계 AI 워크플로우, SEO, 콘텐츠 생성

---

## 프로젝트 5: 도구를 사용하는 AI 에이전트 (고급)

**만들 것:** 도구를 사용하는 자율 에이전트

**시간:** 3-4시간

### 완전한 코드

```python
# ai_agent.py
import streamlit as st
from openai import OpenAI
import json
import requests

client = OpenAI()

# 도구 정의
tools = [{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "웹에서 정보 검색",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색 쿼리"}
            },
            "required": ["query"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "수학 계산 수행",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "수학 표현식"}
            },
            "required": ["expression"]
        }
    }
}]

def search_web(query):
    # 플레이스홀더 - 실제 검색 API 통합
    return f"{query}에 대한 검색 결과"

def calculate(expression):
    try:
        return str(eval(expression))
    except:
        return "계산 오류"

# 에이전트 루프
def run_agent(user_query):
    messages = [{"role": "user", "content": user_query}]
    
    for _ in range(5):  # 최대 5번 반복
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            # 도구 실행
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

st.title("AI 에이전트 🤖")

query = st.text_input("에이전트가 무엇을 하길 원하나요?")
if query:
    with st.spinner("에이전트 작동 중..."):
        result = run_agent(query)
    st.write(result)
```

**배울 것:** 함수 호출, 에이전트, 도구 사용

---

## 프로젝트 6: 풀스택 AI 앱 (전문가)

**만들 것:** FastAPI 백엔드 + React 프론트엔드를 갖춘 완전한 앱

**시간:** 8-10시간

### 백엔드 (FastAPI)

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

### 프론트엔드 (React)

```jsx
// frontend/src/App.jsx
import { useState } from 'react';

function App() {
  const [messages, setMessages] = useState([]);
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
      <button onClick={sendMessage}>보내기</button>
    </div>
  );
}
```

### 배포

```bash
# 백엔드
cd backend
uvicorn main:app --reload

# 프론트엔드
cd frontend
npm start
```

**배울 것:** API 디자인, 풀스택 개발, 프로덕션 배포

---

## 다음 단계

이 프로젝트들을 완료한 후:
1. 기능 추가 (인증, 데이터베이스, 파일 업로드)
2. 프로덕션에 배포 (Vercel, Railway, AWS)
3. 나만의 AI 제품 만들기!

**리소스:**
- OpenAI 문서
- LangChain 문서
- Streamlit 갤러리
