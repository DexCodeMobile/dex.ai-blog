---
title: "LangChain 튜토리얼: AI 애플리케이션 구축"
description: "LangChain 프레임워크 종합 가이드"
date: 2026-01-31
draft: false
tags: ["LangChain", "프레임워크", "튜토리얼"]
categories: ["ai-tools"]
---

## 왜 LangChain인가?

**LangChain이 간단하게 해주는 것:**
- ✅ 복잡한 AI 워크플로우 구축
- ✅ LLM을 데이터 소스에 연결
- ✅ 자율 에이전트 생성
- ✅ 대화 메모리 관리
- ✅ 표준화된 도구 통합

## 빠른 시작

```bash
pip install langchain langchain-openai
```

### 첫 체인

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# 구성요소 생성
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template("{topic}에 관한 농담을 해줘")
output_parser = StrOutputParser()

# 체인 구축
chain = prompt | llm | output_parser

# 실행
result = chain.invoke({"topic": "프로그래밍"})
print(result)
```

## 체인

### 간단한 LLMChain

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI()
prompt = PromptTemplate(
    input_variables=["product"],
    template="{product}에 대한 태그라인을 작성하세요"
)

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("스마트 물병"))
```

### 순차적 체인

```python
from langchain.chains import SimpleSequentialChain

# 체인 1: 아이디어 생성
idea_prompt = PromptTemplate(
    input_variables=["topic"],
    template="{topic}에 관한 스타트업 아이디어를 생성하세요"
)
idea_chain = LLMChain(llm=llm, prompt=idea_prompt)

# 체인 2: 피치 작성
pitch_prompt = PromptTemplate(
    input_variables=["idea"],
    template="다음 아이디어에 대한 2문장 피치를 작성하세요: {idea}"
)
pitch_chain = LLMChain(llm=llm, prompt=pitch_prompt)

# 결합
overall_chain = SimpleSequentialChain(
    chains=[idea_chain, pitch_chain],
    verbose=True
)

print(overall_chain.run("지속가능성"))
```

### 라우터 체인

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# 전문화된 프롬프트 정의
python_template = """당신은 Python 전문가입니다. {input}"""
javascript_template = """당신은 JavaScript 전문가입니다. {input}"""

prompt_infos = [
    {
        "name": "python",
        "description": "Python 질문에 적합",
        "prompt_template": python_template
    },
    {
        "name": "javascript",
        "description": "JavaScript 질문에 적합",
        "prompt_template": javascript_template
    }
]

# 라우터 체인 생성
chain = MultiPromptChain.from_prompts(llm, prompt_infos, verbose=True)

print(chain.run("Python에서 파일을 어떻게 읽어?"))
print(chain.run("JavaScript promise가 뭐야?"))
```

## 에이전트

### 간단한 에이전트

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

def search_tool(query: str) -> str:
    # 플레이스홀더 - 실제 검색 통합
    return f"{query}에 대한 검색 결과"

def calculator_tool(expression: str) -> str:
    return str(eval(expression))

tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="정보 검색"
    ),
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="수학 표현식 계산"
    )
]

llm = ChatOpenAI(temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

print(agent.run("25 * 17은 얼마야?"))
```

### ReAct 에이전트

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# ReAct 프롬프트 가져오기
prompt = hub.pull("hwchase17/react")

# 에이전트 생성
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

response = agent_executor.invoke({
    "input": "Python 튜토리얼을 검색하고 상위 3개를 알려줘"
})
print(response["output"])
```

## 메모리

### Conversation Buffer Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

print(conversation.predict(input="안녕, 나는 숨리야"))
print(conversation.predict(input="내 이름이 뭐야?"))  # 기억함!
print(conversation.predict(input="농담 해줘"))

# 히스토리 보기
print(memory.buffer)
```

### Conversation Summary Memory

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 긴 대화가 요약됨
for i in range(10):
    conversation.predict(input=f"우주에 관한 사실 {i}번을 알려줘")

# 요약 확인
print(memory.buffer)
```

### Vector Store Memory

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 벡터 저장소 생성
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

# 메모리 생성
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
memory = VectorStoreRetrieverMemory(retriever=retriever)

# 컨텍스트 저장
memory.save_context(
    {"input": "내가 좋아하는 색은 파란색이야"},
    {"output": "알겠습니다!"}
)

# 관련 컨텍스트 검색
print(memory.load_memory_variables({"prompt": "내가 좋아하는 색이 뭐야?"})["history"])
```

## 리트리버 & RAG

### 문서 로딩

```python
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 문서 로드
loader = TextLoader("data.txt")
# 또는: loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 청크로 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"{len(chunks)}개의 청크 생성")
```

### 벡터 저장소 RAG

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# 벡터 저장소 생성
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# QA 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 질문하기
result = qa_chain.run("문서의 주요 주제가 뭘까요?")
print(result)
```

### 완전한 RAG 시스템

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 메모리 생성
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 대화형 RAG 생성
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# 후속 질문하기
print(qa({"question": "이 문서는 무엇에 관한 거야?"}))
print(qa({"question": "좋아, 자세히 설명해줘?"}))  # 메모리 사용!
```

## 콜백

```python
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

class CustomCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM 시작, 프롬프트: {prompts[0][:50]}...")
    
    def on_llm_end(self, response, **kwargs):
        print(f"LLM 종료. 토큰: {response.llm_output['token_usage']}")

llm = ChatOpenAI(
    callbacks=[CustomCallback(), StreamingStdOutCallbackHandler()]
)

llm.invoke("짧은 이야기를 써줘")
```

## 완전한 애플리케이션

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

class ChatApp:
    def __init__(self, system_message="당신은 도움이 되는 어시스턴트입니다."):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        
        # 체인 생성
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=prompt,
            verbose=False
        )
    
    def chat(self, message):
        return self.conversation.predict(input=message)
    
    def reset(self):
        self.memory.clear()

# 사용법
app = ChatApp(system_message="당신은 Python 튜터입니다.")
print(app.chat("데코레이터가 뭐야?"))
print(app.chat("예제를 보여줘"))
app.reset()
```

## 모범 사례

### 1. 적절한 체인 타입 선택

```python
# 간단한 작업 → LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# 순차적 단계 → SequentialChain
chain = SimpleSequentialChain(chains=[chain1, chain2])

# 도구 필요 → Agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# 문서 Q&A → RetrievalQA
chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
```

### 2. 메모리 관리

```python
# 짧은 대화 → BufferMemory
memory = ConversationBufferMemory()

# 긴 대화 → SummaryMemory
memory = ConversationSummaryMemory(llm=llm)

# 특정 사실 → VectorStoreMemory
memory = VectorStoreRetrieverMemory(retriever=retriever)
```

### 3. 오류 처리

```python
from langchain.schema import OutputParserException

try:
    result = chain.run(input_text)
except OutputParserException as e:
    print(f"파서 오류: {e}")
except Exception as e:
    print(f"예상치 못한 오류: {e}")
```

## 일반적인 패턴

**문서와 채팅:**
```python
# 로드 → 분할 → 임베딩 → 저장 → 검색 → 생성
```

**에이전트 워크플로우:**
```python
# 질문 → 계획 → 도구 사용 → 관찰 → 답변
```

**다단계 추론:**
```python
# 메모리가 있는 순차적 체인 또는 에이전트
```
