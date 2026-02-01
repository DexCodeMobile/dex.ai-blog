---
title: "LangChain Tutorial: Building AI Applications"
description: "Comprehensive guide to LangChain framework"
date: 2026-01-31
draft: false
tags: ["LangChain", "framework", "tutorial"]
categories: ["ai-tools"]
---

## Why LangChain?

**LangChain simplifies:**
- ✅ Building complex AI workflows
- ✅ Connecting LLMs to data sources
- ✅ Creating autonomous agents
- ✅ Managing conversation memory
- ✅ Standardized tool integrations

## Quick Start

```bash
pip install langchain langchain-openai
```

### First Chain

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Create components
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
output_parser = StrOutputParser()

# Build chain
chain = prompt | llm | output_parser

# Run it
result = chain.invoke({"topic": "programming"})
print(result)
```

## Chains

### Simple LLMChain

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI()
prompt = PromptTemplate(
    input_variables=["product"],
    template="Write a tagline for {product}"
)

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("smart water bottle"))
```

### Sequential Chains

```python
from langchain.chains import SimpleSequentialChain

# Chain 1: Generate idea
idea_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate a startup idea about {topic}"
)
idea_chain = LLMChain(llm=llm, prompt=idea_prompt)

# Chain 2: Write pitch
pitch_prompt = PromptTemplate(
    input_variables=["idea"],
    template="Write a 2-sentence pitch for: {idea}"
)
pitch_chain = LLMChain(llm=llm, prompt=pitch_prompt)

# Combine
overall_chain = SimpleSequentialChain(
    chains=[idea_chain, pitch_chain],
    verbose=True
)

print(overall_chain.run("sustainability"))
```

### Router Chains

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Define specialized prompts
python_template = """You are a Python expert. {input}"""
javascript_template = """You are a JavaScript expert. {input}"""

prompt_infos = [
    {
        "name": "python",
        "description": "Good for Python questions",
        "prompt_template": python_template
    },
    {
        "name": "javascript",
        "description": "Good for JavaScript questions",
        "prompt_template": javascript_template
    }
]

# Create router chain
chain = MultiPromptChain.from_prompts(llm, prompt_infos, verbose=True)

print(chain.run("How do I read a file in Python?"))
print(chain.run("What are JavaScript promises?"))
```

## Agents

### Simple Agent

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

def search_tool(query: str) -> str:
    # Placeholder - integrate real search
    return f"Search results for: {query}"

def calculator_tool(expression: str) -> str:
    return str(eval(expression))

tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="Search for information"
    ),
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Calculate math expressions"
    )
]

llm = ChatOpenAI(temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

print(agent.run("What is 25 * 17?"))
```

### ReAct Agent

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Get ReAct prompt
prompt = hub.pull("hwchase17/react")

# Create agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

response = agent_executor.invoke({
    "input": "Search for Python tutorials and tell me the top 3"
})
print(response["output"])
```

## Memory

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

print(conversation.predict(input="Hi, I'm Alice"))
print(conversation.predict(input="What's my name?"))  # Remembers!
print(conversation.predict(input="Tell me a joke"))

# View history
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

# Long conversation gets summarized
for i in range(10):
    conversation.predict(input=f"Tell me fact {i} about space")

# Check summary
print(memory.buffer)
```

### Vector Store Memory

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

# Create memory
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
memory = VectorStoreRetrieverMemory(retriever=retriever)

# Save context
memory.save_context(
    {"input": "My favorite color is blue"},
    {"output": "Noted!"}
)

# Retrieve relevant context
print(memory.load_memory_variables({"prompt": "What's my favorite color?"})["history"])
```

## Retrievers & RAG

### Document Loading

```python
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
loader = TextLoader("data.txt")
# OR: loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")
```

### Vector Store RAG

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Ask questions
result = qa_chain.run("What is the main topic of the document?")
print(result)
```

### Complete RAG System

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create conversational RAG
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Ask follow-up questions
print(qa({"question": "What is this document about?"}))
print(qa({"question": "Can you elaborate on that?"}))  # Uses memory!
```

## Callbacks

```python
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

class CustomCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with prompt: {prompts[0][:50]}...")
    
    def on_llm_end(self, response, **kwargs):
        print(f"LLM finished. Tokens: {response.llm_output['token_usage']}")

llm = ChatOpenAI(
    callbacks=[CustomCallback(), StreamingStdOutCallbackHandler()]
)

llm.invoke("Tell me a short story")
```

## Complete Application

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
    def __init__(self, system_message="You are a helpful assistant."):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        
        # Create chain
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

# Usage
app = ChatApp(system_message="You are a Python tutor.")
print(app.chat("What are decorators?"))
print(app.chat("Show me an example"))
app.reset()
```

## Best Practices

### 1. Choose Right Chain Type

```python
# Simple task → LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Sequential steps → SequentialChain
chain = SimpleSequentialChain(chains=[chain1, chain2])

# Need tools → Agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# Q&A on documents → RetrievalQA
chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
```

### 2. Memory Management

```python
# Short conversations → BufferMemory
memory = ConversationBufferMemory()

# Long conversations → SummaryMemory
memory = ConversationSummaryMemory(llm=llm)

# Specific facts → VectorStoreMemory
memory = VectorStoreRetrieverMemory(retriever=retriever)
```

### 3. Error Handling

```python
from langchain.schema import OutputParserException

try:
    result = chain.run(input_text)
except OutputParserException as e:
    print(f"Parser error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Common Patterns

**Chat with docs:**
```python
# Load → Split → Embed → Store → Retrieve → Generate
```

**Agent workflow:**
```python
# Question → Plan → Use Tools → Observe → Answer
```

**Multi-step reasoning:**
```python
# Sequential chains or agents with memory
```
