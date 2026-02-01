---
title: "Building RAG Systems: Complete Guide"
description: "Learn how to build Retrieval-Augmented Generation systems"
date: 2026-01-31
draft: false
tags: ["RAG", "embeddings", "vector-database"]
categories: ["ai-tools"]
---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** = Search + AI Generation

**Why RAG?**
- ✅ LLMs don't know recent/private data
- ✅ RAG adds your data to context
- ✅ More accurate, factual responses
- ✅ Cite sources

**How it works:**
1. **Index**: Split docs → Embed → Store in vector DB
2. **Query**: User asks → Find relevant docs
3. **Generate**: LLM answers using retrieved docs

## Quick Start (5 minutes)

```bash
pip install langchain langchain-openai chromadb
```

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. Prepare documents
documents = ["Python is a programming language", "JavaScript is for web dev"]

# 2. Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = splitter.create_documents(documents)

# 3. Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# 4. Create QA chain
llm = ChatOpenAI()
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# 5. Ask questions
print(qa.run("What is Python?"))
```

## Document Processing

### 1. Text Splitting

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Max characters per chunk
    chunk_overlap=200,    # Overlap between chunks
    separators=["\\n\\n", "\\n", " ", ""]  # Split priority
)

chunks = splitter.split_text(long_text)
print(f"Created {len(chunks)} chunks")
```

**Chunking strategies:**
```python
# Small chunks (better precision)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Large chunks (more context)
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# Semantic splitting (by paragraph)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    separators=["\\n\\n", "\\n", ". ", " "]
)
```

### 2. Loading Documents

```python
from langchain.document_loaders import (
    TextLoader, PyPDFLoader, UnstructuredMarkdownLoader, 
    WebBaseLoader, DirectoryLoader
)

# Load text file
loader = TextLoader("data.txt")
docs = loader.load()

# Load PDF
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Load from web
loader = WebBaseLoader("https://example.com/article")
docs = loader.load()

# Load all files in directory
loader = DirectoryLoader("./docs", glob="**/*.md")
docs = loader.load()
```

### 3. Metadata Extraction

```python
from langchain.schema import Document

# Add metadata
docs_with_metadata = []
for i, chunk in enumerate(chunks):
    doc = Document(
        page_content=chunk,
        metadata={
            "source": "manual.pdf",
            "page": i // 5,  # Estimate page
            "chunk_id": i
        }
    )
    docs_with_metadata.append(doc)
```

## Embeddings

### OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # or "text-embedding-3-large"
)

# Embed single text
vector = embeddings.embed_query("Hello world")
print(f"Vector dimension: {len(vector)}")  # 1536

# Embed multiple texts
vectors = embeddings.embed_documents(["Text 1", "Text 2"])
```

### Other Embedding Models

```python
# HuggingFace
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Cohere
from langchain.embeddings import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")
```

## Vector Databases

### Chroma (Local)

```python
from langchain.vectorstores import Chroma

# Create
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Save to disk
)

# Load existing
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Search
results = vectorstore.similarity_search("query", k=3)
```

### FAISS (Fast)

```python
from langchain.vectorstores import FAISS

# Create
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save/Load
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings)

# Search with scores
results = vectorstore.similarity_search_with_score("query", k=3)
for doc, score in results:
    print(f"Score: {score}, Text: {doc.page_content[:100]}")
```

### Pinecone (Cloud)

```python
from langchain.vectorstores import Pinecone
import pinecone

# Initialize
pinecone.init(api_key="your-key", environment="us-west1-gcp")
index_name = "my-index"

# Create
vectorstore = Pinecone.from_documents(chunks, embeddings, index_name=index_name)

# Search
results = vectorstore.similarity_search("query", k=3)
```

## Retrieval Strategies

### 1. Similarity Search

```python
# Basic similarity
results = vectorstore.similarity_search("Python programming", k=3)

# With scores
results = vectorstore.similarity_search_with_score("Python", k=3)
for doc, score in results:
    print(f"Score: {score:.3f}")
```

### 2. MMR (Maximum Marginal Relevance)

```python
# Diverse results (less redundancy)
results = vectorstore.max_marginal_relevance_search(
    "Python programming",
    k=5,
    fetch_k=20,  # Fetch more, then filter
    lambda_mult=0.5  # 0=diverse, 1=similar
)
```

### 3. Metadata Filtering

```python
# Filter by metadata
results = vectorstore.similarity_search(
    "Python",
    k=3,
    filter={"source": "tutorial.pdf", "page": 1}
)
```

### 4. Multi-Query Retrieval

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# Generates multiple queries automatically
results = retriever.get_relevant_documents("Explain Python")
```

## Generation

### Basic QA Chain

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # stuff, map_reduce, refine, map_rerank
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

answer = qa_chain.run("What is Python used for?")
```

### Conversational RAG

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Multi-turn conversation
print(qa({"question": "What is in the document?"})["answer"])
print(qa({"question": "Tell me more about that"})["answer"])  # Uses memory!
```

### Custom Prompts

```python
from langchain.prompts import PromptTemplate

template = \"\"\"Use the following context to answer the question.
If you don't know, say "I don't know."

Context: {context}

Question: {question}

Answer:\"\"\"

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)
```

## Complete RAG System

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader, TextLoader

class RAGSystem:
    def __init__(self, docs_path):
        # Load documents
        loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        
        # Create vector store
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(chunks, embeddings)
        
        # Create LLM
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Create memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create chain
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            return_source_documents=True
        )
    
    def ask(self, question):
        result = self.qa({"question": question})
        
        # Print sources
        print("\\nSources:")
        for doc in result["source_documents"]:
            print(f"- {doc.metadata.get('source', 'Unknown')}")
        
        return result["answer"]
    
    def reset(self):
        self.memory.clear()

# Usage
rag = RAGSystem("./docs")
print(rag.ask("What is the main topic?"))
print(rag.ask("Can you elaborate?"))
rag.reset()
```

## Best Practices

### 1. Chunking

```python
# ✅ Good chunk size
chunk_size = 1000  # 1-2 paragraphs

# ✅ Overlap prevents context loss
chunk_overlap = 200  # 20% of chunk_size

# ❌ Too small: loses context
# ❌ Too large: dilutes relevance
```

### 2. Retrieval

```python
# ✅ Retrieve more than you need
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ✅ Use MMR for diversity
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)
```

### 3. Prompt Engineering

```python
# ✅ Clear instructions
prompt = \"\"\"Answer based on context only.
If unsure, say "I don't have enough information."

Context: {context}
Question: {question}
Answer:\"\"\"

# ❌ Vague
prompt = "Answer: {question}"
```

### 4. Evaluation

```python
# Test questions
test_cases = [
    {"question": "What is X?", "expected": "X is..."},
    {"question": "How to do Y?", "expected": "To do Y..."}
]

for case in test_cases:
    answer = qa_chain.run(case["question"])
    print(f"Q: {case['question']}")
    print(f"A: {answer}")
    print(f"Expected: {case['expected']}\\n")
```

## Common Issues

**Low relevance:**
```python
# Solution: Better chunking, more documents, tune k
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

**Slow performance:**
```python
# Solution: Use FAISS instead of Chroma
vectorstore = FAISS.from_documents(chunks, embeddings)
```

**High costs:**
```python
# Solution: Use smaller embeddings, fewer chunks
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Cheaper
```

## Advanced: Re-ranking

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Compressor (re-ranker)
compressor = LLMChainExtractor.from_llm(llm)

# Compressed retriever
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Returns most relevant after re-ranking
docs = retriever.get_relevant_documents("query")
```
