# RAG Implementation with Pinecone

# Setup Instructions

- **Python Version**: 3.10 or higher 
- **pip**: Latest version
- **API Keys**: OpenAI API key, Pinecone API key

- Setup free trial account with Pinecone
- Get the default API key (use the free tier)


**Create a directory for the project**

```
mkdir rag_with_pinecone
cd rag_with_pinecone
```


**Create a new python virtual env for hosting project's dependencies**

```
python -m venv venv_pinecone
```


**Activate venv**

For LINUX :

```
source venv_pinecone/bin/activate
```

For Windows :

```
source venv_pinecone/Scripts/activate
```

Install python dependencies

```
pip install -r requirements.txt
```

# Edit .env file

**Modify sample.env file with API key and index name and save as ".env" file.**


Run setup.py (to create index and upsert data)

```
python setup.py
```


Run app with uvicorn (on default port 8000)

```
uvicorn main:app --reload
```


Send a query using curl command from terminal (cmd)

```
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is agentic AI?"}'
```


Send a query using FastAPI frontend

Launch - http://localhost:8000/docs and use "Try it out" option to send post query.

# Architecture Overview

![alt text](Architecture.png)

# Project Structure

```
rag-project/
│
├── main.py                 # FastAPI entry point 
├── graph.py                # LangGraph workflow (retrieve → generate → confidence)
│
├── ingest/
│   ├── __init__.py
│   ├── pdf_ingestor.py     # PDF loading & chunking logic
│   └── vector_store.py     # Embeddings / Pinecone setup
│
├── prompts/
│   └── rag_prompt.py       # PDF-only prompt template
│
├── config/
│   └── settings.py         # API keys, model configs
│
├── data/
│   └── Agentic_AI_Ebook.pdf
│
├── requirements.txt
├── README.md
├── .env
└── .gitignore

```

# RAG + LangGraph Explanation

**What is RAG?**
- **Retrieval-Augmented Generation** combines information retrieval with text generation
- Instead of relying on LLM's training knowledge, we retrieve relevant context from our knowledge base
- The LLM then generates answers based ONLY on this retrieved context


**Why LangGraph?**
- **LangGraph** is a framework for building stateful, multi-step LLM workflows
- It provides a graph-based approach where each step (node) processes and transforms state
- Benefits:
  - **Clear workflow visualization**: Easy to understand and debug
  - **State management**: Automatically tracks data through the pipeline
  - **Modularity**: Each node (retrieve, context, generate) is independent
  - **Extensibility**: Easy to add new nodes (e.g., re-ranking, filtering)
  - **Production-ready**: Built-in error handling and logging


**Our LangGraph Pipeline:**
1. **Retrieve Node**: Query → Embeddings → Vector Search → Top Chunks
2. **Context Node**: Chunks → Formatted Context String
3. **Generate Node**: Context + Query → Grounded Prompt → LLM → Answer


# Design Decisions

- The PDF is chunked using recursive character splitting with a chunk size of 512 characters and overlap of 51 characters.

- A chunk size of 512 provides a strong balance between semantic completeness and retrieval precision.
- Smaller chunks improve the accuracy of vector similarity search by reducing irrelevant matches.
- An overlap of 51 characters ensures continuity of information across chunk boundaries and prevents loss of context between adjacent chunks.
- This configuration is well-suited for embedding models and keeps prompt context within LLM token limits.

This combination improves both retrieval relevance and answer grounding.


### Why this Embedding Model was chosen

The system uses OpenAI / HuggingFace embedding models optimized for semantic similarity search.

- Produces dense vector representations that capture semantic meaning effectively.

- Performs well on technical and long-form documents such as PDFs.

- Fully compatible with vector databases like Pinecone and FAISS.

### Why LangGraph Was Used

LangGraph was selected to implement a state-based AI workflow instead of a single linear chain.
It enables explicit modeling of retrieval, generation, and validation as independent nodes.Improves transparency and explainability of the RAG pipeline.
LangGraph provides fine-grained control over how information flows through the system.

### Grounding Enforce

Grounding is enforced through both architectural constraints and strict prompt control, ensuring all answers are sourced exclusively from the provided PDF.

**Retrieval-First Workflow:** The user query is never sent directly to the LLM. All responses must pass through a retrieval node that fetches relevant PDF chunks.

**Strict Context-Only Prompting:**
The LLM is instructed to answer only using retrieved context. If the answer is not present, it responds:  “Not found in document.”

**LangGraph-Controlled Execution:**  LangGraph enforces execution order, preventing answer generation without retrieved context and enabling optional validation steps.

**Transparent Output:** Each response returns the final answer, retrieved context chunks, and a confidence score.

# Sample Questions

Example 1

```
$ curl -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"query": "What is agentic AI?"}'
```
Response from RAG Agent
```
{"answer":"Agentic AI refers to systems capable of autonomous decision-making and action in pursuit of specific objectives. It has evolved from theoretical ideas to practical systems that are shaping industries."}
```
Example 2
```
$ curl -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"query": "What are agents composed of?"}'
```
Response from RAG Agent
```
{"answer":"Agents are composed of key elements that include the interaction between the agent, its environment, and its actions. They can be reactive or proactive, utilizing memory and learning from interactions. At the core of agentic systems is the decision-making layer, which employs sophisticated algorithms, including machine learning models, reinforcement learning, or rule-based reasoning, to analyze data and make informed decisions based on the system's objectives and the current environmental context."}
```

Example 3

```
$  curl -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"query":"How do agent workflows differ from LLM chains?"}'
```
Response from RAG Agent
```
{"answer":"Agent workflows differ from LLM chains in that agents are goal-driven systems capable of performing actions autonomously in a dynamic environment, while LLM chains primarily focus on processing and generating human-like text. Agents operate within a network, learning from past experiences to improve decision-making, whereas LLMs generate responses based on input without the same level of autonomous action or contextual learning."}
```

Example 4

```
$ curl -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"query":"What are limitations of agentic systems?"}'
```
Response from RAG Agent
```
{"answer":"The limitations of agentic systems include:\n\n1. **Complex System Design**: Designing agentic systems can be challenging due to the complexities of inter-agent communication and dependencies.\n\n2. **Interoperability**: Integrating agents with existing systems such as ERP, WMS, and TMS can be difficult, especially when agents are built on varying technologies or standards.\n\n3. **Data Security**: Sensitive data shared between agents and external vendors can be vulnerable to breaches, raising concerns about data security.\n\n4. **Communication and Coordination**: Ensuring seamless communication among diverse agents can be difficult due to differences in functionality and sophistication.\n\n5. **Conflict Management**: Managing conflicts that arise between agents can be a significant challenge.\n\n6. **Scalability**: As the number of agents increases, maintaining performance and efficiency can become problematic.\n\n7. **Cost of Implementation**: The initial investment required to implement agentic systems can be high.\n\n8. **Slow Development**: The development process for agentic systems can be slow due to the complexities involved.\n\n9. **Sophisticated Orchestration**: Coordinating the actions of multiple agents in a cohesive manner can be complex and requires sophisticated orchestration strategies."}
```

Example 5

```
$ curl -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"query":"What are the core components of an Agentic AI system according to the ebook?"}'
```
Response from RAG Agent
```
{"answer":"The core components of an Agentic AI system include Perception, Reasoning, Planning, Learning, and Execution. These components work together to enable autonomous decision-making and action, allowing agents to perceive and interact with their environment, drive their decisions, and adapt and learn over time."}
```
Example 6
```
$ curl -X POST http://localhost:8000/query   -H "Content-Type: application/json"   -d '{"query":"Explain the difference between traditional RAG and Agentic RAG mentioned in the text."}'
```
Response from RAG Agent
```
{"answer":"Not found in document."}
```
