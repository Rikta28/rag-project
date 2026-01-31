# RAG Implementation with Pinecone

- Setup free trial account with Pinecone
- Get the default API key (use the free tier)


Create a directory for the project

```
mkdir rag_with_pinecone
cd rag_with_pinecone
```


Create a new python virtual env for hosting project's dependencies

```
python -m venv venv_pinecone
```


Activate venv

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

**Edit .env file**

Modify sample.env file with API key and index name and save as ".env" file.


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

