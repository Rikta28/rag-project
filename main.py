from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from graph import graph

app = FastAPI()

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: dict


@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=request.query)
            ]
        }
    )
    
    final_answer = {
        "ai_response": result["messages"][-1].content,
        "retrieved_chunks": result["messages"][1].additional_kwargs["documents"],
        "confidence_score": result["messages"][1].additional_kwargs["retrieval_confidence"]
    }

    return QueryResponse(answer=final_answer)

