from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from graph import graph

app = FastAPI()

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=request.query)
            ]
        }
    )

    final_answer = result["messages"][-1].content

    return QueryResponse(answer=final_answer)

