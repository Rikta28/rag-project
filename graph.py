import os

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import StateGraph
from langgraph.graph import END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

from langchain_pinecone import PineconeVectorStore

from typing_extensions import TypedDict
from typing import Annotated

from dotenv import dotenv_values
config = dotenv_values(".env")


# Initialise embedding model (using HuggingFaceEmbeddings wrapper)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

# Pass Pinecone API key
pc = Pinecone(api_key=config["PINECONE_API_KEY"])

# Initialise vector store
os.environ["PINECONE_API_KEY"] = config["PINECONE_API_KEY"]
vectorstore = PineconeVectorStore(index_name=config["INDEX_NAME"], embedding=embedding_model)

# Declare the model
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.1,
    api_key=config["OPENAI_API_KEY"]
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)


def get_user_query(state: State):
    # latest message is the user query
    last_message = state["messages"][-1]

    if not last_message.content.strip():
        return {
            "messages": [
                HumanMessage(content="Invalid user query")
            ]
        }

    # passing nothing explicitly, state will passed to the next node anyway
    return {}


def retrieve_data_pinecone(state: State):
    # Extract user query from state
    query = state["messages"][-1].content

    retrieved_docs = vectorstore.similarity_search_with_score(
        query,
        k=3
    )

    docs = [
        {"content": doc.page_content, "score": score}
        for doc, score in retrieved_docs
    ]

    avg_score = round(
        sum(score for _, score in retrieved_docs) / len(retrieved_docs),
        2
    ) if retrieved_docs else 0.0

    return {
        "messages": [
            AIMessage(
                content=docs,
                additional_kwargs={
                    "documents": docs,
                    "retrieval_confidence": avg_score,
                },
            )
        ]
    }


def generate_response(state: State):
    # Messages order:
    # - user query
    # - retrieved docs
    user_query = state["messages"][0].content
    docs_content = state["messages"][-1].content

    prompt = f"""
    You are an assistant that answers questions exclusively about Agentic AI.
    
    Here's a question:
    {user_query}
    
    Only use the below context to answer the question:
    
    {docs_content}
    
    If the question is not related to Agentic AI or if the provided context doesn't have the details about the question then respond with answer "Not found in document".

    Answer:
    """
    
    answer = llm.invoke(prompt)

    return {
        "messages": [
            AIMessage(content=answer.content)
        ]
    }


graph_builder.add_node("get_user_query", get_user_query)
graph_builder.add_node("retrieve_data_pinecone", retrieve_data_pinecone)
graph_builder.add_node("generate_response", generate_response)

graph_builder.add_edge(START, "get_user_query")
graph_builder.add_edge("get_user_query", "retrieve_data_pinecone")
graph_builder.add_edge("retrieve_data_pinecone", "generate_response")
graph_builder.add_edge("generate_response", END)

graph = graph_builder.compile()



