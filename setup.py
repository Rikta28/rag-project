import os

from pinecone import Pinecone
from pinecone import ServerlessSpec

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore

from dotenv import dotenv_values
config = dotenv_values(".env")


# Initialise embedding model (using HuggingFaceEmbeddings wrapper)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

# Pass Pinecone API key
pc = Pinecone(api_key=config["PINECONE_API_KEY"])

# Create index
index_name = config["INDEX_NAME"]

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=768, # dimensionality of embedding model (multi-qa-mpnet-base-dot-v1)
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled",
        tags={
            "environment": "development"
        }
    )

# Initialise vector store
os.environ["PINECONE_API_KEY"] = config["PINECONE_API_KEY"]
vectorstore = PineconeVectorStore(index_name=config["INDEX_NAME"], embedding=embedding_model)

# Load and process documents
pdf_folder_path = "./docs"
loader = DirectoryLoader(pdf_folder_path, loader_cls=PyPDFLoader)
documents = loader.load()

# Split documents into chunks
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=51
)

chunked_documents = recursive_splitter.split_documents(documents)

# Upsert document chunks
vectorstore.add_documents(chunked_documents)

# Verify index stats
index = pc.Index(index_name)
stats = index.describe_index_stats()

print(stats)
