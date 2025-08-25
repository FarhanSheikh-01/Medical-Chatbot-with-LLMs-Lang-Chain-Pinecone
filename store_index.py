from dotenv import load_dotenv
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

if PINECONE_API_KEY is not None:
	os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if GROK_API_KEY is not None:
	os.environ["GROK_API_KEY"] = GROK_API_KEY
 
extracted_docs = load_pdf_files('data')
minimal_docs = filter_to_minimal_docs(extracted_docs)
texts_chunk = text_split(minimal_docs)

embedding = download_embeddings()

Pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=Pinecone_api_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384, 
        metric="cosine", 
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.has_index(index_name)

docsearch = PineconeVectorStore.from_documents(texts_chunk, embedding, index_name=index_name) 
 