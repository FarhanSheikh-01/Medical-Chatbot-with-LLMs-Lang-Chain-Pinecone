from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# Extract text from PDF files in the 'data' directory
def load_pdf_files(directory):
    loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents


# Filter documents to keep only those with minimal content
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filters out documents that are shorter than the specified minimum length.
    
    Args:
        docs (List[Document]): List of Document objects to filter.
        min_length (int): Minimum length of document content to keep.
        
    Returns:
        List[Document]: Filtered list of Document objects.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get('source')
        minimal_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))
    return minimal_docs


# Split documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


# Download and return embeddings model
def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


