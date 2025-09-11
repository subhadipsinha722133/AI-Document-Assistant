from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

def create_embeddings():
    """
    Initialize local sentence transformer embeddings
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings

def create_vector_store(chunks, embeddings):
    """
    Create FAISS vector store from text chunks
    """
    # Check if we have valid chunks
    if not chunks:
        raise ValueError("No text chunks available for creating vector store.")
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store