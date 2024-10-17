from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model, document):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Returns:
    - VectorStoreIndex: An index of vector embeddings for efficient similarity queries.
    """
    try:
        logging.info("Initializing the Gemini embedding model...")
        
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")

        logging.info("Creating the storage context...")
        storage_context = StorageContext.from_defaults()

        logging.info("Creating the index from documents...")
        index = VectorStoreIndex.from_documents(
            documents=document,
            storage_context=storage_context,
            embed_model=gemini_embed_model,  
            chunk_size=800,  
            chunk_overlap=20 
        )

        index.storage_context.persist()

        logging.info("Index created successfully. Setting up the query engine...")
        query_engine = index.as_query_engine(llm=model)
        
        return query_engine

    except Exception as e:
        logging.error("Error occurred while downloading Gemini embedding.", exc_info=True)
        raise customexception(e, sys)
