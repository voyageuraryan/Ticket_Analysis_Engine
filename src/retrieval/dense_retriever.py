"""
Dense retrieval using vector embeddings.
"""

import numpy as np
from typing import List, Dict
from loguru import logger


class DenseRetriever:
    """Dense retrieval using vector similarity."""
    
    def __init__(self, vector_store, embedder):
        """
        Initialize dense retriever.
        
        Args:
            vector_store: ChromaVectorStore instance
            embedder: GeminiEmbedder instance
        """
        self.vector_store = vector_store
        self.embedder = embedder
        logger.info("Initialized DenseRetriever")
    
    def retrieve(self, query: str, k: int = 10, where: Dict = None) -> List[Dict]:
        """
        Retrieve similar documents using vector similarity.
        
        Args:
            query: Query text
            k: Number of results to return
            where: Metadata filter
            
        Returns:
            List of retrieved documents with metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=k,
            where=where
        )
        
        # Format results
        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            retrieved_docs.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                'rank': i + 1,
                'retriever': 'dense'
            })
        
        logger.debug(f"Dense retrieval found {len(retrieved_docs)} documents")
        return retrieved_docs
