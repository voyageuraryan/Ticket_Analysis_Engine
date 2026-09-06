"""
ChromaDB vector store implementation.
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional
from loguru import logger
from pathlib import Path


class ChromaVectorStore:
    """Manage ChromaDB vector store for tickets."""
    
    def __init__(self, persist_directory: str = "data/vector_store"):
        """
        Initialize ChromaDB client.
        
        Args:
            persist_directory: Directory to persist database
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        logger.info(f"Initialized ChromaDB at {self.persist_directory}")
        
        self.collection = None
    
    def create_collection(self, name: str = "sap_tickets", reset: bool = False):
        """
        Create or get collection.
        
        Args:
            name: Collection name
            reset: If True, delete existing collection
        """
        if reset:
            try:
                self.client.delete_collection(name=name)
                logger.info(f"Deleted existing collection: {name}")
            except:
                pass
        
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection '{name}' ready with {self.collection.count()} documents")
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict],
        ids: Optional[List[str]] = None,
        batch_size: int = 5000
    ):
        """
        Add documents to collection in batches.
        
        Args:
            texts: Document texts
            embeddings: Document embeddings
            metadatas: Document metadata
            ids: Document IDs (auto-generated if None)
            batch_size: Max batch size (ChromaDB limit is ~5461)
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        # Add documents in batches to avoid ChromaDB batch size limit
        total_docs = len(texts)
        for i in range(0, total_docs, batch_size):
            end_idx = min(i + batch_size, total_docs)
            batch_texts = texts[i:end_idx]
            batch_embeddings = embeddings[i:end_idx].tolist()
            batch_metadatas = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            self.collection.add(
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            logger.info(f"Added batch {i//batch_size + 1}: {end_idx}/{total_docs} documents")
        
        logger.info(f"✅ Added all {total_docs} documents to collection")
    
    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Metadata filter
            
        Returns:
            Search results
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        count = self.collection.count()
        
        # Sample metadata
        sample = self.collection.get(limit=1)
        
        return {
            'total_documents': count,
            'sample_metadata': sample['metadatas'][0] if sample['metadatas'] else {}
        }
    
    def get_all_documents(self) -> Dict:
        """Get all documents from collection."""
        return self.collection.get()
