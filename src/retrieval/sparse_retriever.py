"""
Sparse retrieval using BM25.
"""

from rank_bm25 import BM25Okapi
from typing import List, Dict
from loguru import logger
import pickle
from pathlib import Path


class SparseRetriever:
    """Sparse retrieval using BM25."""
    
    def __init__(self, corpus: List[str] = None, metadata: List[Dict] = None, index_path: str = None):
        """
        Initialize sparse retriever.
        
        Args:
            corpus: List of documents
            metadata: List of metadata dicts
            index_path: Path to save/load index
        """
        self.corpus = corpus
        self.metadata = metadata
        self.index_path = Path(index_path) if index_path else None
        self.bm25 = None
        
        if corpus and metadata:
            self._build_index()
        elif index_path and Path(index_path).exists():
            self._load_index()
        
        logger.info("Initialized SparseRetriever")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def _build_index(self):
        """Build BM25 index."""
        logger.info(f"Building BM25 index for {len(self.corpus)} documents")
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully")
    
    def save_index(self, path: str = None):
        """Save BM25 index to disk."""
        save_path = Path(path) if path else self.index_path
        if not save_path:
            raise ValueError("No save path specified")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'corpus': self.corpus,
                'metadata': self.metadata
            }, f)
        
        logger.info(f"BM25 index saved to {save_path}")
    
    def _load_index(self):
        """Load BM25 index from disk."""
        with open(self.index_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.corpus = data['corpus']
            self.metadata = data['metadata']
        
        logger.info(f"BM25 index loaded from {self.index_path}")
    
    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """
        Retrieve documents using BM25.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of retrieved documents with metadata
        """
        if not self.bm25:
            raise ValueError("BM25 index not initialized")
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = scores.argsort()[-k:][::-1]
        
        # Format results
        retrieved_docs = []
        for rank, idx in enumerate(top_k_indices):
            retrieved_docs.append({
                'id': self.metadata[idx].get('id', f'doc_{idx}'),
                'text': self.corpus[idx],
                'metadata': self.metadata[idx],
                'score': float(scores[idx]),
                'rank': rank + 1,
                'retriever': 'sparse'
            })
        
        logger.debug(f"Sparse retrieval found {len(retrieved_docs)} documents")
        return retrieved_docs
