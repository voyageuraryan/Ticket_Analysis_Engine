"""
Hybrid retrieval combining dense and sparse methods.
"""

from typing import List, Dict
from loguru import logger
from collections import defaultdict


class HybridRetriever:
    """Hybrid retrieval using Reciprocal Rank Fusion (RRF)."""
    
    def __init__(self, dense_retriever, sparse_retriever, k: int = 60):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_retriever: DenseRetriever instance
            sparse_retriever: SparseRetriever instance
            k: RRF constant (default 60)
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.k = k
        logger.info(f"Initialized HybridRetriever with RRF k={k}")
    
    def _reciprocal_rank_fusion(
        self, 
        dense_results: List[Dict], 
        sparse_results: List[Dict]
    ) -> List[Dict]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            
        Returns:
            Fused and re-ranked results
        """
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        doc_data = {}
        
        # Process dense results
        for doc in dense_results:
            doc_id = doc['id']
            rank = doc['rank']
            rrf_scores[doc_id] += 1.0 / (self.k + rank)
            doc_data[doc_id] = doc
        
        # Process sparse results
        for doc in sparse_results:
            doc_id = doc['id']
            rank = doc['rank']
            rrf_scores[doc_id] += 1.0 / (self.k + rank)
            
            # Keep doc data (prefer dense if exists)
            if doc_id not in doc_data:
                doc_data[doc_id] = doc
        
        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Format results
        fused_results = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            doc = doc_data[doc_id].copy()
            doc['rrf_score'] = score
            doc['hybrid_rank'] = rank + 1
            fused_results.append(doc)
        
        return fused_results
    
    def retrieve(self, query: str, k: int = 10, strategy: str = 'rrf') -> List[Dict]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Query text
            k: Number of final results to return
            strategy: Fusion strategy ('rrf', 'dense_only', 'sparse_only')
            
        Returns:
            List of retrieved documents
        """
        if strategy == 'dense_only':
            return self.dense_retriever.retrieve(query, k=k)
        elif strategy == 'sparse_only':
            return self.sparse_retriever.retrieve(query, k=k)
        
        # Retrieve from both methods (get more candidates for fusion)
        retrieve_k = k * 2
        dense_results = self.dense_retriever.retrieve(query, k=retrieve_k)
        sparse_results = self.sparse_retriever.retrieve(query, k=retrieve_k)
        
        # Fuse results
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)
        
        # Return top-k
        final_results = fused_results[:k]
        
        logger.debug(f"Hybrid retrieval returned {len(final_results)} documents")
        return final_results
    
    def retrieve_with_diversity(
        self, 
        query: str, 
        k: int = 10, 
        ensure_modules: List[str] = None
    ) -> List[Dict]:
        """
        Retrieve documents with module diversity (for RAFT).
        
        Args:
            query: Query text
            k: Number of results
            ensure_modules: List of modules to ensure representation
            
        Returns:
            Diverse set of retrieved documents
        """
        # Get more candidates
        candidates = self.retrieve(query, k=k*3)
        
        if not ensure_modules:
            return candidates[:k]
        
        # Ensure diversity
        selected = []
        modules_seen = set()
        
        # First pass: Get at least one from each module
        for doc in candidates:
            module = doc['metadata'].get('Module')
            if module in ensure_modules and module not in modules_seen:
                selected.append(doc)
                modules_seen.add(module)
            if len(modules_seen) == len(ensure_modules):
                break
        
        # Second pass: Fill remaining slots with top-ranked
        for doc in candidates:
            if doc not in selected:
                selected.append(doc)
            if len(selected) == k:
                break
        
        logger.info(f"Retrieved {len(selected)} documents with diversity across {len(modules_seen)} modules")
        return selected
