"""
Distractor selection for RAFT dataset.
"""

import numpy as np
from typing import List, Dict
from loguru import logger


class DistractorSelector:
    """Select distractor documents for RAFT training."""
    
    def __init__(self, similarity_threshold: float = 0.6):
        """
        Initialize distractor selector.
        
        Args:
            similarity_threshold: Minimum similarity for distractors
        """
        self.similarity_threshold = similarity_threshold
        logger.info(f"Initialized DistractorSelector with threshold={similarity_threshold}")
    
    def select_distractors(
        self,
        target_doc: Dict,
        retrieved_docs: List[Dict],
        n_distractors: int = 2
    ) -> List[Dict]:
        """
        Select distractor documents from different modules.
        
        Strategy: Prioritize high-similarity distractors (confusing), but accept
        any different-module document if needed (better than skipping).
        
        Args:
            target_doc: Target document (golden)
            retrieved_docs: List of retrieved similar documents
            n_distractors: Number of distractors to select
            
        Returns:
            List of distractor documents
        """
        target_module = target_doc['metadata']['Module']
        
        # Filter documents from different modules (basic requirement)
        candidates = [
            doc for doc in retrieved_docs
            if doc['metadata'].get('Module') != target_module
        ]
        
        # If we have NO candidates at all, return empty
        if len(candidates) == 0:
            logger.debug(f"No different-module documents found for {target_module} ticket")
            return []
        
        # Strategy: Try to get high-quality distractors, but accept lower quality
        # rather than skipping the document entirely
        
        # Tier 1: High similarity (very confusing - best quality)
        tier1 = [doc for doc in candidates if doc.get('similarity', 0) >= self.similarity_threshold]
        
        # Tier 2: Medium similarity (somewhat confusing - good quality)
        tier2 = [doc for doc in candidates if 0.3 <= doc.get('similarity', 0) < self.similarity_threshold]
        
        # Tier 3: Any different-module document (low similarity - acceptable quality)
        tier3 = [doc for doc in candidates if doc.get('similarity', 0) < 0.3]
        
        # Select distractors: prioritize higher tiers, but use lower tiers if needed
        selected = []
        
        # Fill from Tier 1 first
        selected.extend(tier1[:n_distractors])
        
        # If not enough, add from Tier 2
        if len(selected) < n_distractors:
            needed = n_distractors - len(selected)
            selected.extend(tier2[:needed])
        
        # If still not enough, add from Tier 3
        if len(selected) < n_distractors:
            needed = n_distractors - len(selected)
            selected.extend(tier3[:needed])
        
        # Sort by similarity (most confusing first)
        selected.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        logger.debug(f"Selected {len(selected)} distractors for {target_module} ticket (T1:{len(tier1[:n_distractors])}, T2:{len(tier2) if len(selected) > len(tier1) else 0}, T3:{len(tier3) if len(selected) > len(tier1) + len(tier2) else 0})")
        
        return selected
    
    def select_golden_docs(
        self,
        target_doc: Dict,
        retrieved_docs: List[Dict],
        n_golden: int = 1
    ) -> List[Dict]:
        """
        Select golden documents from same module.
        
        Args:
            target_doc: Target document
            retrieved_docs: List of retrieved similar documents
            n_golden: Number of golden docs to select
            
        Returns:
            List of golden documents
        """
        target_module = target_doc['metadata']['Module']
        target_id = target_doc.get('id')
        
        # Filter documents from same module (excluding target itself)
        candidates = [
            doc for doc in retrieved_docs
            if doc['metadata'].get('Module') == target_module
            and doc.get('id') != target_id
        ]
        
        # Sort by similarity
        candidates.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Select top-n golden docs
        golden_docs = candidates[:n_golden]
        
        logger.debug(f"Selected {len(golden_docs)} golden docs for {target_module} ticket")
        
        return golden_docs
