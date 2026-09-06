"""
RAFT dataset generation with parallel processing support.
"""

import json
from typing import List, Dict, Optional
from loguru import logger
from tqdm import tqdm
import google.generativeai as genai
import ollama
from groq import Groq, RateLimitError
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class ParallelRAFTGenerator:
    """Manage parallel RAFT generation with multiple API keys."""
    
    def __init__(self, generators: List['RAFTGenerator']):
        """Initialize with a list of RAFTGenerator instances."""
        self.generators = generators
        self.num_workers = len(generators)
        logger.info(f"🚀 Initialized ParallelRAFTGenerator with {self.num_workers} workers")
    
    def generate_raft_dataset_parallel(
        self,
        documents: List[Dict],
        distractor_selector,
        hybrid_retriever,
        sap_modules: List[str],
        n_golden: int = 1,
        n_distractors: int = 2,
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 50
    ) -> List[Dict]:
        """Generate RAFT dataset using parallel processing."""
        logger.info(f"Generating RAFT dataset for {len(documents)} documents")
        
        # Load existing checkpoint
        raft_examples = []
        start_idx = 0
        processed_ids = set()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"✅ Found checkpoint at {checkpoint_path}")
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    raft_examples = checkpoint_data.get('examples', [])
                    start_idx = checkpoint_data.get('last_index', 0) + 1
                logger.info(f"✅ Resumed: {len(raft_examples)} examples done, starting from {start_idx}")
            except Exception as e:
                logger.error(f"Checkpoint load failed: {e}")
                start_idx = 0
                raft_examples = []
        
        # Thread-safe lock
        lock = threading.Lock()
        completed_count = len(raft_examples)
        
        def process_document(doc_tuple, generator):
            """Process document with assigned generator."""
            idx, doc = doc_tuple
            
            try:
                # Retrieve similar documents (increased to 50 for better module diversity)
                retrieved = hybrid_retriever.retrieve(query=doc['text'], k=50)
                
                # Select golden and distractor docs
                golden_docs = distractor_selector.select_golden_docs(
                    target_doc=doc, retrieved_docs=retrieved, n_golden=n_golden
                )
                distractor_docs = distractor_selector.select_distractors(
                    target_doc=doc, retrieved_docs=retrieved, n_distractors=n_distractors
                )
                
                # Skip only if we have NO golden docs or NO distractors at all
                if len(golden_docs) == 0 or len(distractor_docs) == 0:
                    return None
                
                # Create RAFT example
                raft_example = generator.create_raft_example(
                    target_doc=doc,
                    golden_docs=golden_docs,
                    distractor_docs=distractor_docs,
                    sap_modules=sap_modules
                )
                
                return (idx, raft_example)
                
            except Exception as e:
                logger.error(f"Failed doc {doc.get('id')}: {e}")
                return None
        
        # Process in parallel
        documents_to_process = list(enumerate(documents[start_idx:], start=start_idx))
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Assign documents to workers in round-robin fashion
            futures = []
            for i, doc_tuple in enumerate(documents_to_process):
                generator = self.generators[i % self.num_workers]
                future = executor.submit(process_document, doc_tuple, generator)
                futures.append(future)
            
            # Collect results with progress bar
            with tqdm(total=len(documents), initial=start_idx, desc="🚀 Parallel RAFT generation") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    
                    if result is not None:
                        idx, raft_example = result
                        
                        with lock:
                            raft_examples.append(raft_example)
                            completed_count += 1
                            
                            # Checkpoint
                            if checkpoint_path and (completed_count % checkpoint_interval == 0):
                                self._save_checkpoint(checkpoint_path, raft_examples, idx)
                                logger.info(f"💾 Checkpoint: {completed_count} examples")
                    
                    pbar.update(1)
        
        # Final checkpoint
        if checkpoint_path:
            self._save_checkpoint(checkpoint_path, raft_examples, len(documents) - 1)
        
        logger.info(f"✅ Generated {len(raft_examples)} RAFT examples")
        return raft_examples
    
    def _save_checkpoint(self, checkpoint_path: str, examples: List[Dict], last_index: int):
        """Save checkpoint."""
        checkpoint_data = {
            'examples': examples,
            'last_index': last_index,
            'total_examples': len(examples)
        }
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    def save_raft_dataset(self, raft_examples: List[Dict], output_path: str):
        """Save RAFT dataset."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(raft_examples, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(raft_examples)} examples to {output_path}")


class RAFTGenerator:
    """Generate RAFT training dataset."""
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash", 
                 llm_provider: str = "groq", groq_api_key: str = None, groq_model: str = "llama-3.1-8b-instant",
                 use_local_llm: bool = False, local_model: str = "gemma:2b",
                 groq_api_keys: Optional[List[str]] = None,
                 worker_id: int = 0):
        """
        Initialize RAFT generator.
        
        Args:
            api_key: Gemini API key (optional if using Groq/Ollama)
            model_name: Model for generating reasoning (Gemini)
            llm_provider: LLM provider to use ("groq", "ollama", or "gemini")
            groq_api_key: Groq API key (for Groq provider)
            groq_model: Groq model name
            use_local_llm: Whether to use local Ollama LLM (legacy parameter)
            local_model: Local Ollama model name
            groq_api_keys: List of backup Groq API keys for rotation
            worker_id: Worker ID for parallel processing (0-based)
        """
        self.llm_provider = llm_provider
        self.groq_model = groq_model
        self.local_model = local_model
        self.worker_id = worker_id
        
        # Setup API key rotation for Groq
        self.groq_api_keys = []
        self.current_key_index = 0
        
        # Initialize based on provider
        if llm_provider == "groq":
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY is required when using Groq provider")
            
            # Build list of API keys (primary + backups)
            self.groq_api_keys = [groq_api_key]
            if groq_api_keys:
                self.groq_api_keys.extend(groq_api_keys)
            
            # For parallel processing, assign a specific key to this worker
            if worker_id < len(self.groq_api_keys):
                assigned_key = self.groq_api_keys[worker_id]
                self.current_key_index = worker_id
            else:
                assigned_key = self.groq_api_keys[0]
            
            self.groq_client = Groq(api_key=assigned_key)
            self.model = None
            
            if worker_id == 0:  # Only log once
                logger.info(f"Initialized RAFTGenerator with Groq model: {groq_model}")
                if len(self.groq_api_keys) > 1:
                    logger.info(f"✅ Parallel processing: {len(self.groq_api_keys)} API keys available")
        elif llm_provider == "ollama" or use_local_llm:
            self.groq_client = None
            self.model = None
            if worker_id == 0:
                logger.info(f"Initialized RAFTGenerator with local Ollama model: {local_model}")
        else:  # gemini
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is required when using Gemini provider")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.groq_client = None
            if worker_id == 0:
                logger.info(f"Initialized RAFTGenerator with Gemini model: {model_name}")
    
    def _rotate_api_key(self) -> bool:
        """Rotate to the next available Groq API key."""
        if self.llm_provider != "groq" or len(self.groq_api_keys) <= 1:
            return False
        
        self.current_key_index += 1
        if self.current_key_index >= len(self.groq_api_keys):
            logger.error("❌ All API keys exhausted!")
            return False
        
        new_key = self.groq_api_keys[self.current_key_index]
        self.groq_client = Groq(api_key=new_key)
        logger.warning(f"🔄 Switched to backup API key #{self.current_key_index + 1}")
        return True
    
    def _call_groq_with_retry(self, prompt: str, max_retries: int = 5) -> Optional[str]:
        """Call Groq API with retry and key rotation."""
        for attempt in range(max_retries):
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=2048
                )
                return response.choices[0].message.content.strip()
            except RateLimitError as e:
                logger.warning(f"⚠️  Rate limit hit (Key #{self.current_key_index + 1}): {str(e)}")
                if self._rotate_api_key():
                    logger.info(f"✅ Continuing with fresh rate limits (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(2)
                    continue
                else:
                    if attempt < max_retries - 1:
                        logger.warning(f"⏳ All keys exhausted, waiting 60 seconds...")
                        time.sleep(60)
                        # Reset to first key after waiting
                        self.current_key_index = 0
                        self.groq_client = Groq(api_key=self.groq_api_keys[0])
                        logger.info("🔄 Reset to first API key")
                    else:
                        return None
            except Exception as e:
                logger.error(f"❌ API call failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return None
        return None
    
    def create_raft_example(
        self,
        target_doc: Dict,
        golden_docs: List[Dict],
        distractor_docs: List[Dict],
        sap_modules: List[str]
    ) -> Dict:
        """
        Create a single RAFT training example.
        
        Args:
            target_doc: Target document to classify
            golden_docs: Golden documents (same module)
            distractor_docs: Distractor documents (different modules)
            sap_modules: List of all SAP modules
            
        Returns:
            RAFT example dict
        """
        # Extract target info
        target_text = target_doc['text']
        target_module = target_doc['metadata']['Module']
        
        # Format question
        question = f"""Given the following SAP ticket, identify the correct module and explain your reasoning.

Ticket:
{target_text}

What SAP module does this ticket belong to?
Available modules: {', '.join(sap_modules)}"""
        
        # Format context documents
        context = []
        
        # Add golden documents
        for i, doc in enumerate(golden_docs):
            context.append({
                "type": "golden",
                "doc_id": f"context_{i+1}",
                "module": doc['metadata']['Module'],
                "text": doc['text'][:500]  # Truncate for context
            })
        
        # Add distractor documents
        for i, doc in enumerate(distractor_docs):
            context.append({
                "type": "distractor",
                "doc_id": f"context_{len(golden_docs)+i+1}",
                "module": doc['metadata']['Module'],
                "text": doc['text'][:500]  # Truncate for context
            })
        
        # Generate reasoning answer
        answer = self._generate_reasoning(
            target_text=target_text,
            target_module=target_module,
            context=context,
            sap_modules=sap_modules
        )
        
        return {
            "question": question,
            "context": context,
            "answer": answer,
            "target_module": target_module,
            "target_id": target_doc.get('id')
        }
    
    def _generate_reasoning(
        self,
        target_text: str,
        target_module: str,
        context: List[Dict],
        sap_modules: List[str]
    ) -> str:
        """
        Generate chain-of-thought reasoning for answer.
        
        Args:
            target_text: Target ticket text
            target_module: Correct module
            context: Context documents
            sap_modules: List of modules
            
        Returns:
            Reasoning text
        """
        # Format context for prompt
        context_text = "\n\n".join([
            f"Context Document {i+1} ({ctx['type']}):\nModule: {ctx['module']}\n{ctx['text']}"
            for i, ctx in enumerate(context)
        ])
        
        prompt = f"""You are an expert SAP support analyst. Analyze the following ticket and provide a detailed classification with reasoning.

Target Ticket:
{target_text}

Context Documents (some may be distractors):
{context_text}

Task: Classify this ticket into one of these SAP modules: {', '.join(sap_modules)}

Provide your answer in this format:

Module: [module name]

Reasoning:
1. [First reasoning step]
2. [Second reasoning step]
3. [Third reasoning step]

Key Indicators:
- [Important term 1]: [Why it's relevant]
- [Important term 2]: [Why it's relevant]

Confidence: [High/Medium/Low]

Now provide your analysis:"""
        
        try:
            # Generate response based on provider
            if self.llm_provider == "groq":
                result = self._call_groq_with_retry(prompt)
                if result is None:
                    logger.error("Failed to generate reasoning after retries")
                    return f"Module: {target_module}\n\nReasoning:\nBased on the ticket content, this is classified as {target_module}.\n\nConfidence: High"
                return result
            elif self.llm_provider == "ollama":
                response = ollama.generate(model=self.local_model, prompt=prompt)
                return response['response'].strip()
            else:  # gemini
                response = self.model.generate_content(prompt)
                return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate reasoning: {e}")
            # Fallback simple answer
            return f"Module: {target_module}\n\nReasoning:\nBased on the ticket content, this is classified as {target_module}.\n\nConfidence: High"
    
    def generate_raft_dataset(
        self,
        documents: List[Dict],
        distractor_selector,
        hybrid_retriever,
        sap_modules: List[str],
        n_golden: int = 1,
        n_distractors: int = 2,
        delay: float = 0.0,
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 50
    ) -> List[Dict]:
        """
        Generate complete RAFT dataset with checkpoint support.
        
        Args:
            documents: List of all documents
            distractor_selector: DistractorSelector instance
            hybrid_retriever: HybridRetriever instance
            sap_modules: List of SAP modules
            n_golden: Number of golden docs per example
            n_distractors: Number of distractors per example
            delay: Delay between API calls
            checkpoint_path: Path to save checkpoints (for resume capability)
            checkpoint_interval: Save checkpoint every N examples
            
        Returns:
            List of RAFT examples
        """
        logger.info(f"Generating RAFT dataset for {len(documents)} documents")
        
        # Load existing checkpoint if available
        raft_examples = []
        start_idx = 0
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"✅ Found checkpoint at {checkpoint_path}")
            logger.info("Loading existing progress...")
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    raft_examples = checkpoint_data.get('examples', [])
                    start_idx = checkpoint_data.get('last_index', 0) + 1
                logger.info(f"✅ Resumed from checkpoint: {len(raft_examples)} examples already generated")
                logger.info(f"   Starting from document {start_idx}/{len(documents)}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Starting from scratch...")
                start_idx = 0
                raft_examples = []
        
        # Process documents with clean progress bar
        pbar = tqdm(
            documents[start_idx:], 
            desc="🚀 RAFT Generation",
            initial=start_idx, 
            total=len(documents),
            unit="doc",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for idx, doc in enumerate(pbar):
            actual_idx = start_idx + idx
            
            try:
                # Retrieve similar documents (increased to 50 for better module diversity)
                retrieved = hybrid_retriever.retrieve(
                    query=doc['text'],
                    k=50  # More candidates = better chance of finding different modules
                )
                
                # Select golden and distractor docs
                golden_docs = distractor_selector.select_golden_docs(
                    target_doc=doc,
                    retrieved_docs=retrieved,
                    n_golden=n_golden
                )
                
                distractor_docs = distractor_selector.select_distractors(
                    target_doc=doc,
                    retrieved_docs=retrieved,
                    n_distractors=n_distractors
                )
                
                # Skip only if we have NO golden docs or NO distractors at all
                # Allow examples with fewer distractors than requested (better than skipping)
                if len(golden_docs) == 0 or len(distractor_docs) == 0:
                    pbar.write(f"⚠️  Skipped {doc.get('id')} - no context")
                    continue
                
                # Create RAFT example
                raft_example = self.create_raft_example(
                    target_doc=doc,
                    golden_docs=golden_docs,
                    distractor_docs=distractor_docs,
                    sap_modules=sap_modules
                )
                
                raft_examples.append(raft_example)
                
                # Update progress bar with current stats
                pbar.set_postfix({
                    'examples': len(raft_examples),
                    'skipped': actual_idx + 1 - len(raft_examples)
                })
                
                # Save checkpoint periodically
                if checkpoint_path and (len(raft_examples) % checkpoint_interval == 0):
                    self._save_checkpoint(checkpoint_path, raft_examples, actual_idx)
                    pbar.write(f"💾 Checkpoint: {len(raft_examples)} examples saved")
                
                time.sleep(delay)  # Rate limiting
                
            except Exception as e:
                pbar.write(f"❌ Failed: {doc.get('id')}")
                continue
        
        pbar.close()
        
        # Save final checkpoint
        if checkpoint_path:
            self._save_checkpoint(checkpoint_path, raft_examples, len(documents) - 1)
            logger.info(f"💾 Final checkpoint saved")
        
        logger.info(f"Generated {len(raft_examples)} RAFT examples")
        return raft_examples
    
    def _save_checkpoint(self, checkpoint_path: str, examples: List[Dict], last_index: int):
        """Save checkpoint with current progress."""
        checkpoint_data = {
            'examples': examples,
            'last_index': last_index,
            'total_examples': len(examples)
        }
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    def generate_raft_dataset_parallel(
        self,
        documents: List[Dict],
        distractor_selector,
        hybrid_retriever,
        sap_modules: List[str],
        n_golden: int = 1,
        n_distractors: int = 2,
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 50,
        num_workers: int = 3
    ) -> List[Dict]:
        """
        Generate RAFT dataset using parallel processing with multiple API keys.
        
        Args:
            documents: List of all documents
            distractor_selector: DistractorSelector instance
            hybrid_retriever: HybridRetriever instance
            sap_modules: List of SAP modules
            n_golden: Number of golden docs per example
            n_distractors: Number of distractors per example
            checkpoint_path: Path to save checkpoints
            checkpoint_interval: Save checkpoint every N examples
            num_workers: Number of parallel workers (should match number of API keys)
            
        Returns:
            List of RAFT examples
        """
        logger.info(f"🚀 Parallel processing enabled with {num_workers} workers")
        logger.info(f"Generating RAFT dataset for {len(documents)} documents")
        
        # Load existing checkpoint if available
        raft_examples = []
        start_idx = 0
        processed_ids = set()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"✅ Found checkpoint at {checkpoint_path}")
            logger.info("Loading existing progress...")
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    raft_examples = checkpoint_data.get('examples', [])
                    start_idx = checkpoint_data.get('last_index', 0) + 1
                    processed_ids = set(ex['question'].split('Ticket:')[1].split('\n')[0].strip() 
                                      for ex in raft_examples if 'question' in ex)
                logger.info(f"✅ Resumed from checkpoint: {len(raft_examples)} examples already generated")
                logger.info(f"   Starting from document {start_idx}/{len(documents)}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Starting from scratch...")
                start_idx = 0
                raft_examples = []
                processed_ids = set()
        
        # Thread-safe lock for updating results
        lock = threading.Lock()
        
        def process_document(doc_tuple):
            """Process a single document (thread-safe)."""
            idx, doc = doc_tuple
            
            # Skip if already processed
            if doc['id'] in processed_ids:
                return None
            
            try:
                # Retrieve similar documents
                retrieved = hybrid_retriever.retrieve(
                    query=doc['text'],
                    k=20
                )
                
                # Select golden and distractor docs
                golden_docs = distractor_selector.select_golden_docs(
                    target_doc=doc,
                    retrieved_docs=retrieved,
                    n_golden=n_golden
                )
                
                distractor_docs = distractor_selector.select_distractors(
                    target_doc=doc,
                    retrieved_docs=retrieved,
                    n_distractors=n_distractors
                )
                
                # Skip if not enough context
                if len(golden_docs) == 0 or len(distractor_docs) < n_distractors:
                    logger.warning(f"Skipping document {doc.get('id')} - insufficient context")
                    return None
                
                # Create RAFT example
                raft_example = self.create_raft_example(
                    target_doc=doc,
                    golden_docs=golden_docs,
                    distractor_docs=distractor_docs,
                    sap_modules=sap_modules
                )
                
                return (idx, raft_example)
                
            except Exception as e:
                logger.error(f"Failed to create RAFT example for doc {doc.get('id')}: {e}")
                return None
        
        # Process documents in parallel
        documents_to_process = list(enumerate(documents[start_idx:], start=start_idx))
        completed_count = len(raft_examples)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_doc = {executor.submit(process_document, doc_tuple): doc_tuple 
                           for doc_tuple in documents_to_process}
            
            # Process completed tasks with progress bar
            with tqdm(total=len(documents), initial=start_idx, desc="Generating RAFT examples") as pbar:
                for future in as_completed(future_to_doc):
                    result = future.result()
                    
                    if result is not None:
                        idx, raft_example = result
                        
                        with lock:
                            raft_examples.append(raft_example)
                            completed_count += 1
                            
                            # Save checkpoint periodically
                            if checkpoint_path and (completed_count % checkpoint_interval == 0):
                                self._save_checkpoint(checkpoint_path, raft_examples, idx)
                                logger.info(f"💾 Checkpoint saved: {completed_count} examples")
                    
                    pbar.update(1)
        
        # Save final checkpoint
        if checkpoint_path:
            self._save_checkpoint(checkpoint_path, raft_examples, len(documents) - 1)
            logger.info(f"💾 Final checkpoint saved")
        
        logger.info(f"Generated {len(raft_examples)} RAFT examples")
        return raft_examples
    
    def save_raft_dataset(self, raft_examples: List[Dict], output_path: str):
        """Save RAFT dataset to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(raft_examples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(raft_examples)} RAFT examples to {output_path}")
