"""
Create RAFT dataset with golden and distractor documents.
This is optional but improves model performance.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.config import Config
from src.embeddings import GeminiEmbedder
from src.vector_store import ChromaVectorStore
from src.retrieval import DenseRetriever, SparseRetriever, HybridRetriever
from src.raft import RAFTGenerator, DistractorSelector, ParallelRAFTGenerator


def main():
    """Create RAFT dataset."""
    
    # Setup logging
    log_dir = Config.OUTPUTS_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default logger (console output)
    logger.remove()
    
    # Add file logger (all levels)
    logger.add(log_dir / "create_raft.log", rotation="10 MB", level="DEBUG")
    
    # Add console logger (only INFO and above, no DEBUG spam)
    logger.add(
        lambda msg: print(msg, end=''),
        format="<level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    logger.info("=" * 80)
    logger.info("\nSAP MODULE CLASSIFIER - CREATE RAFT DATASET")
    logger.info("=" * 80)
    
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return
    
    # Get paths
    paths = Config.get_paths()
    train_path = paths['processed'] / "train_test_split" / "train.csv"
    
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error("Please run scripts/01_prepare_data.py first")
        return
    
    # Load training data (sample for RAFT - full dataset would be too expensive)
    logger.info("\nLoading training data...")
    train_df = pd.read_csv(train_path)
    
    # Use ALL tickets for RAFT (full dataset for better performance)
    # For small datasets (8,611 samples), using all data is recommended
    logger.info("Preparing RAFT dataset...")
    logger.info("Using FULL dataset (all samples) for optimal model performance")
    
    raft_df = train_df.copy()
    
    # Show distribution
    logger.info(f"\nTotal samples: {len(raft_df)}")
    logger.info("Distribution per module:")
    for module in Config.SAP_MODULES:
        count = len(raft_df[raft_df['Module'] == module])
        logger.info(f"  {module}: {count} samples")
    
    # Initialize components
    logger.info("\nInitializing components...")
    
    embedder = GeminiEmbedder(
        api_key=Config.GOOGLE_API_KEY,
        model_name=Config.EMBEDDING_MODEL
    )
    
    vector_store = ChromaVectorStore(persist_directory=str(paths['vector_store']))
    vector_store.create_collection(name=Config.COLLECTION_NAME, reset=False)
    
    bm25_index_path = Config.DATA_DIR / "embeddings" / "bm25_index.pkl"
    sparse_retriever = SparseRetriever(index_path=str(bm25_index_path))
    
    dense_retriever = DenseRetriever(vector_store, embedder)
    hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever, k=Config.RRF_K)
    
    distractor_selector = DistractorSelector(similarity_threshold=Config.SIMILARITY_THRESHOLD)
    
    # Initialize RAFT generator(s) based on LLM provider
    if Config.LLM_PROVIDER == "groq":
        logger.info(f"Using Groq API with model: {Config.GROQ_MODEL}")
        
        # Collect all API keys
        all_api_keys = [Config.GROQ_API_KEY]
        backup_keys = Config.get_groq_backup_keys()
        if backup_keys:
            all_api_keys.extend(backup_keys)
        
        logger.info(f"🔑 Total API keys available: {len(all_api_keys)}")
        
        # Strategy: Use rotation for better rate limit handling
        # Parallel processing can cause all workers to hit limits simultaneously
        # Rotation allows continuous processing by switching keys on rate limits
        
        if len(all_api_keys) >= 4:
            # With 4+ keys, use rotation strategy (better for rate limits)
            logger.info(f"🔄 Using API key ROTATION strategy")
            logger.info(f"   Benefit: Continuous processing, automatic rate limit recovery")
            
            raft_generator = RAFTGenerator(
                llm_provider="groq",
                groq_api_key=Config.GROQ_API_KEY,
                groq_model=Config.GROQ_MODEL,
                groq_api_keys=backup_keys,  # Pass all backup keys for rotation
                worker_id=0
            )
            use_parallel = False
            
        else:
            # With 3 or fewer keys, use parallel processing
            logger.info(f"🚀 Using PARALLEL processing strategy")
            logger.info(f"   Expected speedup: ~{len(all_api_keys)}x faster")
            
            generators = []
            for i, api_key in enumerate(all_api_keys):
                generator = RAFTGenerator(
                    llm_provider="groq",
                    groq_api_key=api_key,
                    groq_model=Config.GROQ_MODEL,
                    groq_api_keys=[],  # Each worker uses its own key
                    worker_id=i
                )
                generators.append(generator)
            
            raft_generator = ParallelRAFTGenerator(generators)
            use_parallel = True
        
    else:
        logger.info(f"Using Gemini API with model: {Config.LLM_MODEL}")
        raft_generator = RAFTGenerator(
            api_key=Config.GOOGLE_API_KEY,
            model_name=Config.LLM_MODEL,
            llm_provider="gemini"
        )
        use_parallel = False
    
    # Prepare documents
    logger.info("\nPreparing documents...")
    documents = []
    for idx, row in raft_df.iterrows():
        documents.append({
            'id': str(row['Incident']),
            'text': row['Combined_Text'],
            'metadata': {
                'Module': row['Module'],
                'Priority': row['Priority'],
                'Team': row['Team']
            }
        })
    
    # Generate RAFT dataset
    logger.info("\nGenerating RAFT dataset...")
    logger.info("This will take a while due to API calls...")
    
    # Setup checkpoint path
    raft_dir = paths['raft']
    raft_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = raft_dir / "raft_checkpoint.json"
    
    if use_parallel:
        # Parallel processing - much faster!
        num_workers = len(generators)
        estimated_time = (len(documents) * Config.API_DELAY_SECONDS / 60) / num_workers
        logger.info(f"🚀 Parallel mode: {num_workers} workers")
        logger.info(f"⚡ Estimated time: ~{estimated_time:.1f} minutes (~{estimated_time/60:.1f} hours)")
        logger.info("💾 Checkpoint system enabled - progress saved every 25 examples")
        
        raft_examples = raft_generator.generate_raft_dataset_parallel(
            documents=documents,
            distractor_selector=distractor_selector,
            hybrid_retriever=hybrid_retriever,
            sap_modules=Config.SAP_MODULES,
            n_golden=1,
            n_distractors=2,
            checkpoint_path=str(checkpoint_path),
            checkpoint_interval=25
        )
    else:
        # Sequential processing
        estimated_time = len(documents) * Config.API_DELAY_SECONDS / 60
        logger.info(f"Estimated time: ~{estimated_time:.1f} minutes")
        logger.info("💾 Checkpoint system enabled - progress saved every 25 examples")
        
        raft_examples = raft_generator.generate_raft_dataset(
            documents=documents,
            distractor_selector=distractor_selector,
            hybrid_retriever=hybrid_retriever,
            sap_modules=Config.SAP_MODULES,
            n_golden=1,
            n_distractors=2,
            delay=Config.API_DELAY_SECONDS,
            checkpoint_path=str(checkpoint_path),
            checkpoint_interval=25
        )
    
    # Save RAFT dataset
    raft_dir = paths['raft']
    raft_dir.mkdir(parents=True, exist_ok=True)
    
    raft_path = raft_dir / "raft_dataset.json"
    raft_generator.save_raft_dataset(raft_examples, str(raft_path))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("RAFT DATASET CREATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total RAFT examples: {len(raft_examples)}")
    logger.info(f"Saved to: {raft_path}")
    
    # Module distribution
    module_counts = {}
    for example in raft_examples:
        module = example['target_module']
        module_counts[module] = module_counts.get(module, 0) + 1
    
    logger.info("\nRAFT examples per module:")
    for module, count in sorted(module_counts.items()):
        logger.info(f"  {module}: {count}")
    
    logger.info("\n✅ RAFT dataset ready for training/fine-tuning")
    logger.info("   You can now use this dataset to improve the model")


if __name__ == "__main__":
    main()
