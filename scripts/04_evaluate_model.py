"""
Evaluate model on test set.
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
from src.rag import SAPModuleClassifier
from src.evaluation import ModelEvaluator


def main():
    """Evaluate model on test set."""
    
    # Setup logging
    log_dir = Config.OUTPUTS_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "evaluation.log", rotation="10 MB")
    
    logger.info("=" * 80)
    logger.info("SAP MODULE CLASSIFIER - MODEL EVALUATION")
    logger.info("=" * 80)
    
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return
    
    # Get paths
    paths = Config.get_paths()
    test_path = paths['processed'] / "train_test_split" / "test.csv"
    
    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        logger.error("Please run scripts/01_prepare_data.py first")
        return
    
    # Load test data
    logger.info("\nLoading test data...")
    test_df = pd.read_csv(test_path)
    logger.info(f"Loaded {len(test_df)} test samples")
    
    # Sample for faster evaluation (optional)
    if len(test_df) > 200:
        logger.info(f"Sampling 200 test samples for faster evaluation...")
        test_df = test_df.sample(200, random_state=Config.RANDOM_SEED)
    
    # Initialize components
    logger.info("\nInitializing classifier...")
    
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
    
    # Initialize classifier based on LLM provider
    if Config.LLM_PROVIDER == "groq":
        logger.info(f"Using Groq API with model: {Config.GROQ_MODEL}")
        
        # Collect all API keys for rotation
        all_api_keys = [Config.GROQ_API_KEY]
        backup_keys = Config.get_groq_backup_keys()
        if backup_keys:
            all_api_keys.extend(backup_keys)
        
        logger.info(f"🔑 Total API keys available: {len(all_api_keys)}")
        logger.info(f"🔄 Using API key ROTATION strategy for evaluation")
        
        classifier = SAPModuleClassifier(
            llm_provider="groq",
            api_key=Config.GROQ_API_KEY,
            groq_api_keys=backup_keys,
            hybrid_retriever=hybrid_retriever,
            sap_modules=Config.SAP_MODULES,
            model_name=Config.GROQ_MODEL,
            temperature=Config.TEMPERATURE,
            use_raft=True
        )
    else:
        logger.info(f"Using Gemini API with model: {Config.LLM_MODEL}")
        
        classifier = SAPModuleClassifier(
            llm_provider="gemini",
            api_key=Config.GOOGLE_API_KEY,
            hybrid_retriever=hybrid_retriever,
            sap_modules=Config.SAP_MODULES,
            model_name=Config.LLM_MODEL,
            temperature=Config.TEMPERATURE,
            use_raft=True
        )
    
    # Evaluate
    logger.info("\nRunning evaluation...")
    logger.info("This may take a while...")
    
    evaluator = ModelEvaluator(classifier, Config.SAP_MODULES)
    results = evaluator.evaluate(
        test_df=test_df,
        output_dir=str(paths['results'])
    )
    
    logger.info("\n✅ Evaluation complete!")
    logger.info(f"   Results saved to: {paths['results']}")


if __name__ == "__main__":
    main()
