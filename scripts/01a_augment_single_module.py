"""
Augment a single module at a time.
This helps avoid rate limits by processing one module per run.

Usage:
    python scripts/01a_augment_single_module.py HR_Payroll
    python scripts/01a_augment_single_module.py Procurement
    python scripts/01a_augment_single_module.py Connections
    python scripts/01a_augment_single_module.py FICO
    python scripts/01a_augment_single_module.py ABAP
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger
import os

from src.config import Config
from src.data_preparation import TicketAugmenter


def main():
    """Augment a single module."""
    
    # Get module name from command line
    if len(sys.argv) < 2:
        print("Usage: python scripts/01a_augment_single_module.py <MODULE_NAME>")
        print("\nAvailable modules:")
        print("  - HR_Payroll")
        print("  - Procurement")
        print("  - Connections")
        print("  - FICO")
        print("  - ABAP")
        print("  - Basis (usually doesn't need augmentation)")
        return
    
    module_arg = sys.argv[1]
    
    # Map command line argument to actual module name
    module_map = {
        'HR_Payroll': 'HR & Payroll',
        'Procurement': 'Procurement',
        'Connections': 'Connections',
        'FICO': 'FICO',
        'ABAP': 'ABAP',
        'Basis': 'Basis'
    }
    
    if module_arg not in module_map:
        print(f"Error: Unknown module '{module_arg}'")
        print("\nAvailable modules:")
        for key in module_map.keys():
            print(f"  - {key}")
        return
    
    module_name = module_map[module_arg]
    
    # Setup logging
    log_dir = Config.OUTPUTS_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / f"augment_{module_arg}.log", rotation="10 MB")
    
    logger.info("=" * 80)
    logger.info(f"AUGMENTING MODULE: {module_name}")
    logger.info("=" * 80)
    
    # Get paths
    paths = Config.get_paths()
    processed_dir = paths['processed']
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load cleaned data
    clean_path = processed_dir / "cleaned_tickets.csv"
    
    if not clean_path.exists():
        logger.error(f"Cleaned data not found: {clean_path}")
        logger.error("Please run scripts/01_prepare_data.py first to clean the data")
        return
    
    logger.info(f"Loading cleaned data from {clean_path}")
    df_clean = pd.read_csv(clean_path)
    logger.info(f"Loaded {len(df_clean)} records")
    
    # Check current count for this module
    module_df = df_clean[df_clean['Module'] == module_name]
    logger.info(f"Current {module_name} records: {len(module_df)}")
    
    # Initialize augmenter
    logger.info("\n" + "=" * 80)
    logger.info("Initializing Augmenter")
    logger.info("=" * 80)
    
    if Config.LLM_PROVIDER == "groq":
        logger.info(f"Using Groq API with model: {Config.GROQ_MODEL}")
        logger.info(f"Delay between requests: {Config.API_DELAY_SECONDS} seconds")
        
        # Get backup keys for automatic rotation
        backup_keys = Config.get_groq_backup_keys()
        if backup_keys:
            logger.info(f"✅ Automatic API key rotation enabled: {len(backup_keys) + 1} keys total")
        
        augmenter = TicketAugmenter(
            llm_provider="groq",
            groq_api_key=Config.GROQ_API_KEY,
            groq_model=Config.GROQ_MODEL,
            target_counts=Config.TARGET_COUNTS,
            groq_api_keys=backup_keys
        )
    elif Config.LLM_PROVIDER == "ollama":
        logger.info(f"Using Ollama with model: {Config.LOCAL_LLM_MODEL}")
        augmenter = TicketAugmenter(
            llm_provider="ollama",
            use_local_llm=True,
            local_model=Config.LOCAL_LLM_MODEL,
            target_counts=Config.TARGET_COUNTS
        )
    else:  # gemini
        logger.info(f"Using Gemini API with model: {Config.LLM_MODEL}")
        augmenter = TicketAugmenter(
            llm_provider="gemini",
            api_key=Config.GOOGLE_API_KEY,
            model_name=Config.LLM_MODEL,
            target_counts=Config.TARGET_COUNTS
        )
    
    # Augment this module
    logger.info("\n" + "=" * 80)
    logger.info(f"Augmenting {module_name}")
    logger.info("=" * 80)
    
    df_augmented_module = augmenter.augment_single_module(
        df_clean,
        module_name,
        delay_seconds=Config.API_DELAY_SECONDS
    )
    
    if len(df_augmented_module) == 0:
        logger.info(f"No augmentation needed for {module_name}")
        return
    
    # Save augmented data for this module
    module_output_path = processed_dir / f"augmented_{module_arg}.csv"
    df_augmented_module.to_csv(module_output_path, index=False)
    logger.info(f"Saved {len(df_augmented_module)} augmented records to {module_output_path}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info(f"AUGMENTATION COMPLETE FOR {module_name}")
    logger.info("=" * 80)
    logger.info(f"Original {module_name} records: {len(module_df)}")
    logger.info(f"Augmented records generated: {len(df_augmented_module)}")
    logger.info(f"Target: {Config.TARGET_COUNTS[module_name]}")
    logger.info(f"\nSaved to: {module_output_path}")
    
    logger.info("\n✅ Next: Run this script for other modules or run scripts/01b_combine_augmented.py to combine all")


if __name__ == "__main__":
    main()
