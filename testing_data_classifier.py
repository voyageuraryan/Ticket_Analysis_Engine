"""
Testing Data Classifier
=======================
This script reads the extracted tickets from testing_data_extraction.py,
classifies them using the RAG system, and adds predicted modules for human validation.

Author: SAP Module Classification System
Date: 2026-01-28
"""

import pandas as pd
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.rag.pipeline import SAPModuleClassifier
from src.embeddings.gemini_embedder import GeminiEmbedder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('testing_data_classifier.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Reduce console verbosity
console = logging.StreamHandler()
console.setLevel(logging.INFO)

def find_latest_extraction_file() -> Path:
    """Find the most recent extraction file from testing_data_extraction.py"""
    testing_dir = Path("data/testing")
    if not testing_dir.exists():
        raise FileNotFoundError(f"Testing directory not found: {testing_dir}")
    
    # Find all extraction files
    extraction_files = list(testing_dir.glob("resolved_tickets_*.xlsx"))
    if not extraction_files:
        raise FileNotFoundError(f"No extraction files found in {testing_dir}")
    
    # Get the most recent file
    latest_file = max(extraction_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found latest extraction file: {latest_file.name}")
    return latest_file


def classify_tickets(df: pd.DataFrame, classifier: SAPModuleClassifier) -> pd.DataFrame:
    """
    Classify all tickets in the dataframe using the RAG system.
    
    Args:
        df: DataFrame with columns [Incident_Number, Summary, Description, Owner, ...]
        classifier: Initialized SAPModuleClassifier instance
        
    Returns:
        DataFrame with added columns [Predicted_Module, Confidence, Reasoning]
    """
    logger.info("="*80)
    logger.info("Starting ticket classification")
    logger.info("="*80)
    
    results = []
    total_tickets = len(df)
    
    for idx, row in df.iterrows():
        ticket_num = idx + 1
        incident_number = row.get('Incident_Number', 'UNKNOWN')
        summary = row.get('Summary', '')
        description = row.get('Description', '')
        owner = row.get('Owner', 'N/A')
        
        logger.info(f"\nProcessing ticket {ticket_num}/{total_tickets}: {incident_number}")
        logger.info(f"  Summary: {summary[:80]}...")
        
        # Combine summary and description for classification
        ticket_text = f"{summary}\n{description}"
        
        try:
            # Classify using RAG system
            result = classifier.predict(ticket_text)
            
            predicted_module = result.get('module', 'UNKNOWN')
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('reasoning', 'N/A')
            
            logger.info(f"  ✅ Predicted Module: {predicted_module}")
            logger.info(f"  ✅ Confidence: {confidence:.1%}")
            logger.info(f"  ✅ Owner: {owner}")
            
            results.append({
                'Incident_Number': incident_number,
                'Summary': summary,
                'Description': description,
                'Owner': owner,
                'Predicted_Module': predicted_module,
                'Confidence': f"{confidence:.1%}",
                'Reasoning': reasoning,
                'Status': row.get('Status', 'Resolved'),
                'Resolved_Date': row.get('Resolved_Date', ''),
                'Extracted_At': row.get('Extracted_At', ''),
                'Classification_Status': 'Success'
            })
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"  ❌ Classification failed for {incident_number}: {str(e)}")
            results.append({
                'Incident_Number': incident_number,
                'Summary': summary,
                'Description': description,
                'Owner': owner,
                'Predicted_Module': 'ERROR',
                'Confidence': '0%',
                'Reasoning': f'Classification failed: {str(e)}',
                'Status': row.get('Status', 'Resolved'),
                'Resolved_Date': row.get('Resolved_Date', ''),
                'Extracted_At': row.get('Extracted_At', ''),
                'Classification_Status': 'Failed'
            })
    
    logger.info("\n" + "="*80)
    logger.info("Classification complete")
    logger.info("="*80)
    
    return pd.DataFrame(results)


def save_classified_data(df: pd.DataFrame, output_dir: Path = Path("data/testing/classified")) -> Path:
    """
    Save classified data to Excel file for human validation.
    
    Args:
        df: DataFrame with classification results
        output_dir: Directory to save the output file
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_file = output_dir / f"classified_tickets_{timestamp}.xlsx"
    
    # Save to Excel with formatting
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Classified Tickets')
        
        # Get the worksheet
        worksheet = writer.sheets['Classified Tickets']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)  # Cap at 50
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    logger.info(f"✅ Classified data saved to: {output_file}")
    return output_file


def print_summary(df: pd.DataFrame):
    """Print classification summary statistics"""
    logger.info("\n" + "="*80)
    logger.info("CLASSIFICATION SUMMARY")
    logger.info("="*80)
    
    total = len(df)
    successful = len(df[df['Classification_Status'] == 'Success'])
    failed = len(df[df['Classification_Status'] == 'Failed'])
    
    logger.info(f"Total tickets classified: {total}")
    logger.info(f"Successful classifications: {successful}")
    logger.info(f"Failed classifications: {failed}")
    logger.info(f"Success rate: {(successful/total)*100:.1f}%")
    
    if successful > 0:
        logger.info("\nModule Distribution:")
        module_counts = df[df['Classification_Status'] == 'Success']['Predicted_Module'].value_counts()
        for module, count in module_counts.items():
            logger.info(f"  {module}: {count} tickets")
        
        logger.info("\nOwner Distribution:")
        owner_counts = df['Owner'].value_counts()
        for owner, count in owner_counts.items():
            logger.info(f"  {owner}: {count} tickets")
    
    logger.info("="*80)


def main():
    """Main execution function"""
    try:
        logger.info("="*80)
        logger.info("SAP Module Classification - Testing Data")
        logger.info("="*80)
        
        # Step 1: Find latest extraction file
        logger.info("\nSTEP 1: Loading extracted tickets")
        logger.info("="*80)
        extraction_file = find_latest_extraction_file()
        df = pd.read_excel(extraction_file)
        logger.info(f"✅ Loaded {len(df)} tickets from {extraction_file.name}")
        
        # Step 2: Initialize classifier
        logger.info("\nSTEP 2: Initializing RAG classifier")
        logger.info("="*80)
        
        # Initialize embedder
        embedder = GeminiEmbedder(api_key=Config.GOOGLE_API_KEY)
        
        # Initialize classifier with Gemini API
        classifier = SAPModuleClassifier(
            llm_provider="gemini",
            api_key=Config.GOOGLE_API_KEY,
            vector_store_path=Config.VECTOR_STORE_PATH,
            bm25_index_path=Config.BM25_INDEX_PATH
        )
        
        logger.info("✅ RAG classifier initialized successfully")
        logger.info(f"   - LLM Provider: Gemini API")
        logger.info(f"   - Vector Store: {Config.VECTOR_STORE_PATH}")
        logger.info(f"   - BM25 Index: {Config.BM25_INDEX_PATH}")
        
        # Step 3: Classify tickets
        logger.info("\nSTEP 3: Classifying tickets")
        logger.info("="*80)
        classified_df = classify_tickets(df, classifier)
        
        # Step 4: Save results
        logger.info("\nSTEP 4: Saving classified data")
        logger.info("="*80)
        output_file = save_classified_data(classified_df)
        
        # Step 5: Print summary
        print_summary(classified_df)
        
        logger.info("\n" + "="*80)
        logger.info("✅ CLASSIFICATION COMPLETE!")
        logger.info("="*80)
        logger.info(f"Output file: {output_file}")
        logger.info("\nNext steps:")
        logger.info("1. Open the classified Excel file")
        logger.info("2. Review and validate the predicted modules")
        logger.info("3. Edit any incorrect predictions")
        logger.info("4. Use the validated data for further training/analysis")
        logger.info("="*80)
        
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {str(e)}")
        logger.error("Please run testing_data_extraction.py first to extract tickets.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Classification failed: {str(e)}")
        logger.exception("Full error traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
