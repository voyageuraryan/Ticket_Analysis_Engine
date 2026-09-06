"""
Data cleaning module for SAP ticket data.
"""

import pandas as pd
import re
from typing import Optional
from loguru import logger


class TicketCleaner:
    """Clean and normalize SAP ticket data."""
    
    def __init__(self):
        self.sap_modules = [
            'Basis', 'HR & Payroll', 'Procurement', 
            'Connections', 'FICO', 'ABAP'
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Clean individual text field.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove ticket IDs (e.g., INC1234567)
        text = re.sub(r'\b(INC|RITM|CHG|PRB)\d+\b', '', text, flags=re.IGNORECASE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def combine_fields(self, summary: str, description: str) -> str:
        """
        Combine summary and description with separator.
        
        Args:
            summary: Ticket summary
            description: Ticket description
            
        Returns:
            Combined text
        """
        summary_clean = self.clean_text(summary)
        description_clean = self.clean_text(description)
        
        return f"{summary_clean} [SEP] {description_clean}"
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean entire dataset.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning dataset with {len(df)} records")
        
        # Create copy
        df_clean = df.copy()
        
        # Handle null Summary (1 record)
        if df_clean['Summary'].isna().any():
            logger.warning("Found null Summary values, filling with 'No summary provided'")
            df_clean['Summary'].fillna('No summary provided', inplace=True)
        
        # Clean text fields
        df_clean['Summary_Clean'] = df_clean['Summary'].apply(self.clean_text)
        df_clean['Description_Clean'] = df_clean['Description'].apply(self.clean_text)
        
        # Create combined field
        df_clean['Combined_Text'] = df_clean.apply(
            lambda row: self.combine_fields(row['Summary'], row['Description']),
            axis=1
        )
        
        # Remove empty records
        initial_len = len(df_clean)
        df_clean = df_clean[df_clean['Combined_Text'].str.len() > 10]
        removed = initial_len - len(df_clean)
        if removed > 0:
            logger.warning(f"Removed {removed} records with insufficient text")
        
        # Validate modules
        invalid_modules = ~df_clean['Module'].isin(self.sap_modules)
        if invalid_modules.any():
            logger.error(f"Found {invalid_modules.sum()} records with invalid modules")
            df_clean = df_clean[~invalid_modules]
        
        logger.info(f"Cleaning complete. Final dataset: {len(df_clean)} records")
        
        return df_clean
