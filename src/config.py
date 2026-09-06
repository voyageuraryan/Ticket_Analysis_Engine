"""
Configuration management for SAP Module Classifier.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict

# Load environment variables
load_dotenv()

class Config:
    """Application configuration."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    
    # API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Backup Groq API Keys for automatic rotation
    GROQ_API_KEY_BACKUP1 = os.getenv("GROQ_API_KEY_BACKUP1")
    GROQ_API_KEY_BACKUP2 = os.getenv("GROQ_API_KEY_BACKUP2")
    GROQ_API_KEY_BACKUP3 = os.getenv("GROQ_API_KEY_BACKUP3")
    
    # Model Settings
    # Note: For deprecated google.generativeai library, use models/gemini-embedding-001
    # For new google.genai library (when upgraded), use text-embedding-004
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # LLM Provider Selection
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq, ollama, or gemini
    
    # Groq Settings (Fast cloud inference)
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    # Local LLM Settings (Ollama)
    USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "gemma:2b")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Retrieval Settings
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))
    RRF_K = int(os.getenv("RRF_K", "60"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
    
    # Application Settings
    APP_ENV = os.getenv("APP_ENV", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    
    # Vector Store Settings
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sap_tickets")
    
    # Data Paths
    RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/Historic_Data_CSV.xlsx")
    PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed")
    RAFT_DATA_PATH = os.getenv("RAFT_DATA_PATH", "data/raft")
    
    # Augmentation Settings
    TARGET_COUNTS = {
        'Basis': int(os.getenv("TARGET_BASIS", "4302")),
        'HR & Payroll': int(os.getenv("TARGET_HR_PAYROLL", "2000")),
        'Procurement': int(os.getenv("TARGET_PROCUREMENT", "1500")),
        'Connections': int(os.getenv("TARGET_CONNECTIONS", "1500")),
        'FICO': int(os.getenv("TARGET_FICO", "1500")),
        'ABAP': int(os.getenv("TARGET_ABAP", "1500"))
    }
    
    # Training Settings
    TRAIN_SIZE = float(os.getenv("TRAIN_SIZE", "0.7"))
    VAL_SIZE = float(os.getenv("VAL_SIZE", "0.15"))
    TEST_SIZE = float(os.getenv("TEST_SIZE", "0.15"))
    RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
    
    # API Rate Limiting
    # Groq free tier: 30 RPM, 6000 TPM - use 2.0s delay to stay at exactly 30 req/min
    API_DELAY_SECONDS = float(os.getenv("API_DELAY_SECONDS", "2.0"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    
    # SAP Modules
    SAP_MODULES = ['Basis', 'HR & Payroll', 'Procurement', 'Connections', 'FICO', 'ABAP']
    
    @classmethod
    def validate(cls):
        """Validate configuration."""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set in environment variables")
        
        # Create directories if they don't exist
        for path in [cls.DATA_DIR, cls.OUTPUTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)
        
        return True
    
    @classmethod
    def get_paths(cls) -> Dict[str, Path]:
        """Get all configured paths."""
        return {
            'base': cls.BASE_DIR,
            'data': cls.DATA_DIR,
            'outputs': cls.OUTPUTS_DIR,
            'raw_data': cls.BASE_DIR / cls.RAW_DATA_PATH,
            'processed': cls.BASE_DIR / cls.PROCESSED_DATA_PATH,
            'raft': cls.BASE_DIR / cls.RAFT_DATA_PATH,
            'vector_store': cls.BASE_DIR / cls.VECTOR_STORE_PATH,
            'logs': cls.OUTPUTS_DIR / 'logs',
            'results': cls.OUTPUTS_DIR / 'results'
        }
    
    @classmethod
    def get_groq_backup_keys(cls) -> list:
        """Get list of backup Groq API keys (excluding None values)."""
        backup_keys = []
        # Support up to 5 backup keys (6 total with primary)
        for i in range(1, 6):
            key = os.getenv(f"GROQ_API_KEY_BACKUP{i}")
            if key:
                backup_keys.append(key)
        return backup_keys


# Validate on import
if __name__ != "__main__":
    try:
        Config.validate()
    except ValueError as e:
        print(f"⚠️  Configuration Warning: {e}")
        print("Please create a .env file with your GOOGLE_API_KEY")
