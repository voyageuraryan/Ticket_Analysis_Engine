"""
Streamlit web interface for SAP Module Classifier.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from loguru import logger

from src.config import Config
from src.embeddings import GeminiEmbedder
from src.vector_store import ChromaVectorStore
from src.retrieval import DenseRetriever, SparseRetriever, HybridRetriever
from src.rag import SAPModuleClassifier


# Page config
st.set_page_config(
    page_title="SAP Module Classifier",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .high-confidence {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .medium-confidence {
        background-color: #fff3cd;
        border-color: #ffc107;
    }
    .low-confidence {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_classifier():
    """Initialize classifier (cached)."""
    try:
        Config.validate()
        
        embedder = GeminiEmbedder(
            api_key=Config.GOOGLE_API_KEY,
            model_name=Config.EMBEDDING_MODEL
        )
        
        vector_store = ChromaVectorStore(
            persist_directory=str(Config.BASE_DIR / Config.VECTOR_STORE_PATH)
        )
        vector_store.create_collection(name=Config.COLLECTION_NAME, reset=False)
        
        bm25_index_path = Config.DATA_DIR / "embeddings" / "bm25_index.pkl"
        if not bm25_index_path.exists():
            st.error("BM25 index not found. Please run scripts/02_build_vector_store.py first")
            return None
        
        sparse_retriever = SparseRetriever(index_path=str(bm25_index_path))
        dense_retriever = DenseRetriever(vector_store, embedder)
        hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever, k=Config.RRF_K)
        
        # Streamlit app always uses Gemini API
        classifier = SAPModuleClassifier(
            llm_provider="gemini",
            api_key=Config.GOOGLE_API_KEY,
            hybrid_retriever=hybrid_retriever,
            sap_modules=Config.SAP_MODULES,
            model_name=Config.LLM_MODEL,
            temperature=Config.TEMPERATURE,
            use_raft=True
        )
        
        return classifier
    
    except Exception as e:
        st.error(f"Failed to initialize classifier: {e}")
        return None


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 SAP Module Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligent ticket classification using Hybrid RAG with RAFT")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This tool uses a **Hybrid RAG system** to classify SAP support tickets into modules:
        
        - **Basis**: System administration
        - **HR & Payroll**: Employee management
        - **Procurement**: Purchasing
        - **Connections**: Integrations
        - **FICO**: Finance & Controlling
        - **ABAP**: Development
        
        **Features:**
        - Semantic + Keyword search
        - RAFT method with distractors
        - Explainable predictions
        - Confidence scoring
        """)
        
        st.header("⚙️ Settings")
        top_k = st.slider("Similar tickets to retrieve", 5, 20, 10)
        show_similar = st.checkbox("Show similar tickets", value=True)
        show_reasoning = st.checkbox("Show detailed reasoning", value=True)
    
    # Initialize classifier
    with st.spinner("Initializing classifier..."):
        classifier = initialize_classifier()
    
    if classifier is None:
        st.error("❌ Classifier initialization failed. Please check configuration.")
        st.info("Make sure you have:")
        st.code("""
1. Created .env file with GOOGLE_API_KEY
2. Run scripts/01_prepare_data.py
3. Run scripts/02_build_vector_store.py
        """)
        return
    
    st.success("✅ Classifier ready!")
    
    # Main interface
    st.markdown("---")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Manual Entry", "Batch Upload (CSV)"],
        horizontal=True
    )
    
    if input_method == "Manual Entry":
        # Manual entry form
        with st.form("ticket_form"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                incident_number = st.text_input("Incident Number (optional)", placeholder="INC1234567")
            
            with col2:
                summary = st.text_input("Ticket Summary *", placeholder="Brief description of the issue")
            
            description = st.text_area(
                "Ticket Description *",
                placeholder="Detailed description of the issue, error messages, steps to reproduce, etc.",
                height=150
            )
            
            submitted = st.form_submit_button("🔍 Classify Ticket", use_container_width=True)
        
        if submitted:
            if not summary or not description:
                st.error("❌ Please provide both summary and description")
            else:
                with st.spinner("Analyzing ticket..."):
                    result = classifier.predict(
                        summary=summary,
                        description=description,
                        incident_number=incident_number,
                        top_k=top_k
                    )
                
                # Display results
                display_prediction(result, show_similar, show_reasoning)
    
    else:
        # Batch upload
        st.markdown("### 📤 Batch Upload")
        st.info("Upload a CSV file with columns: `summary`, `description`, and optionally `incident_number`")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.write(f"Loaded {len(df)} tickets")
            st.dataframe(df.head())
            
            if st.button("🚀 Classify All Tickets"):
                with st.spinner(f"Classifying {len(df)} tickets..."):
                    tickets = df.to_dict('records')
                    results = classifier.predict_batch(tickets, top_k=top_k)
                
                # Create results dataframe
                results_df = pd.DataFrame([
                    {
                        'Incident': r['input'].get('incident_number', ''),
                        'Summary': r['input']['summary'][:50] + '...',
                        'Predicted_Module': r['module'],
                        'Confidence': f"{r['confidence']:.2%}",
                        'Needs_Review': '⚠️' if r['needs_review'] else '✅'
                    }
                    for r in results
                ])
                
                st.success(f"✅ Classified {len(results)} tickets!")
                st.dataframe(results_df)
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="classification_results.csv",
                    mime="text/csv"
                )


def display_prediction(result, show_similar=True, show_reasoning=True):
    """Display prediction results."""
    
    # Determine confidence class
    confidence = result['confidence']
    if confidence >= 0.8:
        conf_class = "high-confidence"
        conf_emoji = "🟢"
    elif confidence >= 0.6:
        conf_class = "medium-confidence"
        conf_emoji = "🟡"
    else:
        conf_class = "low-confidence"
        conf_emoji = "🔴"
    
    # Main prediction box
    st.markdown(f"""
    <div class="prediction-box {conf_class}">
        <h2>{conf_emoji} Predicted Module: {result['module']}</h2>
        <h3>Confidence: {confidence:.1%} ({result['confidence_level']})</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if result.get('needs_review'):
        st.warning("⚠️ **Manual Review Recommended** - Confidence is below threshold")
    
    # Detailed reasoning
    if show_reasoning and result.get('reasoning'):
        with st.expander("🧠 Detailed Reasoning", expanded=True):
            st.markdown("**Analysis Steps:**")
            st.text(result['reasoning'])
            
            if result.get('key_indicators'):
                st.markdown("**Key Indicators:**")
                st.text(result['key_indicators'])
    
    # Similar tickets
    if show_similar and result.get('similar_tickets'):
        with st.expander("📋 Similar Historical Tickets", expanded=False):
            for i, ticket in enumerate(result['similar_tickets'][:5]):
                st.markdown(f"""
                **{i+1}. Ticket {ticket['id']}** (Module: {ticket['module']}, Similarity: {ticket['similarity']:.2%})
                
                {ticket['text']}
                
                ---
                """)
    
    # Raw response (for debugging)
    with st.expander("🔧 Raw Response (Debug)", expanded=False):
        st.json(result)


if __name__ == "__main__":
    main()
