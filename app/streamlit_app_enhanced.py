"""
Enhanced Streamlit web interface for SAP Module Classifier with Employee Management.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime
from loguru import logger

from src.config import Config
from src.embeddings import GeminiEmbedder
from src.vector_store import ChromaVectorStore
from src.retrieval import DenseRetriever, SparseRetriever, HybridRetriever
from src.rag import SAPModuleClassifier
from src.assignment import EmployeeManager, AssignmentEngine


# Page config
st.set_page_config(
    page_title="SAP Module Classifier & Assignment",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .employee-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
    .status-available {
        color: #28a745;
        font-weight: bold;
    }
    .status-on-leave {
        color: #dc3545;
        font-weight: bold;
    }
    .status-busy {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the classification and assignment system."""
    try:
        # Validate config
        Config.validate()
        
        # Initialize embedder
        embedder = GeminiEmbedder(
            api_key=Config.GOOGLE_API_KEY,
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Initialize vector store
        paths = Config.get_paths()
        vector_store = ChromaVectorStore(persist_directory=str(paths['vector_store']))
        vector_store.create_collection(name=Config.COLLECTION_NAME, reset=False)
        
        # Initialize retrievers
        bm25_index_path = Config.DATA_DIR / "embeddings" / "bm25_index.pkl"
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
        
        # Initialize employee manager and assignment engine
        assignment_dir = Config.DATA_DIR / "assignments"
        employee_manager = EmployeeManager(assignment_dir)
        assignment_engine = AssignmentEngine(employee_manager)
        
        return classifier, employee_manager, assignment_engine, True
    
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        return None, None, None, False


def render_sidebar():
    """Render sidebar navigation."""
    st.sidebar.title("🎯 Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        [
            "🎫 Classify & Assign Tickets",
            "👥 Employee Management",
            "📊 Assignment Dashboard",
            "⚙️ Settings"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**SAP Module Classifier**\n\n"
        "Automatically classify SAP tickets and assign them to employees with intelligent load balancing."
    )
    
    return page


def render_classification_page(classifier, assignment_engine):
    """Render ticket classification and assignment page."""
    st.markdown('<h1 class="main-header">🎫 SAP Ticket Classification & Assignment</h1>', unsafe_allow_html=True)
    
    # Input method selection
    input_method = st.radio(
        "Input Method",
        ["Single Ticket", "Batch Upload"],
        horizontal=True
    )
    
    if input_method == "Single Ticket":
        render_single_ticket_form(classifier, assignment_engine)
    else:
        render_batch_upload_form(classifier, assignment_engine)


def render_single_ticket_form(classifier, assignment_engine):
    """Render single ticket classification form."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Ticket Details")
        
        ticket_id = st.text_input("Ticket ID", placeholder="e.g., INC0012345")
        summary = st.text_input("Summary", placeholder="Brief description of the issue")
        description = st.text_area(
            "Description",
            placeholder="Detailed description of the issue...",
            height=150
        )
        priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
    
    with col2:
        st.subheader("Options")
        auto_assign = st.checkbox("Auto-assign to employee", value=True)
        show_reasoning = st.checkbox("Show classification reasoning", value=True)
        show_similar = st.checkbox("Show similar tickets", value=False)
    
    if st.button("🚀 Classify & Assign", type="primary", use_container_width=True):
        if not summary or not description:
            st.error("Please provide both summary and description")
            return
        
        with st.spinner("Classifying ticket..."):
            # Classify using predict method (same as original app)
            result = classifier.predict(
                summary=summary,
                description=description,
                incident_number=ticket_id or None,
                top_k=10
            )
            
            # Display results
            st.markdown("---")
            st.subheader("📋 Classification Results")
            
            # Confidence level
            confidence = result['confidence']
            if confidence >= 0.8:
                conf_class = "high-confidence"
                conf_label = "High"
            elif confidence >= 0.6:
                conf_class = "medium-confidence"
                conf_label = "Medium"
            else:
                conf_class = "low-confidence"
                conf_label = "Low"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Module", result['module'])
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
            with col3:
                st.metric("Confidence Level", conf_label)
            
            # Assignment
            if auto_assign:
                st.markdown("---")
                st.subheader("👤 Employee Assignment")
                
                assignment = assignment_engine.assign_ticket(
                    module=result['module'],
                    ticket_id=ticket_id or "TEMP_" + datetime.now().strftime("%Y%m%d%H%M%S"),
                    priority=priority
                )
                
                if assignment:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"✅ Assigned to: **{assignment['assigned_to']['name']}**")
                        if assignment['assigned_to']['email']:
                            st.info(f"📧 {assignment['assigned_to']['email']}")
                    
                    with col2:
                        st.metric(
                            "Daily Tickets",
                            f"{assignment['daily_count']}/{assignment['max_daily']}",
                            delta=f"{assignment['remaining']} remaining"
                        )
                else:
                    st.error("❌ No available employees for this module")
            
            # Reasoning
            if show_reasoning and 'reasoning' in result:
                with st.expander("🧠 Classification Reasoning"):
                    st.write(result['reasoning'])
            
            # Similar tickets
            if show_similar and 'similar_tickets' in result:
                with st.expander("📎 Similar Tickets"):
                    for i, ticket in enumerate(result['similar_tickets'][:3], 1):
                        st.markdown(f"**{i}. {ticket.get('Module', 'Unknown')}** (Similarity: {ticket.get('similarity', 0):.2%})")
                        st.text(ticket.get('text', '')[:200] + "...")
                        st.markdown("---")


def render_batch_upload_form(classifier, assignment_engine):
    """Render batch upload form."""
    st.subheader("📤 Batch Upload")
    
    st.info(
        "Upload a CSV file with columns: `Ticket_ID`, `Summary`, `Description`, `Priority`\n\n"
        "The system will classify all tickets and assign them to available employees."
    )
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            required_cols = ['Summary', 'Description']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return
            
            st.success(f"✅ Loaded {len(df)} tickets")
            st.dataframe(df.head())
            
            if st.button("🚀 Process Batch", type="primary"):
                process_batch(df, classifier, assignment_engine)
        
        except Exception as e:
            st.error(f"Error reading file: {e}")


def process_batch(df, classifier, assignment_engine):
    """Process batch of tickets."""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        status_text.text(f"Processing ticket {idx + 1}/{len(df)}...")
        
        # Classify using predict method (same as original app)
        result = classifier.predict(
            summary=row['Summary'],
            description=row['Description'],
            incident_number=row.get('Ticket_ID', f"BATCH_{idx}"),
            top_k=10
        )
        
        # Assign
        ticket_id = row.get('Ticket_ID', f"BATCH_{idx}")
        priority = row.get('Priority', 'Medium')
        
        assignment = assignment_engine.assign_ticket(
            module=result['module'],
            ticket_id=ticket_id,
            priority=priority
        )
        
        results.append({
            'Ticket_ID': ticket_id,
            'Summary': row['Summary'][:50] + "...",
            'Predicted_Module': result['module'],
            'Confidence': f"{result['confidence']:.1%}",
            'Assigned_To': assignment['assigned_to']['name'] if assignment else 'Unassigned',
            'Status': 'Assigned' if assignment else 'Pending'
        })
        
        progress_bar.progress((idx + 1) / len(df))
    
    status_text.text("✅ Batch processing complete!")
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Batch Results")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        "📥 Download Results",
        csv,
        "assignment_results.csv",
        "text/csv",
        key='download-csv'
    )


def render_employee_management_page(employee_manager):
    """Render employee management page."""
    st.markdown('<h1 class="main-header">👥 Employee Management</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📋 Employee List", "➕ Add Employee", "✏️ Edit Employee"])
    
    with tab1:
        render_employee_list(employee_manager)
    
    with tab2:
        render_add_employee_form(employee_manager)
    
    with tab3:
        render_edit_employee_form(employee_manager)


def render_employee_list(employee_manager):
    """Render employee list."""
    st.subheader("Current Employees")
    
    employees = employee_manager.get_all_employees()
    
    if not employees:
        st.info("No employees found. Add employees using the 'Add Employee' tab.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", len(employees))
    
    with col2:
        available = len([e for e in employees if e['status'] == 'available'])
        st.metric("Available", available)
    
    with col3:
        on_leave = len([e for e in employees if e['status'] == 'on_leave'])
        st.metric("On Leave", on_leave)
    
    with col4:
        total_assigned = sum([e.get('total_assigned', 0) for e in employees])
        st.metric("Total Assigned", total_assigned)
    
    st.markdown("---")
    
    # Employee cards
    for emp in employees:
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            
            with col1:
                st.markdown(f"### {emp['name']}")
                st.text(f"ID: {emp['id']}")
            
            with col2:
                st.markdown("**Modules:**")
                st.write(", ".join(emp['modules']))
            
            with col3:
                status = emp['status']
                status_class = f"status-{status.replace('_', '-')}"
                st.markdown(f"<p class='{status_class}'>{status.upper()}</p>", unsafe_allow_html=True)
                
                daily_count = employee_manager.get_daily_count(emp['id'])
                st.text(f"Today: {daily_count}/{emp['max_daily_tickets']}")
            
            with col4:
                st.text(f"Total: {emp.get('total_assigned', 0)}")
                if emp.get('email'):
                    st.text(f"📧 {emp['email']}")
            
            st.markdown("---")


def render_add_employee_form(employee_manager):
    """Render add employee form."""
    st.subheader("Add New Employee")
    
    with st.form("add_employee_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            employee_id = st.text_input("Employee ID*", placeholder="e.g., EMP001")
            name = st.text_input("Full Name*", placeholder="e.g., John Doe")
            email = st.text_input("Email", placeholder="john.doe@company.com")
        
        with col2:
            modules = st.multiselect(
                "SAP Modules*",
                Config.SAP_MODULES,
                help="Select all modules this employee can handle"
            )
            max_daily = st.number_input(
                "Max Daily Tickets",
                min_value=1,
                max_value=100,
                value=20,
                help="Maximum tickets per day"
            )
        
        submitted = st.form_submit_button("➕ Add Employee", type="primary")
        
        if submitted:
            if not employee_id or not name or not modules:
                st.error("Please fill in all required fields (*)")
            else:
                try:
                    employee_manager.add_employee(
                        employee_id=employee_id,
                        name=name,
                        modules=modules,
                        email=email if email else None,
                        max_daily_tickets=max_daily
                    )
                    st.success(f"✅ Successfully added {name}!")
                    st.rerun()
                except ValueError as e:
                    st.error(f"Error: {e}")


def render_edit_employee_form(employee_manager):
    """Render edit employee form."""
    st.subheader("Edit Employee")
    
    employees = employee_manager.get_all_employees()
    
    if not employees:
        st.info("No employees to edit.")
        return
    
    employee_options = {f"{emp['name']} ({emp['id']})": emp['id'] for emp in employees}
    selected = st.selectbox("Select Employee", list(employee_options.keys()))
    
    if selected:
        emp_id = employee_options[selected]
        employee = employee_manager.get_employee(emp_id)
        
        with st.form("edit_employee_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name", value=employee['name'])
                email = st.text_input("Email", value=employee.get('email', ''))
                status = st.selectbox(
                    "Status",
                    ["available", "on_leave", "busy"],
                    index=["available", "on_leave", "busy"].index(employee['status'])
                )
            
            with col2:
                modules = st.multiselect(
                    "SAP Modules",
                    Config.SAP_MODULES,
                    default=employee['modules']
                )
                max_daily = st.number_input(
                    "Max Daily Tickets",
                    min_value=1,
                    max_value=100,
                    value=employee['max_daily_tickets']
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                update_btn = st.form_submit_button("💾 Update Employee", type="primary")
            
            with col2:
                delete_btn = st.form_submit_button("🗑️ Delete Employee", type="secondary")
            
            if update_btn:
                try:
                    employee_manager.update_employee(
                        employee_id=emp_id,
                        name=name,
                        modules=modules,
                        email=email if email else None,
                        max_daily_tickets=max_daily,
                        status=status
                    )
                    st.success(f"✅ Successfully updated {name}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            
            if delete_btn:
                if st.session_state.get('confirm_delete') == emp_id:
                    employee_manager.delete_employee(emp_id)
                    st.success(f"✅ Deleted {employee['name']}")
                    st.session_state.confirm_delete = None
                    st.rerun()
                else:
                    st.session_state.confirm_delete = emp_id
                    st.warning("⚠️ Click Delete again to confirm")


def render_dashboard_page(employee_manager, assignment_engine):
    """Render assignment dashboard."""
    st.markdown('<h1 class="main-header">📊 Assignment Dashboard</h1>', unsafe_allow_html=True)
    
    # Get stats
    stats = assignment_engine.get_assignment_stats()
    summary = employee_manager.get_daily_summary()
    
    # Today's overview
    st.subheader(f"📅 Today's Overview ({stats['date']})")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Assigned", stats['total_tickets_assigned'])
    
    with col2:
        st.metric("Available Employees", stats['available_employees'])
    
    with col3:
        st.metric("At Capacity", stats['employees_at_capacity'])
    
    with col4:
        st.metric("On Leave", stats['employees_on_leave'])
    
    st.markdown("---")
    
    # Module-wise breakdown
    st.subheader("📦 Module-wise Capacity")
    
    module_data = []
    for module, data in stats['by_module'].items():
        utilization = (data['used_capacity'] / data['total_capacity'] * 100) if data['total_capacity'] > 0 else 0
        module_data.append({
            'Module': module,
            'Employees': data['total_employees'],
            'Available': data['available_employees'],
            'Total Capacity': data['total_capacity'],
            'Used': data['used_capacity'],
            'Remaining': data['total_capacity'] - data['used_capacity'],
            'Utilization': f"{utilization:.1f}%"
        })
    
    module_df = pd.DataFrame(module_data)
    st.dataframe(module_df, use_container_width=True)
    
    st.markdown("---")
    
    # Employee details
    st.subheader("👥 Employee Details")
    
    emp_data = []
    for emp in summary['employees']:
        emp_data.append({
            'Name': emp['name'],
            'ID': emp['id'],
            'Modules': ', '.join(emp['modules']),
            'Status': emp['status'],
            'Today': f"{emp['daily_assigned']}/{emp['max_daily']}",
            'Remaining': emp['remaining'],
            'Total': emp['total_assigned']
        })
    
    emp_df = pd.DataFrame(emp_data)
    st.dataframe(emp_df, use_container_width=True)
    
    # Reset button
    st.markdown("---")
    if st.button("🔄 Reset Daily Assignments (Admin)", type="secondary"):
        if st.session_state.get('confirm_reset'):
            employee_manager.reset_daily_assignments()
            st.success("✅ Daily assignments reset!")
            st.session_state.confirm_reset = False
            st.rerun()
        else:
            st.session_state.confirm_reset = True
            st.warning("⚠️ Click again to confirm reset")


def render_settings_page():
    """Render settings page."""
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    st.subheader("System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**LLM Provider:** {Config.LLM_PROVIDER}")
        st.info(f"**Model:** {Config.LLM_MODEL if Config.LLM_PROVIDER == 'gemini' else Config.GROQ_MODEL}")
        st.info(f"**Embedding Model:** {Config.EMBEDDING_MODEL}")
    
    with col2:
        st.info(f"**Vector Store:** ChromaDB")
        st.info(f"**Retrieval:** Hybrid (Dense + Sparse)")
        st.info(f"**SAP Modules:** {len(Config.SAP_MODULES)}")
    
    st.markdown("---")
    
    st.subheader("Available Modules")
    
    for module in Config.SAP_MODULES:
        st.markdown(f"- **{module}**")


def main():
    """Main application."""
    # Initialize system
    classifier, employee_manager, assignment_engine, success = initialize_system()
    
    if not success:
        st.error("❌ Failed to initialize system. Please check configuration and logs.")
        return
    
    # Render sidebar
    page = render_sidebar()
    
    # Render selected page
    if page == "🎫 Classify & Assign Tickets":
        render_classification_page(classifier, assignment_engine)
    elif page == "👥 Employee Management":
        render_employee_management_page(employee_manager)
    elif page == "📊 Assignment Dashboard":
        render_dashboard_page(employee_manager, assignment_engine)
    elif page == "⚙️ Settings":
        render_settings_page()


if __name__ == "__main__":
    main()
