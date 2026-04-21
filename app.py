"""Main application entry point"""
import streamlit as st
import sys
from pathlib import Path

# Add modules and pages to path
sys.path.insert(0, str(Path(__file__).parent))

# Import pages
from views.dashboard_page import dashboard_page
from views.uploader_page import uploader_page
from views.ai_consultant_page import ai_consultant_page
from views.optimization_page import optimization_page
from views.history_page import history_page
from views.deployment_page import deployment_page
from views.carbon_tracker_page import carbon_tracker_page

# Page configuration
st.set_page_config(
    page_title="AI Model Optimization Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = 1
if 'username' not in st.session_state:
    st.session_state.username = "User"
if 'uploaded_model' not in st.session_state:
    st.session_state.uploaded_model = None
if 'optimized_model' not in st.session_state:
    st.session_state.optimized_model = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'original_metrics' not in st.session_state:
    st.session_state.original_metrics = None
if 'optimized_metrics' not in st.session_state:
    st.session_state.optimized_metrics = None
if 'optimization_history' not in st.session_state:
    st.session_state.optimization_history = []
if 'current_model_id' not in st.session_state:
    st.session_state.current_model_id = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"

def main():
    """Main application entry point"""
    
    # Sidebar navigation
    st.sidebar.title("DSA7 - Model Optimizer")
    st.sidebar.markdown("---")
    
    # Navigation buttons
    pages = {
        "Dashboard": "Dashboard",
        "Upload Model": "Upload Model/Data",
        "AI Consultant": "AI Consultant",
        "Optimization Engine": "Optimization Engine",
        "History": "History",
        "Deployment": "Deployment Hub",
        "Carbon Tracker": "Carbon Tracker"
    }
    
    st.sidebar.markdown("### Navigation")
    
    for display_name, page_name in pages.items():
        # Determine if this is the current page
        is_current = st.session_state.current_page == page_name
        
        # Create button with appropriate styling
        if st.sidebar.button(
            display_name,
            use_container_width=True,
            type="primary" if is_current else "secondary",
            key=f"nav_{page_name}"
        ):
            st.session_state.current_page = page_name
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Route to pages based on current_page
    page = st.session_state.current_page
    
    if page == "Dashboard":
        dashboard_page()
    elif page == "Upload Model/Data":
        uploader_page()
    elif page == "AI Consultant":
        ai_consultant_page()
    elif page == "Optimization Engine":
        optimization_page()
    elif page == "History":
        history_page()
    elif page == "Deployment Hub":
        deployment_page()
    elif page == "Carbon Tracker":
        carbon_tracker_page()

if __name__ == "__main__":
    main()