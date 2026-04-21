"""History page - UI only"""
import streamlit as st
import pandas as pd

def history_page():
    """Optimization history page"""
    
    st.title("Optimization History")
    st.markdown("View past optimization runs and track improvements")
    
    if not st.session_state.get('optimization_history'):
        st.info("No optimization history yet. Run some optimizations to see results here!")
        return
    
    # Show history
    st.header("Past Optimizations")
    
    history_df = pd.DataFrame(st.session_state.optimization_history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    # Stats
    if len(history_df) > 0:
        st.header("Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Runs", len(history_df))
        with col2:
            avg_reduction = history_df['size_reduction'].str.rstrip('%').astype(float).mean()
            st.metric("Avg Size Reduction", f"{avg_reduction:.1f}%")
        with col3:
            st.metric("Models Optimized", history_df['model_name'].nunique())
