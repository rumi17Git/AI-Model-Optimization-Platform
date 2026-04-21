"""Dashboard page - UI only"""
import streamlit as st
from modules.analytics import get_model_metrics, get_comparison_data, generate_charts

def dashboard_page():
    """Dashboard page with performance analytics"""
    
    st.title("Performance Dashboard")
    st.markdown("Model optimization analytics and insights")
    
    # Check if model is loaded
    if not st.session_state.get('uploaded_model'):
        st.info("Upload a model to begin analyzing performance metrics. Navigate to **Upload Model/Data** to get started.")
        return
    
    # Original Metrics Section
    if st.session_state.get('original_metrics'):
        st.header("Baseline Performance Metrics")
        
        metrics = st.session_state.original_metrics
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Size", f"{metrics['size']:.2f} MB")
        with col2:
            st.metric("Inference Latency", f"{metrics['latency']:.2f} ms")
        with col3:
            st.metric("Memory Usage", f"{metrics['memory']:.2f} MB")
        with col4:
            st.metric("Model Accuracy", f"{metrics['accuracy']:.1f}%")
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Parameters", f"{metrics['total_params']:,}")
        with col2:
            st.metric("Trainable Parameters", f"{metrics['trainable_params']:,}")
        with col3:
            trainable_pct = (metrics['trainable_params'] / metrics['total_params'] * 100)
            st.metric("Trainability Ratio", f"{trainable_pct:.1f}%")
    
    # Optimization Results Section
    if st.session_state.get('optimized_metrics'):
        st.header("Optimization Performance")
        
        original = st.session_state.original_metrics
        optimized = st.session_state.optimized_metrics
        
        # Get comparison data from business logic
        comparison = get_comparison_data(original, optimized)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Optimized Size", f"{optimized['size']:.2f} MB", 
                     f"-{comparison['size_reduction']:.1f}%", delta_color="inverse")
        with col2:
            st.metric("Optimized Latency", f"{optimized['latency']:.2f} ms", 
                     f"-{comparison['latency_improvement']:.1f}%", delta_color="inverse")
        with col3:
            st.metric("Optimized Memory", f"{optimized['memory']:.2f} MB", 
                     f"-{comparison['memory_reduction']:.1f}%", delta_color="inverse")
        with col4:
            st.metric("Optimized Accuracy", f"{optimized['accuracy']:.1f}%", 
                     f"{comparison['accuracy_change']:+.2f}%")
        
        # Charts Section
        st.header("Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Resource Comparison")
            fig = generate_charts.resource_comparison_chart(original, optimized)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Quality Metrics Radar")
            fig = generate_charts.radar_chart(original, optimized)
            st.plotly_chart(fig, use_container_width=True)
        
        # Generate Report Button
        if st.button("Generate Comprehensive Report"):
            with st.spinner("Creating performance report..."):
                try:
                    from modules.report_generator import generate_optimization_report
                    from datetime import datetime
                    from pathlib import Path
                    import tempfile
                    
                    output_filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    temp_path = Path(tempfile.gettempdir()) / output_filename
                    
                    optimization_config = {
                        'techniques': comparison.get('techniques', [])
                    }
                    
                    generate_optimization_report(
                        original, optimized, optimization_config, str(temp_path)
                    )
                    
                    outputs_dir = Path.cwd() / "outputs"
                    outputs_dir.mkdir(parents=True, exist_ok=True)
                    final_path = outputs_dir / output_filename
                    
                    import shutil
                    shutil.copy(temp_path, final_path)
                    
                    st.success("Report generated successfully")
                    
                    with open(final_path, 'rb') as f:
                        st.download_button(
                            "Download Performance Report",
                            f,
                            file_name=output_filename,
                            mime="application/pdf"
                        )
                
                except Exception as e:
                    st.error(f"Report generation failed: {str(e)}")
    
    # Optimization History
    if st.session_state.get('optimization_history'):
        st.header("Optimization Timeline")
        
        import pandas as pd
        history_df = pd.DataFrame(st.session_state.optimization_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        # Timeline visualization
        if len(history_df) > 1:
            st.subheader("Size Optimization Trend")
            import plotly.express as px
            fig = px.area(
                history_df,
                x='timestamp',
                y='model_size',
                markers=True,
                labels={'model_size': 'Model Size (MB)', 'timestamp': ''},
            )
            st.plotly_chart(fig, use_container_width=True)
