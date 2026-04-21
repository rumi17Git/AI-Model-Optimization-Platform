"""Carbon tracker page - Enhanced visualizations"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

def carbon_tracker_page():
    """Environmental impact tracking with comprehensive visualizations"""
    
    st.title("Carbon Footprint Tracker")
    st.markdown("Monitor the environmental impact of your model optimizations")
    
    if not st.session_state.get('optimized_metrics'):
        st.info("Optimize a model to see environmental impact analysis!")

        return
    
    # Calculate environmental metrics
    original = st.session_state.original_metrics
    optimized = st.session_state.optimized_metrics
    
    size_reduction_mb = original['size'] - optimized['size']
    latency_reduction_ms = original['latency'] - optimized['latency']
    memory_reduction_mb = original['memory'] - optimized['memory']
    
    # Energy calculations (simplified but realistic)
    # Based on industry estimates
    storage_energy_kwh = size_reduction_mb * 0.002  # kWh per MB per year
    compute_energy_kwh = (latency_reduction_ms / 1000) * 0.0001 * 1000000  # Per million inferences
    memory_energy_kwh = memory_reduction_mb * 0.001
    
    total_energy_kwh = storage_energy_kwh + compute_energy_kwh + memory_energy_kwh
    
    # Carbon calculations (average grid)
    carbon_factor = 0.475  # kg CO2 per kWh (US average)
    carbon_saved_kg = total_energy_kwh * carbon_factor
    carbon_saved_tons = carbon_saved_kg / 1000
    
    # Equivalencies for context
    trees_equivalent = carbon_saved_kg / 21  # kg CO2 absorbed per tree per year
    car_miles = carbon_saved_kg / 0.404  # kg CO2 per mile
    phone_charges = total_energy_kwh / 0.012  # kWh per phone charge
    
    # Header metrics
    st.header("Impact Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Energy Saved", f"{total_energy_kwh:.2f} kWh", 
                 help="Total energy saved from storage, compute, and memory reductions")
    with col2:
        st.metric("CO₂ Reduced", f"{carbon_saved_kg:.2f} kg",
                 help="Carbon dioxide emissions prevented")
    with col3:
        st.metric("Storage Saved", f"{size_reduction_mb:.1f} MB",
                 help="Less data to store and transfer")
    with col4:
        st.metric("Compute Saved", f"{latency_reduction_ms:.1f} ms",
                 help="Faster inference = less energy per request")
    
    # Visual equivalencies
    st.markdown("---")
    st.header("Real-World Equivalencies")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Trees planted", f"{trees_equivalent:.1f}", 
                 help="Annual CO₂ absorption equivalent")
    with col2:
        st.metric("Miles not driven", f"{car_miles:.0f}",
                 help="Car emissions equivalent")
    with col3:
        st.metric("Phone charges", f"{phone_charges:.0f}",
                 help="Energy saved equivalent")
    
    # Detailed breakdown
    st.markdown("---")
    st.header("Detailed Environmental Breakdown")
    
    # Energy breakdown chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Energy Savings by Source")
        
        fig = go.Figure(data=[go.Pie(
            labels=['Storage', 'Compute', 'Memory'],
            values=[storage_energy_kwh, compute_energy_kwh, memory_energy_kwh],
            hole=0.4,
            marker_colors=['#667eea', '#f5576c', '#4facfe']
        )])
        
        fig.update_layout(
            annotations=[dict(text=f'{total_energy_kwh:.2f}<br>kWh', 
                            x=0.5, y=0.5, font_size=20, showarrow=False)],
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Carbon Impact Over Time")
        
        # Simulate scaled impact
        scales = ['Per Hour', 'Per Day', 'Per Month', 'Per Year']
        multipliers = [100, 2400, 72000, 876000]  # Inferences at scale
        
        carbon_values = [carbon_saved_kg * m / 1000000 for m in multipliers]
        
        fig = go.Figure(data=[go.Bar(
            x=scales,
            y=carbon_values,
            marker_color=['#667eea', '#764ba2', '#f5576c', '#4facfe'],
            text=[f'{v:.2f} kg' for v in carbon_values],
            textposition='outside'
        )])
        
        fig.update_layout(
            yaxis_title="CO₂ Saved (kg)",
            height=350,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Cumulative impact
    if st.session_state.get('optimization_history'):
        st.markdown("---")
        st.header("Cumulative Environmental Impact")
        
        history = st.session_state.optimization_history
        
        # Calculate cumulative savings
        total_runs = len(history)
        cumulative_carbon = carbon_saved_kg * total_runs
        cumulative_energy = total_energy_kwh * total_runs
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Optimizations", total_runs)
        with col2:
            st.metric("Total Energy Saved", f"{cumulative_energy:.2f} kWh")
        with col3:
            st.metric("Total CO₂ Prevented", f"{cumulative_carbon:.2f} kg")
        
        st.success(f"You've made {total_runs} optimization{'s' if total_runs > 1 else ''} - "
                  f"equivalent to planting {trees_equivalent * total_runs:.0f} trees!")
    
    # Best practices
    st.markdown("---")
    st.header("Sustainability Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Optimization Strategies
        
        **Model Design:**
        - Start with efficient architectures
        - Use knowledge distillation
        - Apply quantization early
        - Profile before deploying
        
        **Deployment:**
        - Use green data centers
        - Optimize batch sizes
        - Cache common predictions
        - Scale resources dynamically
        """)
    
    with col2:
        st.markdown("""
        ### Impact Multipliers
        
        **At Scale:**
        - Mobile deployment: billions of devices
        - Edge computing: reduced cloud dependency
        - Efficient caching: avoid redundant compute
        - Model sharing: one optimization helps many
        
        **Long Term:**
        - Smaller models = longer device life
        - Less heat = less cooling needed
        - Faster inference = better UX
        - Sustainable AI practices
        """)
