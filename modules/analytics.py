"""Analytics and charts - business logic only"""
import plotly.graph_objects as go

def get_comparison_data(original, optimized):
    """Calculate comparison metrics"""
    return {
        'size_reduction': ((original['size'] - optimized['size']) / original['size']) * 100,
        'latency_improvement': ((original['latency'] - optimized['latency']) / original['latency']) * 100,
        'memory_reduction': ((original['memory'] - optimized['memory']) / original['memory']) * 100,
        'accuracy_change': optimized['accuracy'] - original['accuracy'],
    }

class generate_charts:
    """Chart generation utilities"""
    
    @staticmethod
    def resource_comparison_chart(original, optimized):
        """Generate resource comparison bar chart"""
        fig = go.Figure()
        
        categories = ['Size (MB)', 'Latency (ms)', 'Memory (MB)']
        original_values = [original['size'], original['latency'], original['memory']]
        optimized_values = [optimized['size'], optimized['latency'], optimized['memory']]
        
        fig.add_trace(go.Bar(
            name='Baseline',
            x=categories,
            y=original_values,
            text=[f'{v:.1f}' for v in original_values],
            textposition='outside',
        ))
        
        fig.add_trace(go.Bar(
            name='Optimized',
            x=categories,
            y=optimized_values,
            text=[f'{v:.1f}' for v in optimized_values],
            textposition='outside',
        ))
        
        fig.update_layout(barmode='group', height=400)
        return fig
    
    @staticmethod
    def radar_chart(original, optimized):
        """Generate quality metrics radar chart"""
        fig = go.Figure()
        
        metrics_labels = ['Accuracy', 'Speed', 'Efficiency', 'Compactness']
        
        original_scores = [
            original['accuracy'],
            max(0, (1 - min(original['latency'], 200)/200) * 100),
            max(0, (1 - min(original['memory'], 200)/200) * 100),
            max(0, (1 - min(original['size'], 200)/200) * 100)
        ]
        
        optimized_scores = [
            optimized['accuracy'],
            max(0, (1 - min(optimized['latency'], 200)/200) * 100),
            max(0, (1 - min(optimized['memory'], 200)/200) * 100),
            max(0, (1 - min(optimized['size'], 200)/200) * 100)
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=original_scores,
            theta=metrics_labels,
            fill='toself',
            name='Baseline',
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=optimized_scores,
            theta=metrics_labels,
            fill='toself',
            name='Optimized',
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=400
        )
        return fig

def get_model_metrics(model):
    """Get model metrics summary"""
    total_params = sum(p.numel() for p in model.parameters())
    return {
        'total_parameters': total_params,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
