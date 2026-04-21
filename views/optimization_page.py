"""Optimization page - UI with improved layout, tooltips, and KD progress tracking"""
import streamlit as st
from modules.optimization_engine import apply_quantization, apply_pruning, apply_knowledge_distillation
from modules.model_analyzer import analyze_model
import torch
import copy

def optimization_page():
    """Optimization engine page with enhanced UI"""
    
    st.title("Optimization Engine")
    st.markdown("Apply advanced optimization techniques to reduce model size while maintaining accuracy")
    
    # Check if model is uploaded
    if not st.session_state.get('uploaded_model'):
        st.warning("Please upload a model first!")
        st.info("Navigate to **Upload Model/Data** to get started.")
        
        # Show what optimization does
        st.markdown("---")
        st.markdown("### What You Can Do Here")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Quantization**
            - Convert FP32 → INT8
            - ~75% size reduction
            - <1% accuracy loss
            - Faster inference
            """)
        
        with col2:
            st.markdown("""
            **Pruning**
            - Remove redundant weights
            - 10-90% configurable
            - Maintains accuracy
            - Reduces parameters
            """)
        
        with col3:
            st.markdown("""
            **Knowledge Distillation**
            - Teacher → Student learning
            - Maintains accuracy
            - Fast training (5 epochs)
            - Uses synthetic data
            """)
        
        return
    
    st.success(f"Model loaded: {st.session_state.uploaded_model.__class__.__name__}")
    
    # Show current model stats
    if st.session_state.get('original_metrics'):
        metrics = st.session_state.original_metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Size", f"{metrics['size']:.1f} MB", help="Current model size")
        with col2:
            st.metric("Latency", f"{metrics['latency']:.1f} ms", help="Inference time")
        with col3:
            st.metric("Parameters", f"{metrics['total_params']:,}", help="Total parameters")
        with col4:
            st.metric("Accuracy", f"{metrics['accuracy']:.1f}%", help="Estimated accuracy")
    
    st.markdown("---")
    
    # Technique selection with better layout
    st.header("Select Optimization Techniques")
    st.markdown("Choose one or more techniques to apply. Combining techniques yields better results!")
    
    # Create columns for techniques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Quantization")
        use_quantization = st.checkbox(
            "Enable Quantization",
            value=True,
            help="Convert model from 32-bit floats to 8-bit integers. Recommended for all models."
        )
        
        if use_quantization:
            st.info("""
            **What it does:**
            - Converts FP32 → INT8
            - Reduces precision
            - 4x smaller weights
            
            **Benefits:**
            - ~75% size reduction
            - Faster on CPUs
            - Lower memory usage
            
            **Trade-offs:**
            - <1% accuracy loss
            - Quantization artifacts
            """)
            
            quant_backend = st.selectbox(
                "Backend",
                ["fbgemm", "qnnpack"],
                help="fbgemm: x86 CPUs (Intel/AMD)\nqnnpack: ARM CPUs (mobile)"
            )
            
            st.caption("Tip: Use fbgemm for servers, qnnpack for mobile")
    
    with col2:
        st.markdown("### Pruning")
        use_pruning = st.checkbox(
            "Enable Pruning",
            value=False,
            help="Remove unimportant weights. Great when combined with quantization."
        )
        
        if use_pruning:
            st.info("""
            **What it does:**
            - Removes low-magnitude weights
            - Sets weights to zero
            - Reduces parameters
            
            **Benefits:**
            - Configurable reduction
            - Maintains structure
            - Compound with quantization
            
            **Trade-offs:**
            - 1-3% accuracy loss
            - Needs fine-tuning
            """)
            
            prune_amount = st.slider(
                "Pruning Amount (%)",
                min_value=10,
                max_value=90,
                value=50,
                step=10,
                help="Percentage of weights to remove. Start with 30-50% for best results."
            ) / 100
            
            st.caption(f"Removing {prune_amount*100:.0f}% of weights")
            
            # Show expected impact
            if prune_amount <= 0.3:
                st.success("Conservative: Minimal accuracy impact")
            elif prune_amount <= 0.5:
                st.warning("Moderate: Some accuracy loss expected")
            else:
                st.error("Aggressive: Significant accuracy loss possible")
    
    with col3:
        st.markdown("### Knowledge Distillation")
        use_distillation = st.checkbox(
            "Enable Knowledge Distillation",
            value=False,
            help="Train a student model to mimic the teacher. Uses synthetic data for demo."
        )
        
        if use_distillation:
            st.info("""
            **What it does:**
            - Trains student from teacher
            - Uses soft targets (KL divergence)
            - Learns patterns, not just labels
            
            **Benefits:**
            - Best knowledge transfer
            - Maintains accuracy
            - Creates efficient models
            
            **Settings:**
            - Uses synthetic data for demo
            - Fast training (configurable)
            - Real data gives better results
            """)
            
            temperature = st.slider(
                "Temperature",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="Higher = softer probability distributions. 3-5 is typical."
            )
            
            epochs = st.slider(
                "Training Epochs",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="More epochs = better learning but longer time"
            )
            
            alpha = st.slider(
                "Distillation Weight (α)",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Balance between soft targets (teacher) and hard labels. 0.7 = 70% teacher influence"
            )
            
            st.caption(f"T={temperature}, α={alpha}, {epochs} epochs")
            # st.caption("Uses synthetic data (no real dataset needed)")
    
    # Configuration summary
    st.markdown("---")
    st.subheader("Configuration Summary")
    
    techniques = []
    if use_quantization:
        techniques.append(f"Quantization ({quant_backend})")
    if use_pruning:
        techniques.append(f"Pruning ({prune_amount*100:.0f}%)")
    if use_distillation:
        techniques.append(f"Knowledge Distillation (T={temperature}, α={alpha})")
    
    if techniques:
        st.success(f"**Selected techniques:** {' + '.join(techniques)}")
        
        # Show expected reduction
        expected_reduction = 0
        if use_quantization:
            expected_reduction += 75
        if use_pruning:
            expected_reduction += prune_amount * 15
        
        expected_reduction = min(expected_reduction, 90)  # Cap at 90%
        
        if expected_reduction > 0:
            st.info(f"**Expected size reduction:** ~{expected_reduction:.0f}%")
        
        if use_distillation:
            st.info("Knowledge Distillation improves accuracy retention and knowledge transfer")
    else:
        st.warning("No techniques selected. Please select at least one technique above.")
    
    # Run optimization button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        run_button = st.button(
            "Run Optimization",
            type="primary",
            disabled=not techniques,
            use_container_width=True,
            help="Apply selected optimization techniques to your model"
        )
    
    if run_button:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            optimized_model = st.session_state.uploaded_model
            step = 0
            total_steps = sum([use_quantization, use_pruning, use_distillation])
            
            # Apply techniques
            if use_quantization:
                step += 1
                status_text.text(f"Applying quantization... ({step}/{total_steps})")
                progress_bar.progress(step / total_steps)
                
                optimized_model = apply_quantization(
                    optimized_model,
                    backend=quant_backend
                )
                st.success("Quantization applied")
            
            if use_pruning:
                step += 1
                status_text.text(f"Applying pruning... ({step}/{total_steps})")
                progress_bar.progress(step / total_steps)
                
                optimized_model = apply_pruning(optimized_model, amount=prune_amount)
                st.success(f"Pruning applied ({prune_amount*100:.0f}%)")
            
            if use_distillation:
                step += 1
                status_text.text(f"Applying knowledge distillation... ({step}/{total_steps})")
                progress_bar.progress(step / total_steps)
                
                # Show KD-specific progress tracking
                with st.expander("Knowledge Distillation Training Progress", expanded=True):
                    # Force CPU to avoid CUDA compatibility issues
                    device = 'cpu'
                    
                    # Check CUDA availability and warn if forcing CPU
                    if torch.cuda.is_available():
                        st.warning("CUDA detected but using CPU due to compatibility. Training may be slower.")
                    
                    kd_status = st.empty()
                    kd_status.info(f"Initializing training on **{device.upper()}**...")
                    
                    # Progress metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        epoch_metric = st.empty()
                    with col_b:
                        loss_metric = st.empty()
                    with col_c:
                        acc_metric = st.empty()
                    
                    # Training progress chart placeholder
                    chart_placeholder = st.empty()
                    
                    try:
                        # Create a copy of the model as student (ensure CPU)
                        student_model = copy.deepcopy(optimized_model).cpu()
                        teacher_model = copy.deepcopy(st.session_state.uploaded_model).cpu()
                        
                        # Import KD engine for custom training loop with callbacks
                        from modules.optimization_engine import KnowledgeDistillationEngine
                        
                        # Initialize KD engine on CPU
                        kd_engine = KnowledgeDistillationEngine(
                            teacher_model=teacher_model,
                            student_model=student_model,
                            device='cpu'  # Force CPU
                        )
                        
                        # Create synthetic data
                        try:
                            first_layer = next(teacher_model.children())
                            if isinstance(first_layer, torch.nn.Conv2d):
                                input_shape = (first_layer.in_channels, 32, 32)
                            elif isinstance(first_layer, torch.nn.Linear):
                                input_shape = (first_layer.in_features,)
                            else:
                                input_shape = (3, 224, 224)
                        except:
                            input_shape = (3, 224, 224)
                        
                        kd_status.info(f"Creating synthetic training data (shape: {input_shape})...")
                        train_loader, val_loader = kd_engine.create_synthetic_data(
                            input_shape=input_shape,
                            num_samples=500
                        )
                        
                        kd_status.success("Training data ready! Starting training...")
                        
                        # Manual training loop with real-time updates
                        import torch.optim as optim
                        optimizer = optim.Adam(kd_engine.student.parameters(), lr=0.001)
                        
                        train_losses = []
                        train_accs = []
                        
                        for epoch in range(epochs):
                            kd_status.info(f"Training epoch {epoch+1}/{epochs}... (this may take a moment)")
                            
                            # Train one epoch
                            train_loss, train_acc = kd_engine.train_epoch(
                                train_loader=train_loader,
                                optimizer=optimizer,
                                temperature=temperature,
                                alpha=alpha
                            )
                            
                            train_losses.append(train_loss)
                            train_accs.append(train_acc)
                            
                            # Update metrics
                            epoch_metric.metric("Epoch", f"{epoch+1}/{epochs}")
                            loss_metric.metric("Loss", f"{train_loss:.4f}")
                            acc_metric.metric("Accuracy", f"{train_acc:.1f}%")
                            
                            # Update chart
                            import pandas as pd
                            import plotly.graph_objects as go
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=train_losses,
                                mode='lines+markers',
                                name='Loss',
                                line=dict(color='#ef4444', width=2),
                                marker=dict(size=8)
                            ))
                            fig.add_trace(go.Scatter(
                                y=train_accs,
                                mode='lines+markers',
                                name='Accuracy (%)',
                                yaxis='y2',
                                line=dict(color='#22c55e', width=2),
                                marker=dict(size=8)
                            ))
                            
                            fig.update_layout(
                                height=300,
                                margin=dict(l=0, r=0, t=20, b=0),
                                xaxis_title="Epoch",
                                yaxis_title="Loss",
                                yaxis2=dict(
                                    title="Accuracy (%)",
                                    overlaying='y',
                                    side='right'
                                ),
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            
                            chart_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        # Get the trained student model
                        optimized_model = kd_engine.student
                        
                        kd_status.success(f"Knowledge distillation complete! Final accuracy: {train_accs[-1]:.1f}%")
                        st.success(f"Knowledge distillation applied (T={temperature}, α={alpha}, {epochs} epochs)")
                        
                    except Exception as e:
                        kd_status.error(f"KD training failed: {str(e)}")
                        st.warning("Knowledge distillation failed, continuing with model before KD")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())
            
            # Analyze optimized model
            status_text.text("Analyzing optimized model...")
            progress_bar.progress(1.0)
            
            optimized_metrics = analyze_model(optimized_model)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Save results
            st.session_state.optimized_model = optimized_model
            st.session_state.optimized_metrics = optimized_metrics
            
            # Add to history
            from datetime import datetime
            history_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_name': st.session_state.uploaded_model.__class__.__name__,
                'techniques': ', '.join(techniques),
                'original_size': st.session_state.original_metrics['size'],
                'model_size': optimized_metrics['size'],
                'size_reduction': f"{((st.session_state.original_metrics['size'] - optimized_metrics['size']) / st.session_state.original_metrics['size'] * 100):.1f}%",
                'accuracy': f"{optimized_metrics['accuracy']:.2f}%",
                'latency': f"{optimized_metrics['latency']:.2f}ms",
            }
            
            if 'optimization_history' not in st.session_state:
                st.session_state.optimization_history = []
            st.session_state.optimization_history.append(history_entry)
            
            # Save to database
            try:
                from modules.database import get_db_manager
                db = get_db_manager()
                
                techniques_dict = {
                    'quantization': use_quantization,
                    'pruning': use_pruning,
                    'distillation': use_distillation
                }
                
                config = {
                    'quant_backend': quant_backend if use_quantization else None,
                    'prune_amount': prune_amount if use_pruning else None,
                    'temperature': temperature if use_distillation else None,
                    'kd_epochs': epochs if use_distillation else None,
                    'kd_alpha': alpha if use_distillation else None,
                }
                
                db.add_optimization_run(
                    user_id=st.session_state.user_id,
                    model_id=st.session_state.get('current_model_id'),
                    techniques=techniques_dict,
                    config=config,
                    original_metrics=st.session_state.original_metrics,
                    optimized_metrics=optimized_metrics
                )
            except Exception as e:
                st.warning(f"Could not save to database: {str(e)}")
            
            st.success("Optimization completed successfully!")
            st.balloons()
            
        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
            progress_bar.empty()
            status_text.empty()
    
    # Show results with GREEN delta for improvements
    if st.session_state.get('optimized_metrics'):
        st.markdown("---")
        st.header("Optimization Results")
        
        original = st.session_state.original_metrics
        optimized = st.session_state.optimized_metrics
        
        col1, col2, col3, col4 = st.columns(4)
        
        size_reduction = ((original['size'] - optimized['size']) / original['size']) * 100
        latency_improvement = ((original['latency'] - optimized['latency']) / original['latency']) * 100
        memory_reduction = ((original['memory'] - optimized['memory']) / original['memory']) * 100
        accuracy_change = optimized['accuracy'] - original['accuracy']
        
        with col1:
            st.metric(
                "Model Size",
                f"{optimized['size']:.2f} MB",
                f"{size_reduction:.1f}%",
                delta_color="normal",  # Green for reduction
                help="Lower is better"
            )
        
        with col2:
            st.metric(
                "Latency",
                f"{optimized['latency']:.2f} ms",
                f"{latency_improvement:.1f}%",
                delta_color="normal",  # Green for reduction
                help="Lower is better"
            )
        
        with col3:
            st.metric(
                "Memory",
                f"{optimized['memory']:.2f} MB",
                f"{memory_reduction:.1f}%",
                delta_color="normal",  # Green for reduction
                help="Lower is better"
            )
        
        with col4:
            st.metric(
                "Accuracy",
                f"{optimized['accuracy']:.1f}%",
                f"{accuracy_change:+.2f}%",
                delta_color="normal",  # Normal color for accuracy
                help="Higher is better"
            )
        
        # Summary message
        if size_reduction >= 80:
            st.success(f"Excellent! Achieved {size_reduction:.1f}% size reduction!")
        elif size_reduction >= 60:
            st.success(f"Great! Achieved {size_reduction:.1f}% size reduction!")
        elif size_reduction >= 40:
            st.info(f"Good! Achieved {size_reduction:.1f}% size reduction!")
        else:
            st.warning(f"Achieved {size_reduction:.1f}% size reduction. Try adding more techniques!")
        
        # Next steps
        st.markdown("---")
        st.markdown("### Next Steps")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("** View Dashboard**")
            st.caption("See detailed performance comparison and charts")
        
        with col2:
            st.markdown("** Deploy Model**")
            st.caption("Export optimized model for production")
        
        with col3:
            st.markdown("** Check Carbon Impact**")
            st.caption("View environmental benefits")