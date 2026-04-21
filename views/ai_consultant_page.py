"""AI consultant page - UI only"""
import streamlit as st
from modules.groq_client import GroqClient

def ai_consultant_page():
    """AI consultant page"""
    
    st.title("AI Optimization Consultant")
    st.markdown("Powered by Groq - Fast, free AI assistance for model optimization")
    st.info("Using: Llama 3.3 70B • Free Tier")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### AI Assistant Settings")
        
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        use_context = st.checkbox(
            "Share model context with AI",
            value=True,
            help="Let AI see your current model metrics"
        )
        st.session_state.use_context = use_context
        
        st.markdown("---")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        welcome_msg = {
            "role": "assistant",
            "content": """Hello! I'm your AI optimization expert.

I specialize in:
- **Model Optimization** - Quantization, pruning, distillation
- **Deployment** - Edge, mobile, cloud strategies  
- **Performance** - Latency, throughput, memory optimization
- **Frameworks** - PyTorch, TensorFlow, ONNX

Ask me anything about optimizing your ML models!"""
        }
        st.session_state.chat_history.append(welcome_msg)
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about model optimization..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get user context
        user_context = {}
        if st.session_state.get('use_context', True):
            if st.session_state.get('uploaded_model'):
                user_context['model_uploaded'] = True
                user_context['model_name'] = st.session_state.uploaded_model.__class__.__name__
                
                if st.session_state.get('original_metrics'):
                    user_context['original_metrics'] = st.session_state.original_metrics
                
                if st.session_state.get('optimized_metrics'):
                    user_context['optimized'] = True
                    user_context['optimized_metrics'] = st.session_state.optimized_metrics
                    orig = st.session_state.original_metrics
                    opt = st.session_state.optimized_metrics
                    user_context['size_reduction'] = ((orig['size'] - opt['size']) / orig['size'] * 100)
        
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                groq_client = GroqClient()
                result = groq_client.chat(
                    prompt,
                    st.session_state.chat_history[:-1],
                    user_context
                )
                
                response = result['response']
                commands = result.get('commands')
                
                st.markdown(response)
                
                # Execute optimization commands if present
                if commands and st.session_state.get('uploaded_model') and not st.session_state.get('optimized_model'):
                    st.info("🤖 AI suggested optimizations detected. Applying now...")
                    
                    try:
                        from modules.optimization_engine import apply_quantization, apply_pruning
                        from modules.model_analyzer import analyze_model
                        
                        optimized_model = st.session_state.uploaded_model
                        applied_techniques = []
                        
                        # Apply quantization
                        if commands.get('quantization'):
                            optimized_model = apply_quantization(optimized_model)
                            applied_techniques.append("Quantization")
                            st.success("✅ Quantization applied")
                        
                        # Apply pruning
                        if commands.get('pruning'):
                            prune_amount = commands['pruning']
                            optimized_model = apply_pruning(optimized_model, amount=prune_amount)
                            applied_techniques.append(f"Pruning ({prune_amount*100:.0f}%)")
                            st.success(f"✅ Pruning applied ({prune_amount*100:.0f}%)")
                        
                        # Analyze optimized model
                        optimized_metrics = analyze_model(optimized_model)
                        
                        # Save results
                        st.session_state.optimized_model = optimized_model
                        st.session_state.optimized_metrics = optimized_metrics
                        
                        # Show results
                        original = st.session_state.original_metrics
                        size_reduction = ((original['size'] - optimized_metrics['size']) / original['size']) * 100
                        
                        st.success(f"🎉 Optimization complete! Size reduced by {size_reduction:.1f}%")
                        
                        # Add follow-up message
                        auto_response = (
                            f"\n\n✨ I've automatically applied {' + '.join(applied_techniques)} to your model! "
                            f"The size went from {original['size']:.1f} MB to {optimized_metrics['size']:.1f} MB "
                            f"({size_reduction:.1f}% reduction). Check the Dashboard to see detailed results!"
                        )
                        
                        st.markdown(auto_response)
                        response += auto_response
                        
                    except Exception as e:
                        st.error(f"❌ Failed to apply optimizations: {str(e)}")
                        error_msg = f"\n\nI tried to apply the optimizations but encountered an error: {str(e)}"
                        st.markdown(error_msg)
                        response += error_msg

        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Show current context
    if st.session_state.get('use_context', True) and st.session_state.get('original_metrics'):
        with st.expander("Current Model Context (shared with AI)"):
            if st.session_state.get('uploaded_model'):
                st.write(f"**Model:** {st.session_state.uploaded_model.__class__.__name__}")
            
            if st.session_state.get('original_metrics'):
                metrics = st.session_state.original_metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Size", f"{metrics['size']:.1f} MB")
                with col2:
                    st.metric("Latency", f"{metrics['latency']:.1f} ms")
                with col3:
                    st.metric("Params", f"{metrics['total_params']:,}")
            
            if st.session_state.get('optimized_metrics'):
                st.success("Model has been optimized")
                opt = st.session_state.optimized_metrics
                orig = st.session_state.original_metrics
                reduction = ((orig['size'] - opt['size']) / orig['size'] * 100)
                st.write(f"**Size Reduction:** {reduction:.1f}%")
