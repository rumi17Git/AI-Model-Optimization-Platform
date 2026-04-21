"""Model upload page - UI only"""
import streamlit as st
from modules.model_loader import load_model_flexible, create_sample_model
from modules.model_analyzer import analyze_model
from modules.architecture_detector import get_supported_architectures

def uploader_page():
    """Model upload page"""
    
    st.title("Upload Model & Data")
    st.markdown("Upload your trained model for optimization analysis")
    
    st.markdown("---")
    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown("### Quick Start")
        st.info(
            "New to the platform? Click the button to load a sample model and explore the features!"
        )
        if st.button(" Use Sample Model", type="primary"):
            model = create_sample_model()
            st.session_state.uploaded_model = model
            st.session_state.model_format = "pytorch"

            metrics = analyze_model(model)
            st.session_state.original_metrics = metrics

            st.success(" Sample model loaded!")
            st.rerun()

    with col2:
        st.empty()

    # Model upload section
    with st.expander("Upload PyTorch Model", expanded=True):
        st.markdown("### Supported Formats")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.info("""
            **Supported Architectures:**
            - ResNet (18, 34, 50, 101, 152)
            - VGG (11, 13, 16, 19)
            - MobileNet V2
            - EfficientNet (B0-B7)
            - DenseNet (121, 161, 169, 201)
            """)
        
        with col2:
            st.empty()
       
        uploaded_file = st.file_uploader(
            "Choose a model file",
            type=['pt', 'pth'],
            help="Upload your trained PyTorch model"
        )
        
        if uploaded_file is not None:
            st.success(f" File uploaded: {uploaded_file.name}")
            
            # Save to temp and load
            import tempfile
            from pathlib import Path
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                with st.spinner("Loading model... This may take a moment."):
                    model, format_type = load_model_flexible(tmp_path)
                
                st.info(f"**Format detected:** {format_type}")
                
                if model:
                    st.session_state.uploaded_model = model
                    st.session_state.model_format = 'pytorch'
                    
                    # Analyze model
                    with st.spinner("Analyzing model performance..."):
                        metrics = analyze_model(model)
                        st.session_state.original_metrics = metrics
                    
                    # Save to database
                    try:
                        from modules.database import get_db_manager
                        db = get_db_manager()
                        model_id = db.add_model_upload(
                            user_id=st.session_state.user_id,
                            model_name=uploaded_file.name,
                            model_type=model.__class__.__name__,
                            metrics=metrics
                        )
                        st.session_state.current_model_id = model_id
                    except Exception as e:
                        st.warning(f"Could not save to database: {str(e)}")
                    
                    st.success(" Model loaded and analyzed successfully!")
                    
                    # Show basic info
                    st.markdown("### Model Information")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Architecture", model.__class__.__name__)
                    with col2:
                        st.metric("Size", f"{metrics['size']:.1f} MB")
                    with col3:
                        st.metric("Parameters", f"{metrics['total_params']:,}")
                else:
                    st.error(" Could not load model")
            
            except ValueError as e:
                st.error(f" {str(e)}")
                st.markdown("### Troubleshooting")
                st.info("""
                **If you see 'Unknown architecture' error:**
                
                1. **Best solution:** Save your model as a full model:
                   ```python
                   torch.save(model, 'model.pt')  # Not state_dict
                   ```
                
                2. **Alternative:** Use the Sample Model button below to test the app
                
                3. **Check architecture:** Ensure your model is one of the supported architectures listed above
                """)
            
            except Exception as e:
                st.error(f" Error loading model: {str(e)}")
                st.markdown("### Common Issues")
                st.warning("""
                **Possible causes:**
                - Corrupted file
                - Model saved with different PyTorch version
                - Custom model class not available
                
                **Solution:** Try the Sample Model button to verify the app works!
                """)
    
    
    # Display current model info
    if st.session_state.get('original_metrics'):
        st.markdown("---")
        st.markdown("### Current Model Metrics")
        
        metrics = st.session_state.original_metrics
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Size", f"{metrics['size']:.2f} MB")
        with col2:
            st.metric("Latency", f"{metrics['latency']:.2f} ms")
        with col3:
            st.metric("Parameters", f"{metrics['total_params']:,}")
        with col4:
            st.metric("Accuracy", f"{metrics['accuracy']:.1f}%")
        
        st.success(" Model is ready for optimization! Go to **Optimization Engine** to start.")
    
    # Dataset upload section
    with st.expander("Upload Dataset (Optional)"):
        st.markdown("### Training/Validation Data")
        st.info("Dataset upload is optional for this demo. You can optimize models without uploading data.")
        
        uploaded_data = st.file_uploader(
            "Choose dataset file",
            type=['pt', 'pth', 'pkl', 'csv'],
            help="Upload your dataset (optional)"
        )
        
        if uploaded_data is not None:
            st.success(f" Dataset uploaded: {uploaded_data.name}")
            st.session_state.uploaded_data = uploaded_data
