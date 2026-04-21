"""Deployment page - UI only"""
import streamlit as st
import torch
from pathlib import Path

def is_quantized_model(model):
    """Detect if a model contains quantized modules"""
    return any(
        "quantized" in type(m).__module__
        for m in model.modules()
    )


def deployment_page():
    """Model deployment and export page"""
    
    st.title("Deployment Hub")
    st.markdown("Export and deploy your optimized model")
    
    if not st.session_state.get('optimized_model'):
        st.warning("Please optimize a model first!")
        st.info("Go to **Optimization Engine** to create an optimized model.")
        return
    
    st.success("Optimized model ready for export!")
    
    # Export options
    st.markdown("### Export & Deploy Your Optimized Model")
    
    tab1, tab2, tab3 = st.tabs(["Export Options", "Deployment Guide", "Benchmarking"])
    
    with tab1:
        st.markdown("#### Choose Export Format")

        col1, col2 = st.columns([2, 1])

        with col1:
            is_quantized = is_quantized_model(
                st.session_state.optimized_model
            )

            export_options = [
                "PyTorch (.pt)",
                "TorchScript (.pt)",
                "TensorFlow Lite (.tflite)",
                "Core ML (.mlmodel)",
            ]

            if not is_quantized:
                export_options.insert(0, "ONNX (.onnx)")

            export_format = st.selectbox(
                "Export Format",
                export_options,
                help="Select the format based on your deployment target",
            )

            if is_quantized:
                st.warning(
                    "ONNX export is disabled because this model is quantized.\n\n"
                    "PyTorch ONNX export does not currently support quantized "
                    "operators. Use TorchScript or PyTorch format instead."
                )

            if "ONNX" in export_format:
                st.info(
                    "**Best for:** Cross-platform deployment, cloud inference, edge devices"
                )
            elif "PyTorch" in export_format:
                st.info(
                    "**Best for:** PyTorch-based serving, research, further training"
                )
            elif "TorchScript" in export_format:
                st.info(
                    "**Best for:** Production deployment, mobile devices, C++ integration"
                )
            elif "TensorFlow Lite" in export_format:
                st.info(
                    "**Best for:** Mobile apps (Android/iOS), microcontrollers"
                )
            elif "Core ML" in export_format:
                st.info(
                    "**Best for:** iOS and macOS applications"
                )

            st.markdown("#### Export Options")

            include_metadata = st.checkbox(
                "Include model metadata", value=True
            )
            optimize_for_mobile = st.checkbox(
                "Optimize for mobile", value=False
            )

            if st.button(
                " Export Model",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Exporting model..."):
                    try:
                        import tempfile
                        import shutil

                        if "ONNX" in export_format:
                            filename = "optimized_model.onnx"
                        elif "TorchScript" in export_format:
                            filename = "optimized_model_script.pt"
                        elif "TensorFlow Lite" in export_format:
                            filename = "optimized_model.tflite"
                        elif "Core ML" in export_format:
                            filename = "optimized_model.mlmodel"
                        else:
                            filename = "optimized_model.pt"

                        temp_path = (
                            Path(tempfile.gettempdir()) / filename
                        )

                        if "ONNX" in export_format:
                            dummy_input = torch.randn(1, 3, 32, 32)
                            torch.onnx.export(
                                st.session_state.optimized_model,
                                dummy_input,
                                str(temp_path),
                                export_params=True,
                                opset_version=14,
                                do_constant_folding=True,
                                input_names=["input"],
                                output_names=["output"],
                                dynamic_axes={
                                    "input": {0: "batch_size"},
                                    "output": {0: "batch_size"},
                                },
                            )

                        elif "TorchScript" in export_format:
                            scripted = torch.jit.script(
                                st.session_state.optimized_model
                            )
                            scripted.save(str(temp_path))

                        else:
                            torch.save(
                                st.session_state.optimized_model.state_dict(),
                                temp_path,
                            )

                        outputs_dir = Path.cwd() / "outputs"
                        outputs_dir.mkdir(
                            parents=True, exist_ok=True
                        )

                        output_path = outputs_dir / filename
                        shutil.copy(temp_path, output_path)

                        st.success(
                            f"Model exported successfully as {filename}!"
                        )

                        file_size = (
                            output_path.stat().st_size
                            / (1024 * 1024)
                        )
                        st.info(
                            f"**File size:** {file_size:.2f} MB\n\n"
                            f"**Location:** `{output_path}`"
                        )

                        with open(output_path, "rb") as f:
                            st.download_button(
                                " Download Model",
                                f,
                                file_name=filename,
                                mime="application/octet-stream",
                                use_container_width=True,
                            )

                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
                        st.info(
                            "Tip: TorchScript or PyTorch (.pt) "
                            "are the most reliable options."
                        )
        
        with col2:
            st.markdown("#### Export Tips")
            st.markdown(
                "**Format Selection:**\n\n"
                "- ONNX: Universal\n"
                "- PyTorch: Development\n"
                "- TFLite: Mobile\n"
                "- Core ML: iOS/macOS\n\n"
                "**Optimization:**\n\n"
                "Enable mobile optimization for smaller file size and faster loading."
            )
    
    with tab2:
        st.markdown("#### Platform-Specific Deployment Guides")
        
        platform = st.selectbox(
            "Select Target Platform",
            [
                "Cloud (AWS, GCP, Azure)",
                "Mobile (iOS/Android)",
                "Edge Devices (Raspberry Pi, Jetson)",
                "Web Browser",
                "Docker Container"
            ]
        )
        
        if "Cloud" in platform:
            st.markdown("""
            ### Cloud Deployment Guide
            
            #### AWS SageMaker
            ```python
            import sagemaker
            from sagemaker.pytorch import PyTorchModel
            
            model = PyTorchModel(
                model_data='s3://bucket/model.tar.gz',
                role=role,
                framework_version='2.0.0',
                py_version='py39'
            )
            
            predictor = model.deploy(
                instance_type='ml.t2.medium',
                initial_instance_count=1
            )
            ```
            
            #### Google Cloud AI Platform
            ```bash
            gcloud ai-platform models create MODEL_NAME
            gcloud ai-platform versions create VERSION_NAME \\
                --model MODEL_NAME \\
                --runtime-version 2.8 \\
                --python-version 3.9 \\
                --framework pytorch
            ```
            
            #### Azure ML
            ```python
            from azureml.core import Model, Environment
            
            model = Model.register(
                workspace=ws,
                model_path='./model.onnx',
                model_name='optimized_model'
            )
            ```
            """)
        
        elif "Mobile" in platform:
            st.markdown("""
            ### Mobile Deployment Guide
            
            #### iOS (Core ML)
            ```swift
            import CoreML
            
            guard let model = try? OptimizedModel(configuration: MLModelConfiguration()) else {
                fatalError("Failed to load model")
            }
            
            let prediction = try? model.prediction(input: inputData)
            ```
            
            #### Android (TFLite)
            ```kotlin
            val tflite = Interpreter(loadModelFile())
            
            tflite.run(inputArray, outputArray)
            ```
            
            #### React Native
            ```javascript
            import { TensorFlowLite } from 'react-native-tflite';
            
            const model = await TensorFlowLite.loadModel({
                model: 'model.tflite'
            });
            ```
            """)
        
        elif "Edge" in platform:
            st.markdown("""
            ### Edge Device Deployment
            
            #### Raspberry Pi
            ```python
            import torch
            import onnxruntime as ort
            
            # Load ONNX model
            session = ort.InferenceSession('model.onnx')
            
            # Run inference
            outputs = session.run(None, {'input': input_data})
            ```
            
            #### NVIDIA Jetson
            ```python
            import tensorrt as trt
            
            # Convert to TensorRT
            with trt.Builder(TRT_LOGGER) as builder:
                network = builder.create_network()
                # Build engine...
            ```
            """)
        
        elif "Web" in platform:
            st.markdown("""
            ### Web Browser Deployment
            
            #### ONNX.js
            ```javascript
            import * as onnx from 'onnxjs';
            
            const session = new onnx.InferenceSession();
            await session.loadModel('./model.onnx');
            
            const outputMap = await session.run([inputTensor]);
            ```
            
            #### TensorFlow.js
            ```javascript
            import * as tf from '@tensorflow/tfjs';
            
            const model = await tf.loadLayersModel('model.json');
            const prediction = model.predict(inputTensor);
            ```
            """)
        
        elif "Docker" in platform:
            st.markdown("""
            ### Docker Container Deployment
            
            #### Dockerfile
            ```dockerfile
            FROM python:3.9-slim
            
            WORKDIR /app
            COPY requirements.txt .
            RUN pip install -r requirements.txt
            
            COPY model.onnx .
            COPY app.py .
            
            EXPOSE 8000
            CMD ["python", "app.py"]
            ```
            
            #### FastAPI Serving
            ```python
            from fastapi import FastAPI
            import onnxruntime as ort
            
            app = FastAPI()
            session = ort.InferenceSession('model.onnx')
            
            @app.post("/predict")
            async def predict(data: dict):
                output = session.run(None, {'input': data['input']})
                return {"prediction": output[0].tolist()}
            ```
            
            #### Build & Run
            ```bash
            docker build -t model-server .
            docker run -p 8000:8000 model-server
            ```
            """)
    
    with tab3:
        st.markdown("#### Benchmark Your Model")
        
        if st.session_state.optimized_metrics and st.session_state.original_metrics:
            import plotly.graph_objects as go
            
            # Performance comparison table
            st.markdown("### Performance Comparison")
            
            col1, col2, col3 = st.columns(3)
            
            original = st.session_state.original_metrics
            optimized = st.session_state.optimized_metrics
            
            with col1:
                st.markdown("**Original Model**")
                st.metric("Size", f"{original['size']:.2f} MB")
                st.metric("Latency", f"{original['latency']:.2f} ms")
                st.metric("Memory", f"{original['memory']:.2f} MB")
            
            with col2:
                st.markdown("**Optimized Model**")
                st.metric("Size", f"{optimized['size']:.2f} MB")
                st.metric("Latency", f"{optimized['latency']:.2f} ms")
                st.metric("Memory", f"{optimized['memory']:.2f} MB")
            
            with col3:
                st.markdown("**Improvements**")
                size_imp = ((original['size'] - optimized['size']) / original['size']) * 100
                latency_imp = ((original['latency'] - optimized['latency']) / original['latency']) * 100
                memory_imp = ((original['memory'] - optimized['memory']) / original['memory']) * 100
                
                st.metric("Size", f"-{size_imp:.1f}%")
                st.metric("Latency", f"-{latency_imp:.1f}%")
                st.metric("Memory", f"-{memory_imp:.1f}%")
            
            # Deployment readiness
            st.markdown("---")
            st.markdown("### Deployment Readiness")
            
            # Calculate readiness score
            score = 0
            checks = []
            
            if optimized['size'] < 50:
                score += 25
                checks.append(("", "Model size suitable for mobile/edge deployment"))
            else:
                checks.append(("", "Model size may be large for mobile/edge devices"))
            
            if optimized['latency'] < 50:
                score += 25
                checks.append(("", "Low latency suitable for real-time inference"))
            else:
                checks.append(("", "Latency may impact real-time applications"))
            
            if abs(optimized['accuracy'] - original['accuracy']) < 2:
                score += 25
                checks.append(("", "Accuracy well maintained"))
            else:
                checks.append(("", "Noticeable accuracy drop"))
            
            if optimized['memory'] < 100:
                score += 25
                checks.append(("", "Memory usage within acceptable range"))
            else:
                checks.append(("", "High memory usage"))
            
            # Display score
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.metric("Readiness Score", f"{score}%")
            
            with col2:
                st.progress(score / 100)
                
                for icon, check in checks:
                    st.markdown(f"{icon} {check}")

    # st.header("Export Options")
    
    # export_format = st.selectbox(
    #     "Export Format",
    #     ["PyTorch (.pt)", "TorchScript (.pt)", "ONNX (.onnx)"]
    # )
    
    # if st.button("Export Model", type="primary"):
    #     with st.spinner("Exporting model..."):
    #         try:
    #             outputs_dir = Path("outputs")
    #             outputs_dir.mkdir(exist_ok=True)
                
    #             if export_format == "PyTorch (.pt)":
    #                 filename = "optimized_model.pt"
    #                 filepath = outputs_dir / filename
    #                 torch.save(st.session_state.optimized_model.state_dict(), filepath)
    #                 st.success(f"Model exported to {filepath}")
                
    #             elif export_format == "TorchScript (.pt)":
    #                 filename = "optimized_model_script.pt"
    #                 filepath = outputs_dir / filename
    #                 scripted = torch.jit.script(st.session_state.optimized_model)
    #                 scripted.save(str(filepath))
    #                 st.success(f"Model exported to {filepath}")
                
    #             elif export_format == "ONNX (.onnx)":
    #                 st.info("ONNX export requires additional setup")
                
    #             with open(filepath, 'rb') as f:
    #                 st.download_button(
    #                     "Download Model",
    #                     f,
    #                     file_name=filename,
    #                     mime="application/octet-stream"
    #                 )
            
    #         except Exception as e:
    #             st.error(f"Export failed: {str(e)}")
