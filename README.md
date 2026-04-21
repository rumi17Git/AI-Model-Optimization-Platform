# AI Model Optimization Platform

A comprehensive Streamlit application for optimizing machine learning models through quantization, pruning, and knowledge distillation techniques.

## Features

### Core Features
- **Model Upload & Analysis**: Support for PyTorch (.pt, .pth), ONNX (.onnx), and other formats
- **Optimization Engine**: 
  - Quantization (FP32 → INT8)
  - Magnitude-based Pruning
  - Knowledge Distillation
- **Performance Dashboard**: Real-time metrics visualization and comparison
- **AI Consultant**: Interactive chat for optimization advice and best practices
- **Deployment Hub**: Export to multiple formats (ONNX, PyTorch, TorchScript)
- **Carbon Tracker**: Monitor environmental impact and energy consumption
- **PDF Reports**: Generate comprehensive optimization reports

### Metrics Tracked
- Model Size (MB)
- Inference Latency (ms)
- Memory Usage (MB)
- Accuracy (%)
- Total Parameters
- Carbon Footprint (kg CO₂)
- Energy Consumption (kWh)

## System Architecture

```
Frontend (Streamlit)              Backend (FastAPI)
├── Login/Credentials             ├── Optimization Engine
├── Dashboard/History             │   ├── Quantization
├── Model/Data Uploader           │   ├── Pruning
├── AI Consultant Chat            │   └── Knowledge Distillation
├── Optimization Config           ├── Benchmarking Engine
├── Deployment Hub                │   ├── ONNX Runtime
└── Carbon Tracker                │   └── TensorRT
                                  ├── Sustainability Tracker
                                  │   └── Carbon/Energy API
                                  └── SQL Database
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone or extract the application:**
```bash
cd ai_optimization_platform
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. **Start the Streamlit server:**
```bash
streamlit run app.py
```

2. **Access the application:**
Open your browser and navigate to `http://localhost:8501`

3. **Login:**
Use the demo credentials:
- Username: `demo`
- Password: `demo123`

### Workflow

#### 1. Upload Model & Data
- Navigate to "Upload Model/Data"
- Upload your trained model (PyTorch, ONNX, etc.)
- Optionally upload validation dataset
- Or use the sample model for testing

#### 2. Analyze Model
- View original model metrics on the dashboard
- Check model size, latency, accuracy, and parameter count

#### 3. Configure Optimization
- Go to "Optimization Engine"
- Select optimization techniques:
  - **Quantization**: Reduce precision (FP32 → INT8)
  - **Pruning**: Remove less important weights
  - **Distillation**: Create smaller student model
- Configure parameters (pruning amount, temperature, etc.)

#### 4. Run Optimization
- Click "Start Optimization"
- Monitor progress and wait for completion
- View optimized model metrics

#### 5. Review Results
- Dashboard shows before/after comparison
- View performance improvements:
  - Size reduction
  - Latency improvement
  - Accuracy retention
- Check carbon savings in Sustainability Tracker

#### 6. Generate Report
- Click "Generate PDF Report" on dashboard
- Download comprehensive optimization report
- Report includes metrics, findings, and recommendations

#### 7. Export Model
- Navigate to "Deployment Hub"
- Choose export format (ONNX recommended)
- Download optimized model
- Follow deployment guides for your platform

### AI Consultant

Use the AI Consultant for:
- Choosing the right optimization techniques
- Understanding trade-offs
- Troubleshooting issues
- Platform-specific deployment advice

Example questions:
- "What optimization should I use for mobile deployment?"
- "How do I minimize accuracy loss?"
- "Tell me about quantization"
- "Best practices for edge devices?"

## Features in Detail

### Optimization Engine

**Quantization**
- Reduces model precision from FP32 to INT8
- ~4x size reduction
- Minimal accuracy loss (<1%)
- Faster inference on compatible hardware

**Pruning**
- Removes less important weights
- Configurable pruning amount (10-90%)
- Reduces both size and computation
- May require fine-tuning

**Knowledge Distillation**
- Trains smaller model to mimic larger one
- Achieves high compression ratios
- Maintains performance better than other methods
- Requires training time

### Dashboard Features

- Real-time performance metrics
- Before/after comparison visualizations
- Optimization history tracking
- PDF report generation
- Interactive charts and graphs

### Sustainability Tracker

- Carbon footprint estimation
- Energy consumption monitoring
- Regional carbon intensity support
- Environmental equivalents (trees, car miles, etc.)
- Projected savings over deployment period

### Deployment Hub

- Multiple export format support
- Platform-specific deployment guides
- Deployment readiness assessment
- Code examples for various platforms
- Benchmarking tools

## File Structure

```
ai_optimization_platform/
├── app.py                          # Main application entry
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── modules/
    ├── auth.py                     # Authentication
    ├── dashboard.py                # Main dashboard
    ├── uploader.py                 # Model/data upload
    ├── optimization.py             # Optimization engine
    ├── ai_consultant.py            # AI chat consultant
    ├── deployment.py               # Deployment hub
    ├── carbon_tracker.py           # Sustainability tracker
    └── report_generator.py         # PDF report generation
```

## Configuration

### User Management
Edit `modules/auth.py` to add users:
```python
USERS = {
    "username": hashlib.sha256("password".encode()).hexdigest()
}
```

### Optimization Parameters
Default settings can be modified in `modules/optimization.py`:
- Pruning amount: 0.3 (30%)
- Quantization backend: 'fbgemm'
- Distillation temperature: 3.0

### Carbon Intensity
Regional carbon intensity values in `modules/carbon_tracker.py`:
- Global Average: 0.50 kg CO₂/kWh
- Europe: 0.30 kg CO₂/kWh
- USA: 0.40 kg CO₂/kWh
- China: 0.60 kg CO₂/kWh
- Renewable: 0.05 kg CO₂/kWh

## API Reference

### Session State Variables

- `authenticated`: Boolean for login status
- `username`: Current user's username
- `uploaded_model`: PyTorch/ONNX model object
- `uploaded_data`: Dataset reference
- `original_metrics`: Dict of original model metrics
- `optimized_metrics`: Dict of optimized model metrics
- `optimization_history`: List of optimization runs

### Key Functions

**Model Analysis:**
```python
analyze_model(model, sample_input) -> dict
```

**Optimization:**
```python
prune_model(model, amount=0.3) -> model
quantize_model(model, backend='fbgemm') -> model
apply_knowledge_distillation(student, teacher) -> model
```

**Export:**
```python
export_to_onnx(model, filepath, input_shape) -> filepath
```

**Reporting:**
```python
generate_optimization_report(original, optimized, config, output) -> filepath
```

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
pip install --upgrade -r requirements.txt
```

**Model Loading Failures:**
- Ensure model file is not corrupted
- Check PyTorch version compatibility
- Try using the sample model first

**Optimization Errors:**
- Some models may not support all optimizations
- Try individual techniques separately
- Check model architecture compatibility

**PDF Generation Fails:**
```bash
pip install --upgrade reportlab
```

## Performance Tips

1. **For Maximum Speed:**
   - Use quantization first
   - Deploy on GPU when possible
   - Enable batch inference

2. **For Maximum Compression:**
   - Combine pruning + quantization
   - Consider knowledge distillation
   - Use aggressive pruning (50-70%)

3. **For Accuracy Retention:**
   - Start with light quantization
   - Use conservative pruning (10-30%)
   - Fine-tune after optimization

## Future Enhancements

- [ ] SQL database integration
- [ ] HuggingFace API proxy for model analysis
- [ ] Real-time benchmarking on multiple backends
- [ ] Advanced quantization (QAT, mixed precision)
- [ ] Automated hyperparameter tuning
- [ ] Model comparison across versions
- [ ] Cloud deployment integration
- [ ] REST API for programmatic access

## Contributing

To extend the platform:

1. **Add new optimization techniques** in `modules/optimization.py`
2. **Create custom visualizations** in `modules/dashboard.py`
3. **Add deployment targets** in `modules/deployment.py`
4. **Extend AI consultant** knowledge in `modules/ai_consultant.py`

## License

This project is created as a demonstration platform for AI model optimization workflows.

## Support

For issues, questions, or suggestions:
1. Check this README
2. Use the AI Consultant feature
3. Review the code documentation
4. Examine the example workflows

## Acknowledgments

- Built with Streamlit
- Optimization powered by PyTorch
- Visualizations by Plotly
- Reports generated with ReportLab

---

**Version:** 1.0.0  
**Last Updated:** January 2026  
**Status:** Production Ready
