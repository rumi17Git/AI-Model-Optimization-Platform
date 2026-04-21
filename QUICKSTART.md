# Quick Start Guide

## Getting Started in 5 Minutes

### 1. Installation (2 minutes)

```bash
# Navigate to project directory
cd ai_optimization_platform

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application (1 minute)

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 3. Login (30 seconds)

Use demo credentials:
- **Username:** demo
- **Password:** demo123

### 4. Try the Platform (2 minutes)

#### Option A: Use Sample Model (Quickest)
1. Go to "Upload Model/Data"
2. Click "🎲 Use Sample Model"
3. Navigate to "Dashboard" to see metrics
4. Go to "Optimization Engine"
5. Select "Quantization" and "Pruning"
6. Click "🚀 Start Optimization"
7. View results on Dashboard
8. Generate PDF report

#### Option B: Upload Your Own Model
1. Go to "Upload Model/Data"
2. Upload your .pt, .pth, or .onnx file
3. Follow steps 3-8 from Option A

## Example Workflow

### Scenario: Optimize a PyTorch model for mobile deployment

```
1. Upload Model
   └─> "Upload Model/Data" tab
       └─> Choose your model.pt file
       └─> Wait for analysis

2. Check Current Metrics
   └─> "Dashboard" tab
       └─> Note model size, latency
       └─> Identify optimization needs

3. Ask AI Consultant (Optional)
   └─> "AI Consultant" tab
       └─> Ask: "What optimization for mobile?"
       └─> Get personalized recommendations

4. Configure Optimization
   └─> "Optimization Engine" tab
       └─> Enable Quantization
       └─> Enable Pruning (30%)
       └─> Click "Start Optimization"

5. Review Results
   └─> "Dashboard" tab
       └─> Compare before/after metrics
       └─> Generate PDF Report

6. Check Sustainability
   └─> "Carbon Tracker" tab
       └─> View CO₂ savings
       └─> See energy reduction

7. Export Model
   └─> "Deployment Hub" tab
       └─> Select ONNX format
       └─> Download optimized model
       └─> Read deployment guide
```

## Key Features to Explore

### Dashboard
- View all metrics at a glance
- Interactive performance charts
- Optimization history timeline
- Generate PDF reports

### Optimization Engine
**Quantization:**
- Fastest optimization
- 4x size reduction typical
- <1% accuracy loss
- Great for inference speed

**Pruning:**
- Configurable (10-90%)
- Reduces model size
- May need fine-tuning
- Balance size vs. accuracy

**Knowledge Distillation:**
- Maximum compression
- Maintains accuracy
- Requires more time
- Best for large models

### AI Consultant
Ask questions like:
- "What's the best optimization for my model?"
- "How do I reduce accuracy loss?"
- "Tell me about quantization"
- "Deploy to Raspberry Pi?"

### Deployment Hub
- Export to ONNX (recommended)
- Platform-specific guides
- Code examples
- Deployment readiness check

### Carbon Tracker
- Real-time CO₂ estimates
- Energy consumption
- Environmental impact
- Regional calculations

## Tips for Best Results

### 1. Start Conservative
```
First run:  Quantization only
Second run: Quantization + Light Pruning (20%)
Third run:  Increase pruning if needed (30-40%)
```

### 2. Monitor Accuracy
- Check accuracy after each optimization
- If drop > 2%, reduce pruning amount
- Consider fine-tuning if needed

### 3. Use AI Consultant
- Get personalized advice
- Understand trade-offs
- Platform-specific tips

### 4. Generate Reports
- Document your optimization process
- Share results with team
- Track improvements over time

## Common Use Cases

### Mobile App (iOS/Android)
**Goal:** <50MB model, <50ms latency
**Recommended:**
- Quantization (INT8)
- Moderate Pruning (30-40%)
- Export to ONNX or TFLite

### Edge Device (Raspberry Pi, Jetson)
**Goal:** <10MB model, <30ms latency
**Recommended:**
- Aggressive Quantization
- Heavy Pruning (50-70%)
- Knowledge Distillation
- Export to ONNX

### Cloud Deployment
**Goal:** Cost optimization, high throughput
**Recommended:**
- Light Quantization
- Minimal Pruning (10-20%)
- Focus on batch inference
- Export to ONNX

### Web Browser
**Goal:** <5MB model, instant loading
**Recommended:**
- Maximum Quantization
- Heavy Pruning
- Knowledge Distillation
- Export to ONNX (for ONNX.js)

## Troubleshooting Quick Fixes

### "Model won't upload"
- Check file format (.pt, .pth, .onnx)
- Try sample model first
- Ensure file isn't corrupted

### "Optimization fails"
- Try one technique at a time
- Start with Quantization only
- Check model architecture compatibility

### "Accuracy drops too much"
- Reduce pruning amount
- Use quantization only
- Try fine-tuning approach

### "Can't generate PDF"
```bash
pip install --upgrade reportlab
```

## Next Steps

After completing the quick start:

1. **Experiment** with different optimization combinations
2. **Compare** results across multiple runs
3. **Generate** reports for documentation
4. **Deploy** your optimized model
5. **Monitor** performance in production

## Need Help?

1. Check the full README.md
2. Use the AI Consultant in the app
3. Review deployment guides
4. Examine the code examples

---

**Ready to optimize?** Just run `streamlit run app.py` and start!
