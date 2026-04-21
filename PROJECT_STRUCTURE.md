# Project Structure

## Clean Architecture - Separation of Concerns

```
ai_optimization_platform/
├── app.py                          # Main entry point (routing only)
│
├── views/                          # UI Layer (Streamlit pages)
│   ├── dashboard_page.py           # Dashboard UI
│   ├── uploader_page.py            # Upload UI
│   ├── ai_consultant_page.py       # AI chat UI
│   ├── optimization_page.py        # Optimization UI
│   ├── history_page.py             # History UI
│   ├── deployment_page.py          # Deployment UI
│   └── carbon_tracker_page.py      # Carbon tracking UI
│
├── modules/                        # Business Logic Layer
│   ├── model_loader.py             # Model loading logic
│   ├── model_analyzer.py           # Model analysis logic
│   ├── architecture_detector.py    # Architecture detection
│   ├── optimization_engine.py      # Optimization algorithms
│   ├── groq_client.py              # LLM client
│   ├── analytics.py                # Analytics & charts
│   ├── database.py                 # Database ORM
│   └── report_generator.py         # PDF generation
│
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
│
├── requirements.txt                # Dependencies
├── requirements-minimal.txt        # Minimal dependencies
├── init_db.py                      # Database initialization
├── generate_demo_models.py         # Demo model generator
├── README.md                       # Documentation
└── QUICKSTART.md                   # Quick start guide
```

## Architecture Principles

### 1. Separation of Concerns
- **Pages (UI)**: Handle user interaction, display, and form inputs
- **Modules (Logic)**: Handle business logic, computations, and data processing
- **No mixing**: Pages never contain business logic, modules never contain UI code

### 2. Single Responsibility
- Each module has one clear purpose
- Each page manages one view/feature
- Functions do one thing well

### 3. Clean Imports
- Pages import from modules
- Modules don't import from pages
- No circular dependencies

### 4. Testable
- Business logic can be tested without UI
- Modules are pure Python (no Streamlit dependency)
- Easy to unit test

## Module Descriptions

### Views (UI Layer)

**dashboard_page.py**
- Displays performance metrics
- Shows comparison charts
- Generates reports
- Uses: `analytics.py`, `report_generator.py`

**uploader_page.py**
- File upload interface
- Format detection display
- Model info display
- Uses: `model_loader.py`, `model_analyzer.py`

**ai_consultant_page.py**
- Chat interface
- Context display
- Conversation management
- Uses: `groq_client.py`

**optimization_page.py**
- Technique selection UI
- Configuration inputs
- Results display
- Uses: `optimization_engine.py`, `model_analyzer.py`

**history_page.py**
- History table display
- Statistics dashboard
- Uses: `pandas`

**deployment_page.py**
- Export options
- Download buttons
- Uses: `torch`

**carbon_tracker_page.py**
- Environmental metrics
- Impact calculations
- Uses: Simple calculations

### Modules (Business Logic)

**model_loader.py**
- Flexible PyTorch loading
- Format detection
- Architecture reconstruction
- No UI dependencies

**model_analyzer.py**
- Parameter counting
- Size calculation
- Latency estimation
- Returns metrics dictionary

**architecture_detector.py**
- Detects known architectures
- Loads from state dict
- Pattern matching

**optimization_engine.py**
- Quantization algorithms
- Pruning implementation
- Knowledge distillation
- Pure PyTorch operations

**groq_client.py**
- Groq API communication
- Context building
- Topic filtering
- Returns strings

**analytics.py**
- Comparison calculations
- Chart generation (Plotly)
- Metrics aggregation

**database.py**
- SQLAlchemy ORM
- CRUD operations
- Data persistence

**report_generator.py**
- PDF generation
- ReportLab usage
- Report templates

## Benefits of This Structure

### ✅ Maintainability
- Easy to find code
- Clear organization
- Predictable locations

### ✅ Testability
- Modules can be tested independently
- No UI mocking needed
- Unit tests are simple

### ✅ Reusability
- Modules can be used in other projects
- Business logic is portable
- CLI tools can use same modules

### ✅ Scalability
- Easy to add new pages
- Easy to add new modules
- Clear extension points

### ✅ Collaboration
- Frontend/backend separation
- Multiple developers can work simultaneously
- Clear interfaces between layers

## Adding New Features

### Adding a New Page

1. Create `views/new_feature_page.py`
2. Import required modules
3. Create `new_feature_page()` function
4. Add to `app.py` routing

### Adding New Business Logic

1. Create `modules/new_logic.py`
2. Implement pure functions/classes
3. No Streamlit imports
4. Import in relevant pages

### Example

```python
# modules/new_logic.py (Business Logic)
def calculate_something(data):
    """Pure function - no UI"""
    result = data * 2
    return result

# views/new_page.py (UI)
import streamlit as st
from modules.new_logic import calculate_something

def new_page():
    st.title("New Feature")
    data = st.number_input("Input")
    if st.button("Calculate"):
        result = calculate_something(data)
        st.write(result)
```

## Code Quality

- ✅ Type hints where appropriate
- ✅ Docstrings for all functions
- ✅ No hardcoded paths
- ✅ Error handling
- ✅ Clean imports
- ✅ Consistent naming

## Future Enhancements

With this structure, it's easy to add:
- API endpoints (FastAPI alongside Streamlit)
- CLI tools (reuse modules)
- Background workers (use modules directly)
- Tests (pytest on modules)
- Alternative frontends (same modules)
