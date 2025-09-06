# CODR Examples

This folder contains end-to-end example implementations demonstrating advanced ICD-10 coding workflows.

## Examples

### 1. Burr + LLM Agent (GPT-4o)
**File:** `burr_llm_example.py`

Demonstrates intelligent clinical document coding using:
- **Burr state machine** for workflow orchestration
- **GPT-4o** for context-aware decision making
- **Clinical scenario processing** with real-world cases
- **End-to-end coding pipeline** from notes to ICD-10 codes

**Prerequisites:**
```bash
export OPENAI_API_KEY="your-key-here"
pip install burr litellm
```

**Run:**
```bash
PYTHONPATH=. python examples/burr_llm_example.py
```

### 2. Burr + DSPy Agent (Qwen-235B)
**File:** `burr_dspy_example.py`

Demonstrates advanced clinical reasoning using:
- **Burr state machine** for complex workflow management
- **Qwen-235B with reasoning** capabilities for medical decisions
- **DSPy framework** for structured reasoning and optimization
- **Reasoning trace analysis** and confidence tracking

**Prerequisites:**
```bash
export CEREBRAS_API_KEY="your-key-here"  
pip install burr dspy
```

**Run:**
```bash
PYTHONPATH=. python examples/burr_dspy_example.py
```

## Key Features Demonstrated

### Clinical Intelligence
- ✅ **Clinical document understanding** and context extraction
- ✅ **Multi-step reasoning** through complex medical scenarios  
- ✅ **Domain-specific knowledge** application for ICD-10 coding
- ✅ **Confidence scoring** and decision validation

### System Architecture
- ✅ **State management** and workflow persistence
- ✅ **Error handling** and recovery mechanisms
- ✅ **Performance optimization** through structured frameworks
- ✅ **Modular design** allowing different agent backends

### Workflow Orchestration
- ✅ **Burr state machine** integration for complex workflows
- ✅ **Policy-based routing** for different coding scenarios
- ✅ **Parallel processing** capabilities for multiple diagnoses
- ✅ **Audit trails** with complete reasoning traces

## Output Examples

Both examples provide detailed output including:
- **Step-by-step reasoning** traces
- **Confidence scores** for each decision
- **Performance metrics** (timing, success rates)
- **Clinical scenario analysis** with expected vs. actual codes
- **Quality assessments** of reasoning processes

## Test Coverage

The original example functionality has been converted to comprehensive test suites in the `tests/` folder:
- `test_tree_interface.py` - Tree navigation capabilities
- `test_traversal_engine.py` - Domain rule enforcement  
- `test_agents.py` - Agent decision making
- `test_orchestration.py` - Workflow execution
- `test_end_to_end.py` - Complete system integration

Run all tests with:
```bash
python tests/run_tests.py
```