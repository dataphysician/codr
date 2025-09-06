# CODR: Code Ontology-Driven Reasoning

**A Protocol-Driven AI Framework for Intelligent Navigation of Hierarchical Coding Systems**

CODR provides agentic AI tools for intelligent code assignment across any hierarchical classification system. While originally designed for medical coding (ICD-10-CM, CPT, SNOMED-CT), CODR's protocol-driven architecture supports legal taxonomies, academic classifications, organizational standards, and any structured coding domain.

## What CODR Can Do

**Transform Complex Coding Workflows** from manual, error-prone processes into intelligent, evidence-based automation:

- **Document Analysis**: Process unstructured text and extract relevant coding evidence
- **Intelligent Navigation**: Recursively traverse hierarchical coding systems with contextual reasoning
- **Multi-Level Automation**: Choose from manual control, direct LLM automation, or self-optimizing DSPy agents
- **Evidence Generation**: Build comprehensive justifications with citations and audit trails
- **Performance Optimization**: Handle 10,000+ code hierarchies with sub-millisecond operations
- **Universal Compatibility**: Work with any LLM provider and orchestration framework

### Example: ICD-10-CM Medical Coding Journey
```
Source Document: "Patient presents with Type 1 diabetes mellitus with diabetic nephropathy"
                                    ↓
Chapter Selection: "4 - Endocrine, nutritional and metabolic diseases"
                                    ↓
Block Navigation: "E08-E13 - Diabetes mellitus"
                                    ↓
Category Refinement: "E10 - Type 1 diabetes mellitus"
                                    ↓
Specific Code: "E10.2 - Type 1 diabetes mellitus with kidney complications"
```

But CODR works equally well for legal case classifications, academic subject taxonomies, product categorization systems, or any structured hierarchy.

## Core Features

### **Protocol-Driven Architecture**
- **Universal Compatibility**: Any agent works with any orchestrator and coding system
- **Structural Typing**: Python Protocols enable self-optimizing agents without breaking contracts
- **Clean Separation**: Tree structure ↔ Domain rules ↔ Intelligence ↔ Workflow management
- **Domain-Centric Organization**: Co-located domain code (trees, traversals, factories) for better modularity
- **Independent Development**: Each domain can be developed, tested, and maintained separately
- **Scalable Extension**: Add new domains without affecting existing implementations

### **Three-Tier Agent Intelligence**
- **Base Agents**: Manual/deterministic control for testing, validation, and human oversight
- **LLM Agents**: Direct API automation with graceful degradation, multi-provider support, dynamic prompt optimization, and configurable search strategies
- **DSPy Agents**: Self-optimizing structured output with citation extraction, training, and configurable beam search

### **Multi-Provider LLM Support**
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **Anthropic**: Claude Sonnet 4, Claude 3.5 Sonnet, Claude 3 series
- **Cerebras**: Llama 3.1-8B, Qwen 3-235B-Thinking, Qwen 3-32B
- **Google**: Gemini 1.5 Pro, Gemini 2.0 Flash Thinking
- **Custom**: Fully extensible provider registry for any compatible API

### **Multiple Orchestration Patterns**
- **Simple**: Basic workflow management for straightforward use cases
- **Comprehensive**: Enhanced tracking, metrics, and audit trails
- **Burr Integration**: State machine orchestration with persistence and rewind capabilities

### **Evidence-Based Reasoning**
- **Citation Extraction**: Automatic identification of supporting evidence from source documents
- **Contextual Navigation**: Progressive reasoning through each hierarchy level with growing clinical context
- **Audit Trails**: Comprehensive tracking for compliance and validation workflows
- **Performance Metrics**: Sub-millisecond operations with 10,000+ node hierarchies

### **Enhanced Clinical Intelligence**
- **Smart Chapter Selection**: LLM agents receive rich candidate descriptions with clinical context matching
- **Growing Context Integration**: Agents build increasingly specific understanding through iterative decision-making
- **Feedback-Driven Learning**: Rewind functionality with clinical context preservation and mistake correction
- **Code Provenance Tracking**: Complete decision path history for audit trails and learning enhancement
- **Dynamic Prompt Optimization**: Node-specific optimization rules with LLM-driven metaprompt generation and persistent storage
- **Configurable Search Strategies**: Simple integer-based candidate selection (1 for greedy, 3-5 for beam, 10+ for exhaustive) for optimal performance vs coverage balance

## Modular Architecture

CODR's protocol-driven design provides clean separation of concerns with domain-centric organization:

### **Core Foundation Layer** (`core/`)
- **Protocol Interfaces**: `TreeIndex`, `TraversalEngine`, `CandidateAgent` with structural typing
- **Data Types**: `NodeId`, `Action`, `QueryState`, `DecisionContext`, `RichCandidate`
- **LLM Infrastructure**: Centralized provider registry and engine configuration
- **Agent Frameworks**: Base agents, LLM agents, and DSPy agents with unified interfaces

### **Domain-Centric Implementation Layer** (`core/domains/`)
CODR organizes domain-specific code for better modularity and maintainability:

- **Medical Domain** (`core/domains/medical/`):
  - **Trees** (`core/domains/medical/trees/`): ICD-10-CM navigator (✅), CPT (🔄), SNOMED (🔄)
  - **Traversals** (`core/domains/medical/traversals/`): Medical coding rules, guards, and state management
  - **Factories** (`core/domains/medical/factories/`): Domain-specific agent creation functions
- **Future Domains**: Legal (`core/domains/legal/`), Academic (`core/domains/academic/`), etc.
- **Extensible**: Add any hierarchical coding system by implementing protocols in its own domain directory

### **Intelligence Layer** (`core/dag_agents/`)
```
Manual Control    →  Base Agents     →  DeterministicAgent, RandomAgent
Direct Automation →  LLM Agents      →  Multi-provider API automation  
Self-Optimization →  DSPy Agents     →  Structured output + training
```

### **Orchestration Layer** (`core/orchestrators/`)
```
Basic Workflows   →  SimpleOrchestrator           →  Core traverse-decide-apply pattern
Rich Tracking     →  ComprehensiveOrchestrator     →  Metrics, audit trails, validation
State Machines    →  BurrOrchestrator             →  Persistence, rewind, parallel processing
```

## Quick Start

### 1. **Choose Your Intelligence Level**
```python
# Manual control (zero dependencies)
from core.dag_agents import DeterministicAgent
agent = DeterministicAgent()

# Direct LLM automation
from core.domains.medical.factories import create_icd10_llm_agent
agent = create_icd10_llm_agent("openai", "gpt-4o")

# Add node-specific optimization rules
agent.add_rule("E11.22", "Provide clearer clinical differentiation between diabetic nephropathy and chronic kidney disease")
agent.add_rule("ROOT", "Focus on primary clinical presentation vs secondary complications for chapter selection")

# Configure search strategy with simple int values using direct agent creation
from core.dag_agents import create_llm_agent
greedy_agent = create_llm_agent("openai", "gpt-4o", domain="icd10", max_candidates=1)      # greedy search
beam_agent = create_llm_agent("openai", "gpt-4o", domain="icd10", max_candidates=3)        # beam search  
exhaustive_agent = create_llm_agent("openai", "gpt-4o", domain="icd10", max_candidates=10) # exhaustive search

# Self-optimizing DSPy with intuitive parameters
from core.domains.medical.factories import create_icd10_dspy_agent
agent = create_icd10_dspy_agent(
    provider="anthropic", 
    model="claude-sonnet-4",
    max_candidates=3,            # Beam search for better coverage
    reasoning_style="detailed_clinical",  # Structured clinical reasoning
    temperature=0.2              # Lower temperature for consistency
)

# Add node-specific optimization (consistent API for both LLM and DSPy agents)
# agent.add_rule("E11.22", feedback="Focus on differential diagnosis between diabetic nephropathy and chronic kidney disease")
# agent.add_rule("ROOT", feedback="Prioritize systematic chapter selection based on primary clinical presentation")
```

### 2. **Select Your Orchestration Pattern**
```python
# Basic workflow
from core.orchestrators import BaseOrchestrator
orchestrator = BaseOrchestrator(tree, traversal, agent)

# State machine with persistence  
from core.orchestrators import create_burr_app
burr_app = create_burr_app(agent)
result = burr_app.run(inputs={"start_code": "E10", "source_document": "..."})
```

### 3. **Complete System Setup**
```python
# Import domain-specific components
from core.domains.medical.trees import create_navigator
from core.domains.medical.traversals import create_icd_traversal_engine
from core.domains.medical.factories import create_icd10_dspy_agent

# Create complete system
tree = create_navigator()
traversal = create_icd_traversal_engine()
agent = create_icd10_dspy_agent("openai", "gpt-4o")

# Set up orchestrator with rewind capabilities
from core.orchestrators import BaseOrchestrator, create_burr_app
orchestrator = BaseOrchestrator(tree, traversal, agent)

# For advanced workflows with rewind/feedback
burr_app = create_burr_app(agent)

# Run end-to-end code assignment with enhanced clinical context
result = orchestrator.execute_workflow(
    start_code="ROOT",  # Always start from ROOT for comprehensive traversal
    context={
        "source_document": "Patient presents with Type 1 diabetes with nephropathy...",
        "clinical_context": "Polyuria, polydipsia, elevated glucose, proteinuria"
    }
)

# Or use Burr for state management and rewind capabilities
burr_result = burr_app.run(inputs={
    "start_code": "ROOT", 
    "source_document": "Patient presents with diabetes...",
    "clinical_context": "Detailed clinical scenario with symptoms"
})

print(f"Selected Code: {result.final_code}")
print(f"Evidence: {result.citations}")
print(f"Agent Reasoning: {result.agent_reasoning}")

# Rewind example (when using Burr)
# burr_app.run(halt_after=['rewind_to_node'], inputs={
#     'target_node': 'chapter_4', 
#     'feedback': {'reason': 'Wrong chapter', 'correction': 'This is diabetes, use endocrine'}
# })
```

## Use Cases

- **Medical Coding**: ICD-10-CM diagnosis coding, CPT procedure coding, SNOMED-CT terminology
- **Legal Classification**: Case law categorization, regulatory compliance coding
- **Academic Systems**: Subject classification, research taxonomy assignment  
- **Enterprise**: Product categorization, organizational standards, compliance frameworks
- **Research**: Comparative analysis of LLM performance on structured classification tasks

## Why CODR?

- **Universal Compatibility**: Protocol-driven design works with any provider, framework, or domain
- **Scalable Intelligence**: Progressive automation from manual to self-optimizing with contextual learning
- **Production Ready**: Sub-millisecond performance with comprehensive test coverage
- **Audit Compliant**: Complete evidence trails with rewind capabilities for regulatory and compliance workflows
- **Extensible**: Add new domains, providers, or agents without core changes
- **Self-Correcting**: Advanced rewind system enables learning from mistakes with preserved clinical context

## Next Steps

- **Full Documentation**: See [AGENTS.md](./AGENTS.md) for comprehensive architectural specification
- **Run Examples**: Check `examples/` for complete ICD-10 traversal implementations
- **Run Tests**: Execute `tests/` suite to validate all components and protocols
- **Extend**: Add new domains by implementing `TreeIndex` and `TraversalEngine` protocols

---

**CODR transforms complex coding workflows into intelligent, evidence-based automation while maintaining the flexibility to work with any hierarchical classification system.**