"""
Core interfaces and types for the decoupled tree traversal architecture.

This module provides the standardized contracts that allow different components
to work together without tight coupling:

- Tree Providers (ICD, CPT) implement TreeIndex
- Traversal Engines implement TraversalEngine with domain-specific rules
- Orchestrators (Burr, neutral) use these interfaces for step management
- Agents select candidates using only the Core contracts

Version: 1.0.0
"""

CORE_API_VERSION = "1.0.0"

# Import all core types and interfaces for easy access
from .types import (
    NodeId,
    Action,
    GuardOutcome,
    GuardResult,
    RichCandidate,
    RunContext,
    QueryState,
)

from .interfaces import (
    NodeView,
    TreeIndex,
    TraversalEngine,
    DecisionContext,
    CandidateAgent,
    CandidateScorer,
)

from .llm import (
    PROVIDER_REGISTRY,
    get_provider_config,
    get_model_config,
    list_providers,
    list_models,
    validate_provider_setup,
)

# Engine configuration utilities (optional imports - fail gracefully if dependencies missing)
try:
    from .engines import (
        create_dspy_lm, configure_dspy, quick_configure, configure_default,
        create_openai_lm, create_anthropic_lm, create_cerebras_lm, create_google_lm,
        DSPY_ENGINE_AVAILABLE
    )
    DSPY_CONFIG_AVAILABLE = DSPY_ENGINE_AVAILABLE
except ImportError:
    DSPY_CONFIG_AVAILABLE = False

try:
    from .engines import (
        create_llm_config, call_llm, simple_chat, validate_llm_setup,
        create_openai_config, create_anthropic_config, create_cerebras_config, create_google_config,
        LLM_ENGINE_AVAILABLE
    )
    LLM_CONFIG_AVAILABLE = LLM_ENGINE_AVAILABLE
except ImportError:
    LLM_CONFIG_AVAILABLE = False

__all__ = [
    "CORE_API_VERSION",
    # Types
    "NodeId",
    "Action", 
    "GuardOutcome",
    "GuardResult",
    "RichCandidate",
    "RunContext",
    "QueryState",
    # Interfaces
    "NodeView",
    "TreeIndex",
    "TraversalEngine", 
    "DecisionContext",
    "CandidateAgent",
    "CandidateScorer",
    # LLM Provider Configuration
    "PROVIDER_REGISTRY",
    "get_provider_config",
    "get_model_config",
    "list_providers", 
    "list_models",
    "validate_provider_setup",
    # Configuration availability flags
    "DSPY_CONFIG_AVAILABLE",
    "LLM_CONFIG_AVAILABLE",
]