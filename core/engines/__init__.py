"""
Core Engines for Agent Power-Up
===============================

This module contains the engine components that power up different types of agents:
- DSPy engine configuration utilities
- Direct LLM engine configuration utilities

These engines handle the LLM setup concerns separate from agent business logic.
"""

# Import engine utilities for easy access
try:
    from .dspy_engine import (
        create_dspy_lm, configure_dspy, quick_configure, configure_default,
        create_openai_lm, create_anthropic_lm, create_cerebras_lm, create_google_lm,
        get_available_providers, get_available_models, validate_dspy_setup,
        DEFAULT_CONFIGS
    )
    DSPY_ENGINE_AVAILABLE = True
except ImportError:
    DSPY_ENGINE_AVAILABLE = False

try:
    from .llm_engine import (
        create_llm_config, call_llm, simple_chat, validate_llm_setup,
        create_openai_config, create_anthropic_config, create_cerebras_config, create_google_config,
        create_default_config, quick_chat, get_response_text,
        DEFAULT_LLM_CONFIGS
    )
    LLM_ENGINE_AVAILABLE = True
except ImportError:
    LLM_ENGINE_AVAILABLE = False

__all__ = [
    "DSPY_ENGINE_AVAILABLE",
    "LLM_ENGINE_AVAILABLE",
]

# Conditionally export DSPy engine functions
if DSPY_ENGINE_AVAILABLE:
    __all__.extend([
        "create_dspy_lm", "configure_dspy", "quick_configure", "configure_default",
        "create_openai_lm", "create_anthropic_lm", "create_cerebras_lm", "create_google_lm",
        "get_available_providers", "get_available_models", "validate_dspy_setup",
        "DEFAULT_CONFIGS"
    ])

# Conditionally export LLM engine functions  
if LLM_ENGINE_AVAILABLE:
    __all__.extend([
        "create_llm_config", "call_llm", "simple_chat", "validate_llm_setup",
        "create_openai_config", "create_anthropic_config", "create_cerebras_config", "create_google_config",
        "create_default_config", "quick_chat", "get_response_text",
        "DEFAULT_LLM_CONFIGS"
    ])