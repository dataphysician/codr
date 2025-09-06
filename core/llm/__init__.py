"""
Core LLM Provider Configuration
===============================

This module contains the centralized LLM provider registry and utilities
that are shared across different agent engines (DSPy, direct LLM calls, etc.).

The provider registry defines a framework-agnostic configuration format
that can be consumed by different LLM frameworks.
"""

# Import provider utilities
from .llm_providers import (
    PROVIDER_REGISTRY,
    get_provider_config,
    get_model_config, 
    list_providers,
    list_models,
    validate_provider_setup,
    DEFAULT_MODEL_CONFIGS,
    get_default_config,
    list_default_configs
)

__all__ = [
    "PROVIDER_REGISTRY",
    "get_provider_config",
    "get_model_config",
    "list_providers", 
    "list_models",
    "validate_provider_setup",
    "DEFAULT_MODEL_CONFIGS",
    "get_default_config", 
    "list_default_configs"
]