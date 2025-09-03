"""
DSPy Agent Configuration for Multi-Provider LLM Support
======================================================

This module provides DSPy LLM configurations for different providers and models.
Supports easy provider switching with configurable parameters via **kwargs.
"""

import os
import dspy
from typing import Dict, Any
from pydantic import BaseModel, Field


# Provider model configurations with specific parameters
PROVIDER_CONFIGS = {
    "openai": {
        "models": {
            "gpt-5-nano": {
                "model": "openai/gpt-5-nano",
                "api_key_env": "OPENAI_API_KEY",
                "max_tokens": 20000,
                "temperature": 1.0,
            },
            "o3": {
                "model": "openai/o3", 
                "api_key_env": "OPENAI_API_KEY",
                "max_tokens": 20000,
                "temperature": 1.0,
            },
            "gpt-4o": {
                "model": "openai/gpt-4o",
                "api_key_env": "OPENAI_API_KEY"
            },
            "gpt-4o-mini": {
                "model": "openai/gpt-4o-mini", 
                "api_key_env": "OPENAI_API_KEY"
            }
        }
    },
    "anthropic": {
        "models": {
            "claude-opus-4-1-20250805": {
                "model": "anthropic/claude-opus-4-1-20250805",
                "api_key_env": "ANTHROPIC_API_KEY"
            },
            "claude-sonnet-4-20250514": {
                "model": "anthropic/claude-sonnet-4-20250514",
                "api_key_env": "ANTHROPIC_API_KEY"
            }
        }
    },
    "gemini": {
        "models": {
            "gemini-2.0-flash-exp": {
                "model": "gemini/gemini-2.0-flash-exp",
                "api_key_env": "GEMINI_API_KEY"
            },
            "gemini-1.5-pro": {
                "model": "gemini/gemini-1.5-pro",
                "api_key_env": "GEMINI_API_KEY"
            },
            "gemini-1.5-flash": {
                "model": "gemini/gemini-1.5-flash",
                "api_key_env": "GEMINI_API_KEY"
            }
        }
    },
    "cerebras": {
        "models": {
            "llama3.1-8b": {
                "model": "cerebras/llama3.1-8b",
                "api_key_env": "CEREBRAS_API_KEY"
            },
            "llama-3.3-70b": {
                "model": "cerebras/llama-3.3-70b", 
                "api_key_env": "CEREBRAS_API_KEY"
            },
            "qwen-3-235b-a22b-instruct-2507": {
                "model": "cerebras/qwen-3-235b-a22b-instruct-2507",
                "api_key_env": "CEREBRAS_API_KEY",
            },
            "qwen-3-235b-a22b-thinking-2507": {
                "model": "cerebras/qwen-3-235b-a22b-thinking-2507",
                "api_key_env": "CEREBRAS_API_KEY",
            },
        }
    },
    # Example configurations for future custom models that need api_base/base_url
    "custom": {
        "models": {
            "custom-llm-1": {
                "model": "custom/llm-model-1",
                "api_key_env": "CUSTOM_API_KEY",
                "api_base_env": "CUSTOM_API_BASE",  # Will pull from environment if available
                "max_tokens": 4000,
                "temperature": 0.7
            },
            "vllm-hosted": {
                "model": "vllm/llama-7b", 
                "api_key_env": "VLLM_API_KEY",
                "base_url_env": "VLLM_BASE_URL",  # Alternative naming for base URL
                "max_tokens": 2048
            },
            "local-model": {
                "model": "local/custom-model",
                "api_key_env": "LOCAL_API_KEY",
                "api_base_env": "LOCAL_API_BASE",
                "temperature": 0.5,
                "max_tokens": 8000
            }
        }
    }
}


def get_provider_models(provider: str) -> list[str]:
    """Get list of available models for a provider."""
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(f"Unsupported provider: {provider}")
    
    return list(PROVIDER_CONFIGS[provider]["models"].keys())


def get_model_config(provider: str, model_name: str) -> dict[str, Any]:
    """Get configuration parameters for a specific provider/model combination."""
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(f"Unsupported provider: {provider}")
    
    provider_config = PROVIDER_CONFIGS[provider]
    
    if model_name not in provider_config["models"]:
        raise ValueError(f"Model {model_name} not found for provider {provider}")
    
    model_config = provider_config["models"][model_name].copy()
    
    # Get API key from environment
    api_key_env = model_config.pop("api_key_env")
    api_key = os.getenv(api_key_env)
    
    if not api_key:
        raise ValueError(f"API key not found in environment variable: {api_key_env}")
    
    model_config["api_key"] = api_key
    
    # Get API base URL from environment if specified
    if "api_base_env" in model_config:
        api_base_env = model_config.pop("api_base_env")
        api_base = os.getenv(api_base_env)
        if api_base:
            model_config["api_base"] = api_base
    
    # Get base URL from environment if specified (alternative naming)
    if "base_url_env" in model_config:
        base_url_env = model_config.pop("base_url_env")
        base_url = os.getenv(base_url_env)
        if base_url:
            model_config["base_url"] = base_url
    
    return model_config


def create_dspy_lm(provider: str, model_name: str, **kwargs) -> dspy.LM:
    """Create a DSPy LM instance for the specified provider and model."""
    config = get_model_config(provider, model_name)
    config.update(kwargs)
    return dspy.LM(**config)


def configure_dspy(provider: str, model_name: str) -> dspy.LM:
    """Configure DSPy with the specified provider and model, return the LM instance."""
    lm = create_dspy_lm(provider, model_name)
    dspy.configure(lm=lm)
    return lm




# Pre-configured LM instances for common use cases
def create_openai_lm() -> dspy.LM:
    """Create OpenAI model LM instance."""
    return create_dspy_lm("openai", "gpt-4o")


def create_cerebras_lm() -> dspy.LM:
    """Create Cerebras model LM instance.""" 
    return create_dspy_lm("cerebras", "llama3.1-8b")


def create_anthropic_lm() -> dspy.LM:
    """Create Anthropic model LM instance."""
    return create_dspy_lm("anthropic", "claude-sonnet-4-20250514")


# Convenience function for quick setup
def quick_configure(provider: str, model_name: str, **kwargs) -> dspy.LM:
    """Quickly configure DSPy with a provider/model combination."""
    try:
        config = get_model_config(provider, model_name)
        config.update(kwargs)
        lm = dspy.LM(**config)
        dspy.configure(lm=lm)
        print(f"✅ DSPy configured with {provider}/{model_name}")
        return lm
    except Exception as e:
        print(f"❌ Failed to configure DSPy: {e}")
        raise


class CodeNode(BaseModel):
    code: str = Field(pattern=r'\(([A-Z][0-9][0-9](?:\.[0-9A-X-]+)?(?:-[A-Z][0-9][0-9](?:\.[0-9A-X-]+)?)?(?:, ?[A-Z][0-9][0-9](?:\.[0-9A-X-]+)?)*(?:\.?-)?)\)')
    name: str
    instructions: list[tuple[str, str]]


class NodeTransition(dspy.Signature):
    """
    The transition from one node to another.
    Input includes:
    - a clinical document for grounded references
    - a list of any node parents and ancestors
    - the next nodes to traverse
    Outputs include:
    - the decision/s whether to traverse a node, terminate the trajectory, report the current node, start a parallel pathway
    - a node to report for the transition
    - a optional citation from the clinical document
    """

    clinical_note: str = dspy.InputField()
    node_ancestry: list[str] = dspy.InputField()
    next_options: list[Nodes]


# Example usage and testing
if __name__ == "__main__":
    print("DSPy Agent Configuration Demo")
    print("=" * 50)
    
    # List available providers and models
    for provider, config in PROVIDER_CONFIGS.items():
        models = list(config["models"].keys())
        
        print(f"\n{provider.upper()}:")
        print(f"  Models: {', '.join(models)}")
    
    print(f"\n{'=' * 50}")
    print("Usage Examples:")
    print("=" * 50)
    print("# Quick configuration")
    print('lm = quick_configure("openai", "gpt-4o")')
    print("\n# Configuration with custom parameters") 
    print('lm = quick_configure("cerebras", "qwen-3-235b-a22b-thinking-2507", reasoning_effort="high")')
    print("\n# Manual configuration")
    print('lm = create_dspy_lm("anthropic", "claude-sonnet-4-20250514")')
    print("dspy.configure(lm=lm)")
    print("\n# Custom models with api_base (future use)")
    print('# Set environment: CUSTOM_API_KEY=xxx, CUSTOM_API_BASE=https://api.example.com/v1')
    print('# lm = quick_configure("custom", "custom-llm-1")')
    print('# lm = quick_configure("custom", "vllm-hosted")  # Uses base_url_env')