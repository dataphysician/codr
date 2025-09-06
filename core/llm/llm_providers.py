"""
LLM Provider Configuration Registry

Centralized configuration for all LLM providers used across different agent types.
This allows DSPy agents, raw LLM agents, and other frameworks to share the same
provider definitions while maintaining their specific integration patterns.
"""

import os
from typing import Any

# Centralized LLM provider registry
# Each agent framework can import and configure these as needed
# Model names include provider prefixes for compatibility with DSPy and LiteLLM
PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
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
                "api_key_env": "OPENAI_API_KEY",
                "max_tokens": 4096,
                "temperature": 0.7,
            },
            "gpt-4o-mini": {
                "model": "openai/gpt-4o-mini",
                "api_key_env": "OPENAI_API_KEY",
                "max_tokens": 4096, 
                "temperature": 0.7,
            }
        }
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url_env": "ANTHROPIC_BASE_URL",
        "models": {
            "claude-opus-4-1-20250805": {
                "model": "anthropic/claude-opus-4-1-20250805",
                "api_key_env": "ANTHROPIC_API_KEY",
                "max_tokens": 4096,
                "temperature": 0.7,
            },
            "claude-sonnet-4-20250514": {
                "model": "anthropic/claude-sonnet-4-20250514",
                "api_key_env": "ANTHROPIC_API_KEY",
                "max_tokens": 4096,
                "temperature": 0.7,
            },
            "claude-3-5-sonnet-20241022": {
                "model": "anthropic/claude-3-5-sonnet-20241022",
                "api_key_env": "ANTHROPIC_API_KEY",
                "max_tokens": 4096,
                "temperature": 0.7,
            }
        }
    },
    "cerebras": {
        "api_key_env": "CEREBRAS_API_KEY", 
        "base_url_env": "CEREBRAS_BASE_URL",
        "models": {
            "llama3.1-8b": {
                "model": "cerebras/llama3.1-8b",
                "api_key_env": "CEREBRAS_API_KEY",
                "max_tokens": 4096,
                "temperature": 0.7,
            },
            "qwen-3-235b-a22b-thinking-2507": {
                "model": "cerebras/qwen-3-235b-a22b-thinking-2507",
                "api_key_env": "CEREBRAS_API_KEY",
                "reasoning_effort": "high",
                "max_tokens": 32000,
                "temperature": 0.7,
            },
            "qwen-3-32b": {
                "model": "cerebras/qwen-3-32b",
                "api_key_env": "CEREBRAS_API_KEY",
                "max_tokens": 32000,
                "temperature": 0.7,
            }
        }
    },
    "google": {
        "api_key_env": "GOOGLE_API_KEY",
        "base_url_env": "GOOGLE_BASE_URL",
        "models": {
            "gemini-1.5-pro-002": {
                "model": "google/gemini-1.5-pro-002",
                "api_key_env": "GOOGLE_API_KEY",
                "max_tokens": 4096,
                "temperature": 0.7,
            },
            "gemini-2.0-flash-thinking-exp-1219": {
                "model": "google/gemini-2.0-flash-thinking-exp-1219",
                "api_key_env": "GOOGLE_API_KEY",
                "max_tokens": 4096,
                "temperature": 0.7,
            }
        }
    },
    "custom": {
        "api_key_env": "CUSTOM_API_KEY",
        "base_url_env": "CUSTOM_BASE_URL", 
        "models": {
            "custom-llm-1": {
                "model": "custom/custom-llm-1",
                "api_key_env": "CUSTOM_API_KEY",
                "api_base_env": "CUSTOM_API_BASE",
                "max_tokens": 4096,
                "temperature": 0.7,
            },
            "vllm-hosted": {
                "model": "vllm/hosted-model",
                "api_key_env": "VLLM_API_KEY", 
                "base_url_env": "VLLM_BASE_URL",
                "max_tokens": 4096,
                "temperature": 0.7,
            }
        }
    }
}


def get_provider_config(provider: str) -> dict[str, Any]:
    """Get configuration for a specific provider."""
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDER_REGISTRY.keys())}")
    return PROVIDER_REGISTRY[provider]


def get_model_config(provider: str, model: str) -> dict[str, Any]:
    """Get configuration for a specific provider/model combination."""
    provider_config = get_provider_config(provider)
    if model not in provider_config["models"]:
        available = list(provider_config["models"].keys())
        raise ValueError(f"Unknown model '{model}' for provider '{provider}'. Available: {available}")
    
    # Merge provider-level and model-level configs
    config = {
        "provider": provider,
        "api_key_env": provider_config["api_key_env"],
        "base_url_env": provider_config.get("base_url_env"),
        **provider_config["models"][model]
    }
    return config


def list_providers() -> list[str]:
    """List all available providers."""
    return list(PROVIDER_REGISTRY.keys())


def list_models(provider: str) -> list[str]:
    """List all available models for a provider."""
    provider_config = get_provider_config(provider)
    return list(provider_config["models"].keys())


# Centralized default model configurations
# Used by both DSPy and direct LLM engines for consistent defaults
DEFAULT_MODEL_CONFIGS = {
    "fast": ("cerebras", "llama3.1-8b"),                    # Fastest inference, good quality
    "balanced": ("openai", "gpt-4o"),                       # Balance of speed, quality, cost
    "powerful": ("anthropic", "claude-sonnet-4-20250514"),   # Highest quality reasoning
    "economical": ("openai", "gpt-4o-mini"),                # Most cost-effective
    "reasoning": ("cerebras", "qwen-3-235b-a22b-thinking-2507")  # Advanced reasoning capabilities
}


def get_default_config(config_name: str = "balanced") -> tuple[str, str]:
    """Get a default provider/model combination by name."""
    if config_name not in DEFAULT_MODEL_CONFIGS:
        available = list(DEFAULT_MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return DEFAULT_MODEL_CONFIGS[config_name]


def list_default_configs() -> list[str]:
    """List all available default configuration names."""
    return list(DEFAULT_MODEL_CONFIGS.keys())


def validate_provider_setup(provider: str, model: str) -> dict[str, Any]:
    """
    Validate that a provider/model combination is properly configured.
    
    Returns:
        Dict with validation results and any missing environment variables
    """
    try:
        config = get_model_config(provider, model)
        
        # Check environment variables
        api_key = os.getenv(config["api_key_env"])
        base_url = os.getenv(config["base_url_env"]) if config.get("base_url_env") else None
        
        return {
            "valid": True,
            "provider": provider,
            "model": model, 
            "config": config,
            "api_key_available": bool(api_key),
            "base_url_available": bool(base_url) if config.get("base_url_env") else True,
            "missing_env_vars": [
                var for var in [config["api_key_env"], config.get("base_url_env")]
                if var and not os.getenv(var)
            ]
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "provider": provider,
            "model": model
        }