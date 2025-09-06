"""
Direct LLM Configuration Utilities

Centralized configuration for direct LLM calls (non-DSPy) using the provider registry.
Separates LLM configuration concerns from agent implementation logic.
"""

import os
from typing import Any

from ..llm import get_model_config, list_providers, list_models, get_default_config, list_default_configs

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


def create_llm_config(provider: str, model_name: str, **kwargs) -> dict[str, Any]:
    """Create LLM configuration dict for direct LLM calls."""
    config = get_model_config(provider, model_name)
    
    # Handle environment variable resolution
    if "api_key_env" in config:
        api_key_env = config["api_key_env"]
        api_key = os.getenv(api_key_env)
        if api_key:
            config["api_key"] = api_key
        else:
            raise ValueError(f"API key not found in environment: {api_key_env}")
    
    # Handle base URL environment variables
    if "base_url_env" in config:
        base_url_env = config["base_url_env"]
        base_url = os.getenv(base_url_env)
        if base_url:
            config["base_url"] = base_url
    
    # Override with any additional kwargs
    config.update(kwargs)
    
    return config


def call_llm(
    messages: list[dict[str, str]], 
    provider: str, 
    model: str, 
    **kwargs
) -> Any:
    """Make a direct LLM call using LiteLLM."""
    if not LITELLM_AVAILABLE:
        raise ImportError("LiteLLM not available. Install with: pip install litellm")
    
    config = create_llm_config(provider, model, **kwargs)
    
    return litellm.completion(
        model=config["model"],
        messages=messages,
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        max_tokens=config.get("max_tokens", 4096),
        temperature=config.get("temperature", 0.7),
        **{k: v for k, v in config.items() 
           if k not in ["model", "api_key", "base_url", "max_tokens", "temperature", "api_key_env", "base_url_env", "provider"]}
    )


def simple_chat(
    user_message: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    **kwargs
) -> str:
    """Simple chat function for quick LLM interactions."""
    if not LITELLM_AVAILABLE:
        raise ImportError("LiteLLM not available. Install with: pip install litellm")
    
    messages = [{"role": "user", "content": user_message}]
    response = call_llm(messages, provider, model, **kwargs)
    return response.choices[0].message.content


def get_response_text(response: Any) -> str:
    """Extract text from LLM response."""
    return response.choices[0].message.content


def validate_llm_setup(provider: str, model: str) -> dict[str, Any]:
    """Validate that direct LLM calls can be made with the specified provider/model."""
    from ..llm import validate_provider_setup
    validation = validate_provider_setup(provider, model)
    
    if not LITELLM_AVAILABLE:
        validation["litellm_available"] = False
        validation["warning"] = "LiteLLM not installed - direct LLM calls will fail"
    else:
        validation["litellm_available"] = True
    
    return validation


# Pre-configured functions for common providers
def create_openai_config(model: str = "gpt-4o-mini", **kwargs) -> dict[str, Any]:
    """Create OpenAI configuration."""
    return create_llm_config("openai", model, **kwargs)


def create_anthropic_config(model: str = "claude-sonnet-4-20250514", **kwargs) -> dict[str, Any]:
    """Create Anthropic configuration."""
    return create_llm_config("anthropic", model, **kwargs)


def create_cerebras_config(model: str = "llama3.1-8b", **kwargs) -> dict[str, Any]:
    """Create Cerebras configuration."""
    return create_llm_config("cerebras", model, **kwargs)


def create_google_config(model: str = "gemini-1.5-pro-002", **kwargs) -> dict[str, Any]:
    """Create Google configuration."""
    return create_llm_config("google", model, **kwargs)


def create_default_config(config_name: str = "balanced", **kwargs) -> dict[str, Any]:
    """Create configuration with a named default."""
    provider, model = get_default_config(config_name)
    return create_llm_config(provider, model, **kwargs)


def quick_chat(
    user_message: str,
    config_name: str = "balanced",
    **kwargs
) -> str:
    """Quick chat using a named default configuration."""
    provider, model = get_default_config(config_name)
    return simple_chat(user_message, provider, model, **kwargs)