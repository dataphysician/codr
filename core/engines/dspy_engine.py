"""
DSPy Configuration Utilities

Centralized DSPy LLM configuration using the provider registry.
Separates DSPy setup concerns from agent implementation logic.
"""

import os
import dspy
from typing import Any

from ..llm import get_model_config, list_providers, list_models, get_default_config, list_default_configs


def create_dspy_lm(provider: str, model_name: str, **kwargs) -> dspy.LM:
    """Create a DSPy LM instance for the specified provider and model."""
    config = get_model_config(provider, model_name)
    
    # Handle environment variable resolution for DSPy
    if "api_key_env" in config:
        api_key_env = config.pop("api_key_env")
        api_key = os.getenv(api_key_env)
        if api_key:
            config["api_key"] = api_key
        else:
            print(f"Warning: Environment variable {api_key_env} not set")
    
    # Handle base URL environment variables
    if "api_base_env" in config:
        api_base_env = config.pop("api_base_env")
        api_base = os.getenv(api_base_env)
        if api_base:
            config["api_base"] = api_base
    
    if "base_url_env" in config:
        base_url_env = config.pop("base_url_env")
        base_url = os.getenv(base_url_env)
        if base_url:
            config["base_url"] = base_url
    
    # Handle Qwen models that need JSON object response format
    if "qwen" in model_name.lower():
        print(f"🔧 Detected Qwen model: {model_name}, using JSON object response format")
        config["response_format"] = {"type": "json_object"}
        # Remove parameters not supported by Cerebras/Qwen or litellm
        unsupported_params = ["reasoning_effort", "provider"]
        for param in unsupported_params:
            if param in config:
                removed_value = config.pop(param)
                print(f"🔧 Removed unsupported parameter for Qwen: {param}={removed_value}")
    
    # Disable caching by default for testing to ensure actual model calls
    if "cache" not in config:
        config["cache"] = False
    if "cache_in_memory" not in config:
        config["cache_in_memory"] = False
    
    # Override with any additional kwargs, but filter out environment variable keys
    filtered_kwargs = {k: v for k, v in kwargs.items() if not k.endswith("_env")}
    config.update(filtered_kwargs)
    
    # Debug: Log max_tokens being passed to DSPy
    if "max_tokens" in config:
        print(f"🔧 DSPy LM will be created with max_tokens: {config['max_tokens']}")
    
    return dspy.LM(**config)


def configure_dspy(provider: str, model_name: str, **kwargs) -> dspy.LM:
    """Configure DSPy with the specified provider and model, return the LM instance."""
    lm = create_dspy_lm(provider, model_name, **kwargs)
    dspy.configure(lm=lm)
    return lm


def quick_configure(provider: str, model_name: str, **kwargs) -> dspy.LM:
    """Quickly configure DSPy with a provider/model combination."""
    try:
        lm = configure_dspy(provider, model_name, **kwargs)
        print(f"✅ DSPy configured with {provider}/{model_name}")
        return lm
    except Exception as e:
        print(f"❌ Failed to configure DSPy: {e}")
        raise


# Pre-configured LM factory functions for common use cases
def create_openai_lm(model: str = "gpt-4o", **kwargs) -> dspy.LM:
    """Create OpenAI model LM instance."""
    return create_dspy_lm("openai", model, **kwargs)


def create_anthropic_lm(model: str = "claude-sonnet-4-20250514", **kwargs) -> dspy.LM:
    """Create Anthropic model LM instance."""
    return create_dspy_lm("anthropic", model, **kwargs)


def create_cerebras_lm(model: str = "llama3.1-8b", **kwargs) -> dspy.LM:
    """Create Cerebras model LM instance.""" 
    return create_dspy_lm("cerebras", model, **kwargs)


def create_google_lm(model: str = "gemini-1.5-pro-002", **kwargs) -> dspy.LM:
    """Create Google model LM instance."""
    return create_dspy_lm("google", model, **kwargs)


def get_available_providers() -> list[str]:
    """Get list of available providers for DSPy configuration."""
    return list_providers()


def get_available_models(provider: str) -> list[str]:
    """Get list of available models for a provider."""
    return list_models(provider)


def validate_dspy_setup(provider: str, model: str) -> dict[str, Any]:
    """Validate that DSPy can be configured with the specified provider/model."""
    from ..llm import validate_provider_setup
    return validate_provider_setup(provider, model)


def configure_default(config_name: str = "balanced", **kwargs) -> dspy.LM:
    """Configure DSPy with a named default configuration."""
    provider, model = get_default_config(config_name)
    return configure_dspy(provider, model, **kwargs)