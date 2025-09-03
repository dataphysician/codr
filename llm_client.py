"""
Thin Wrapper for LiteLLM Completion Calls

This module provides simple helper functions for agents to make litellm.completion() calls
with provider-specific defaults and parameter formatting. The goal is to make it easy
for agents to switch between providers while using the direct LiteLLM interface.
"""

import os
import litellm


# Provider defaults - can be overridden via environment variables or function parameters
DEFAULTS = {
    "openai": {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "api_key": os.getenv("OPENAI_API_KEY", "")
    },
    "anthropic": {
        "model": os.getenv("ANTHROPIC_MODEL", "claude-opus-4-1-20250805"), 
        "api_key": os.getenv("ANTHROPIC_API_KEY", "")
    },
    "gemini": {
        "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
        "api_key": os.getenv("GEMINI_API_KEY", "")
    },
    "cerebras": {
        "model": os.getenv("CEREBRAS_MODEL", "qwen-3-32b"),
        "api_key": os.getenv("CEREBRAS_API_KEY", "")
    },
}


def openai_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    api_key: str | None = None,
    **kwargs
) -> any:
    """
    OpenAI completion using direct litellm.completion() call.
    
    Args:
        messages: OpenAI format messages
        model: Model name (default: gpt-4o-mini)  
        api_key: OpenAI API key
        **kwargs: Additional litellm.completion parameters
        
    Returns:
        LiteLLM completion response
    """
    effective_model = model or DEFAULTS["openai"]["model"]
    effective_api_key = api_key or DEFAULTS["openai"]["api_key"]
    
    # Format model for OpenAI provider if needed
    if not effective_model.startswith("openai/"):
        effective_model = f"openai/{effective_model}"
    
    return litellm.completion(
        model=effective_model,
        messages=messages,
        api_key=effective_api_key,
        timeout=kwargs.get("timeout", 60),
        **{k: v for k, v in kwargs.items() if k != "timeout"}
    )


def anthropic_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    api_key: str | None = None,
    **kwargs
) -> any:
    """
    Anthropic completion using direct litellm.completion() call.
    
    Args:
        messages: OpenAI format messages
        model: Model name (default: claude-3-5-sonnet-20241022)
        api_key: Anthropic API key
        **kwargs: Additional litellm.completion parameters
        
    Returns:
        LiteLLM completion response
    """
    effective_model = model or DEFAULTS["anthropic"]["model"]
    effective_api_key = api_key or DEFAULTS["anthropic"]["api_key"]
    
    # Format model for Anthropic provider if needed
    if not effective_model.startswith("anthropic/"):
        effective_model = f"anthropic/{effective_model}"
    
    return litellm.completion(
        model=effective_model,
        messages=messages,
        api_key=effective_api_key,
        timeout=kwargs.get("timeout", 60),
        **{k: v for k, v in kwargs.items() if k != "timeout"}
    )


def gemini_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    api_key: str | None = None,
    **kwargs
) -> any:
    """
    Gemini completion using direct litellm.completion() call.
    
    Args:
        messages: OpenAI format messages
        model: Model name (default: gemini-2.0-flash-exp)
        api_key: Google API key
        **kwargs: Additional litellm.completion parameters
        
    Returns:
        LiteLLM completion response
    """
    effective_model = model or DEFAULTS["gemini"]["model"]
    effective_api_key = api_key or DEFAULTS["gemini"]["api_key"]
    
    # Format model for Gemini provider if needed
    if not effective_model.startswith("gemini/"):
        effective_model = f"gemini/{effective_model}"
    
    return litellm.completion(
        model=effective_model,
        messages=messages,
        api_key=effective_api_key,
        timeout=kwargs.get("timeout", 60),
        **{k: v for k, v in kwargs.items() if k != "timeout"}
    )


def cerebras_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    api_key: str | None = None,
    **kwargs
) -> any:
    """
    Cerebras completion using direct litellm.completion() call.
    
    Args:
        messages: OpenAI format messages
        model: Model name (default: qwen-3-235b-a22b-thinking-2507)
        api_key: Cerebras API key
        **kwargs: Additional litellm.completion parameters
        
    Returns:
        LiteLLM completion response
    """
    effective_model = model or DEFAULTS["cerebras"]["model"]
    effective_api_key = api_key or DEFAULTS["cerebras"]["api_key"]
    
    # Format model for Cerebras provider if needed
    if not effective_model.startswith("cerebras/"):
        effective_model = f"cerebras/{effective_model}"
    
    return litellm.completion(
        model=effective_model,
        messages=messages,
        api_key=effective_api_key,
        timeout=kwargs.get("timeout", 60),
        **{k: v for k, v in kwargs.items() if k != "timeout"}
    )


def completion(
    messages: list[dict[str, str]],
    provider: str = "openai",
    **kwargs
) -> any:
    """
    Generic completion function that routes to the appropriate provider.
    
    Args:
        messages: OpenAI format messages
        provider: Provider name ("openai", "anthropic", "gemini", "cerebras")
        **kwargs: Provider-specific parameters
        
    Returns:
        LiteLLM completion response
    """
    if provider == "openai":
        return openai_completion(messages, **kwargs)
    elif provider == "anthropic":
        return anthropic_completion(messages, **kwargs)
    elif provider == "gemini":
        return gemini_completion(messages, **kwargs)
    elif provider == "cerebras":
        return cerebras_completion(messages, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_response_text(response: any) -> str:
    """Extract text from any LiteLLM response."""
    return response.choices[0].message.content


# Convenience function for the most common use case
def chat(
    user_message: str,
    provider: str = "openai",
    **kwargs
) -> str:
    """
    Simple chat function for quick completions.
    
    Args:
        user_message: User's message
        provider: Provider to use
        **kwargs: Provider-specific parameters
        
    Returns:
        Response text
    """
    messages = [{"role": "user", "content": user_message}]
    response = completion(messages, provider=provider, **kwargs)
    return get_response_text(response)