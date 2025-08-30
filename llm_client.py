"""
Thin Wrapper for LiteLLM Completion Calls

This module provides simple helper functions for agents to make litellm.completion() calls
with provider-specific defaults and parameter formatting. The goal is to make it easy
for agents to switch between providers while using the direct LiteLLM interface.
"""

import os
import litellm
from keywell import setup_keywell_handler


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
    "keywell": {
        "model": os.getenv("KEYWELL_MODEL_ID", ""),
        "api_base": os.getenv("KEYWELL_ENDPOINT", ""),
        "api_key": os.getenv("KEYWELL_PAT_TOKEN", ""),
        "sid": os.getenv("KEYWELL_SID", "")
    }
}


def _ensure_keywell_handler():
    """Ensure Keywell custom handler is registered with LiteLLM."""
    try:
        setup_keywell_handler()
    except Exception as e:
        print(f"Warning: Could not setup Keywell handler: {e}")


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


def keywell_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    sid: str | None = None,
    session_id: str | None = None,
    **kwargs
) -> any:
    """
    Keywell completion using direct litellm.completion() call.
    
    Parameter mapping:
        - api_base ← url_endpoint (Keywell Endpoint)
        - api_key ← pat_token (Keywell PAT Token)
        - model ← mydbx/{model_id} (Keywell model for both session init and responses)
        - optional_params.sid ← sid (Keywell SID)
        - optional_params.session_id ← session_id (generated from session initialization)
    
    Args:
        messages: OpenAI format messages
        model: Keywell model ID (required via KEYWELL_MODEL_ID env var)
        api_base: Keywell endpoint URL (url_endpoint)
        api_key: Keywell PAT token (pat_token)
        sid: Keywell SID (goes in optional_params)
        session_id: Session ID from initialization (goes in optional_params)
        **kwargs: Additional litellm.completion parameters
        
    Returns:
        LiteLLM completion response
    """
    # Ensure Keywell handler is registered
    _ensure_keywell_handler()
    
    effective_model = model or DEFAULTS["keywell"]["model"]
    effective_api_base = api_base or DEFAULTS["keywell"]["api_base"]
    effective_api_key = api_key or DEFAULTS["keywell"]["api_key"]
    effective_sid = sid or DEFAULTS["keywell"]["sid"]
    
    # Format model for mydbx provider if needed
    if not effective_model.startswith("mydbx/"):
        effective_model = f"mydbx/{effective_model}"
    
    # Build optional_params
    optional_params = {
        "sid": effective_sid,
        **(kwargs.get("optional_params", {}))
    }
    
    if session_id:
        optional_params["session_id"] = session_id
    
    return litellm.completion(
        model=effective_model,
        messages=messages,
        api_base=effective_api_base,
        api_key=effective_api_key,
        optional_params=optional_params,
        timeout=kwargs.get("timeout", 60),
        **{k: v for k, v in kwargs.items() if k not in ["timeout", "optional_params"]}
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
        provider: Provider name ("openai", "anthropic", "gemini", "keywell")
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
    elif provider == "keywell":
        return keywell_completion(messages, **kwargs)
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