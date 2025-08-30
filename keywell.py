"""
Keywell LiteLLM Custom Handler Configuration

This module configures the Keywell custom handler for LiteLLM integration.
It handles session management and provides the custom handler for Databricks models.
"""

import requests
import litellm
from litellm import CustomLLM
from keywell_config import get_keywell_config, KeywellConfig


def initialize_keywell_session(config: KeywellConfig) -> str:
    """Initialize a new Keywell session and return session ID."""
    config.validate()
    
    auth_payload = {
        "inputs": {
            "question": ["Initialize New Session"],
            "model_id": config.model_id,
            "sid": config.sid
        }
    }
    
    try:
        auth_response = requests.post(
            config.url_endpoint,
            headers={
                "Authorization": f"Bearer {config.pat_token}",
                "Content-Type": "application/json"
            },
            json=auth_payload,
            timeout=60
        )
        auth_response.raise_for_status()
        
        session_id = auth_response.json()["predictions"].get("session_id")
        if not session_id:
            raise ValueError("Failed to get session_id from Keywell API response")
        
        config.session_id = session_id
        
        if config.debug:
            print(f"[Keywell] Initialized new session: {session_id}")
        
        return session_id
        
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to initialize Keywell session: {e}")


class DatabricksPyfuncLLM(CustomLLM):
    """
    LiteLLM custom handler for Keywell Databricks models.
    Handles session management and API communication.
    """

    def __init__(self, config: KeywellConfig | None = None):
        self.config = config or get_keywell_config()
        self._sessions: dict[tuple[str, str], str] = {}  # (sid, model_id) -> session_id

    def _extract_user_content(self, messages: list[dict[str, any]]) -> str:
        """Extract the last user message content from OpenAI message format."""
        for message in reversed(messages or []):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    def _resolve_model_id(self, model: str) -> str:
        """Extract the real Databricks model_id from the model string."""
        return model.split("/", 1)[1] if "/" in model else model

    def completion(
        self,
        *,
        model: str,
        messages: list[dict[str, any]],
        api_base: str | None = None,
        api_key: str | None = None,
        optional_params: dict[str, any] | None = None,
        timeout: int | None = None,
        **kwargs
    ) -> any:
        """Complete chat using Keywell API with OpenAI-compatible interface."""
        # Merge parameters
        params = {**(optional_params or {}), **(kwargs or {})}
        
        # Use provided values or fall back to config
        endpoint = api_base or self.config.url_endpoint
        token = api_key or self.config.pat_token
        
        if not endpoint or not token:
            raise ValueError("Missing required API endpoint or token")
        
        model_id = self._resolve_model_id(model)
        sid = params.get("sid") or self.config.sid
        
        # Session management: explicit > cached > default > new
        session_id = (
            params.get("session_id") or
            self._sessions.get((sid, model_id)) or
            self.config.session_id
        )
        
        # Initialize session if needed
        if not session_id:
            session_id = initialize_keywell_session(self.config)
        
        # Extract user question from messages
        question = self._extract_user_content(messages)
        
        # Build request payload
        inputs = {
            "question": [question],
            "model_id": model_id,
            "sid": sid,
            "session_id": session_id
        }

        url = endpoint.rstrip("/")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        if self.config.debug:
            print(f"[Keywell] POST {url}")
            print(f"[Keywell] Model: {model_id}, SID: {sid}")
            print(f"[Keywell] Question: {question[:100]}...")

        try:
            response = requests.post(
                url,
                headers=headers,
                json={"inputs": inputs},
                timeout=timeout or 60
            )
            response.raise_for_status()
            
            data = response.json() or {}
            predictions = data.get("predictions", {}) or {}
            response_text = predictions.get("ResponseText", "") or ""
            new_session_id = predictions.get("session_id")
            
            # Cache new session ID for future requests
            if new_session_id:
                self._sessions[(sid, model_id)] = new_session_id
            
            if self.config.debug:
                print(f"[Keywell] Response: {response_text[:200]}...")
            
            # Return OpenAI-shaped response using mock_response
            return litellm.completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": ""}],
                mock_response=response_text
            )
            
        except requests.RequestException as e:
            raise RuntimeError(f"Keywell API request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in Keywell completion: {e}")


def setup_keywell_handler(config: KeywellConfig | None = None) -> None:
    """
    Set up and register the Keywell custom handler with LiteLLM.
    Call this once during application initialization.
    """
    effective_config = config or get_keywell_config()
    
    # Create and register the handler
    handler = DatabricksPyfuncLLM(effective_config)
    litellm.custom_provider_map = [
        {"provider": "mydbx", "custom_handler": handler}
    ]
    
    if effective_config.debug:
        print("[Keywell] Custom handler registered with LiteLLM")