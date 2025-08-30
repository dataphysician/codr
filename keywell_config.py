"""
Keywell Configuration Settings

This module contains configuration settings for Keywell LLM integration.
Values can be set here as defaults or overridden via environment variables.
"""

import os

class KeywellConfig:
    """Configuration class for Keywell API settings."""
    
    def __init__(self):
        # Default values - can be overridden by environment variables or at runtime
        self.url_endpoint: str = os.getenv("KEYWELL_ENDPOINT", "")
        self.pat_token: str = os.getenv("KEYWELL_PAT_TOKEN", "")
        self.model_id: str = os.getenv("KEYWELL_MODEL_ID", "")
        self.sid: str = os.getenv("KEYWELL_SID", "")
        
        # Session management
        self._session_id: str | None = None
        self.debug: bool = os.getenv("KEYWELL_DEBUG", "false").lower() == "true"
    
    def update(self, **kwargs) -> 'KeywellConfig':
        """Update configuration values at runtime."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        return self
    
    def validate(self) -> bool:
        """Validate that all required configuration values are set."""
        required_fields = [
            'url_endpoint', 'pat_token', 'model_id', 'sid'
        ]
        
        missing_fields = [field for field in required_fields if not getattr(self, field)]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")
        
        return True
    
    @property
    def session_id(self) -> str | None:
        """Get the current session ID."""
        return self._session_id
    
    @session_id.setter
    def session_id(self, value: str):
        """Set the session ID."""
        self._session_id = value

# Global configuration instance
config = KeywellConfig()

def set_keywell_config(**kwargs) -> KeywellConfig:
    """Convenience function to update global configuration."""
    return config.update(**kwargs)

def get_keywell_config() -> KeywellConfig:
    """Get the global configuration instance."""
    return config