"""
LLM client library for interacting with various LLM API providers.
"""

from .base import LLMProvider, LLMResponse
from .retry import retry_request
from .providers import (
    OpenAIProvider,
    OpenRouterProvider,
    FireworksProvider,
    ChutesProvider,
    GoogleProvider,
    TNGTechProvider,
    XAIProvider,
    MoonshotProvider,
    TinkerProvider,
)

# Convenient mapping of provider names to their implementations
PROVIDER_MAP = {
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
    "fireworks": FireworksProvider,
    "chutes": ChutesProvider,
    "google": GoogleProvider,
    "tngtech": TNGTechProvider,
    "xai": XAIProvider,
    "moonshot": MoonshotProvider,
    "tinker": TinkerProvider,
}

__version__ = "0.1.18"

def get_provider(provider_name):
    """
    Get a provider instance by name.
    
    Args:
        provider_name: Name of the provider to instantiate
        
    Returns:
        Instance of the appropriate LLMProvider subclass
        
    Raises:
        ValueError: If the provider_name is not recognized
    """
    if provider_name.lower() not in PROVIDER_MAP:
        valid_providers = ", ".join(PROVIDER_MAP.keys())
        raise ValueError(f"Unknown provider: '{provider_name}'. Valid providers are: {valid_providers}")
    
    return PROVIDER_MAP[provider_name.lower()]()

__all__ = [
    'LLMProvider',
    'LLMResponse',
    'retry_request',
    'get_provider',
    'OpenAIProvider',
    'OpenRouterProvider',
    'FireworksProvider',
    'ChutesProvider',
    'GoogleProvider',
    'TNGTechProvider',
    'XAIProvider',
    'MoonshotProvider',
    'TinkerProvider',
]
