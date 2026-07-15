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
    GoogleAgentPlatformProvider,
    TNGTechProvider,
    XAIProvider,
    MoonshotProvider,
    StepfunProvider,
    TinkerProvider,
    LocalProvider,
)

# Convenient mapping of provider names to their implementations
PROVIDER_MAP = {
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
    "fireworks": FireworksProvider,
    "chutes": ChutesProvider,
    "google": GoogleProvider,
    "google_agent_platform": GoogleAgentPlatformProvider,
    "tngtech": TNGTechProvider,
    "xai": XAIProvider,
    "moonshot": MoonshotProvider,
    "stepfun": StepfunProvider,
    "tinker": TinkerProvider,
    "local": LocalProvider,
    "openai_compatible": LocalProvider,
}

__version__ = "0.2.1"

# V2 is additive. Legacy imports and behavior above remain unchanged.
from .v2_builder import ConversationBuilder
from .v2_client import (
    AsyncClient,
    Client,
    Model,
    UnknownProviderFieldWarning,
    generate,
    reset_unknown_field_warnings,
)
from .v2_models import (
    Conversation,
    ConversationBusyError,
    ErrorInfo,
    Message,
    ModelResponse,
    ProviderState,
    ReplyOperation,
    RequestAttempt,
    UnboundConversationError,
    WireRecord,
)
from .oauth import (
    OAuthConfig,
    OAuthCredentialStore,
    OAuthCredentials,
    OAuthError,
    OAuthLoginRequest,
    OAuthManager,
)
from .codex_oauth import CodexOAuthManager, codex_oauth_config


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
        raise ValueError(
            f"Unknown provider: '{provider_name}'. Valid providers are: {valid_providers}"
        )

    return PROVIDER_MAP[provider_name.lower()]()


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "retry_request",
    "get_provider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "FireworksProvider",
    "ChutesProvider",
    "GoogleProvider",
    "GoogleAgentPlatformProvider",
    "TNGTechProvider",
    "XAIProvider",
    "MoonshotProvider",
    "StepfunProvider",
    "TinkerProvider",
    "LocalProvider",
    "AsyncClient",
    "Client",
    "Conversation",
    "ConversationBuilder",
    "ConversationBusyError",
    "ErrorInfo",
    "Message",
    "Model",
    "ModelResponse",
    "ProviderState",
    "ReplyOperation",
    "RequestAttempt",
    "UnknownProviderFieldWarning",
    "UnboundConversationError",
    "WireRecord",
    "generate",
    "reset_unknown_field_warnings",
    "OAuthConfig",
    "OAuthCredentialStore",
    "OAuthCredentials",
    "OAuthError",
    "OAuthLoginRequest",
    "OAuthManager",
    "CodexOAuthManager",
    "codex_oauth_config",
]
