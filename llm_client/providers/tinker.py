"""
Tinker provider implementation for the llm_client library.

This provider routes `make_chat_completion_request(...)` to the Tinker Sampling API
(i.e. inference happens via the Tinker service, configured by the `tinker` client).

Model addressing:
  - Recommended: "<base_model>::<renderer>::<tinker_path>"
      e.g. "Qwen/Qwen3-30B-A3B::qwen3::tinker://..."
  - Also supported: "<base_model>::<tinker_path>" (renderer defaults to options or 'role_colon')
  - Also supported: "<tinker_path>" with `base_model=...` in options
"""

import time
from typing import Any, Dict, Optional, Tuple

from ..base import LLMProvider, LLMResponse


class TinkerProvider(LLMProvider):
    """Provider implementation for Tinker Sampling API."""

    provider_name = "tinker"

    def _get_api_key_env_var(self) -> str:
        # Tinker auth and configuration is handled by the `tinker` client itself
        # (e.g. via its own env vars). llm_client does not require an API key here.
        return "TINKER_API_KEY"

    def get_api_key(self):
        # Override to avoid forcing any environment variable for local usage.
        return None

    def make_chat_completion_request(self, messages, model_id, context=None, **options) -> LLMResponse:
        try:
            tinker, renderers, get_tokenizer = self._lazy_imports()

            timeout = options.pop("timeout", None)
            base_url = options.pop("base_url", None) or options.pop("tinker_base_url", None)

            base_model, renderer_name, tinker_path = self._parse_model_id(model_id=model_id, options=options)

            tokenizer = get_tokenizer(base_model)
            renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

            prompt = renderer.build_generation_prompt(messages)
            stop = options.pop("stop", None)
            if stop is None:
                stop = renderer.get_stop_sequences()

            max_tokens = int(options.pop("max_tokens", 4096))
            temperature = float(options.pop("temperature", 0.7))
            top_p = float(options.pop("top_p", 0.95))
            top_k = int(options.pop("top_k", -1))

            sampling_params = tinker.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
            )

            service_client = tinker.ServiceClient(base_url=base_url) if base_url else tinker.ServiceClient()
            if tinker_path:
                sampling_client = service_client.create_sampling_client(
                    model_path=tinker_path,
                    base_model=base_model,
                )
            else:
                sampling_client = service_client.create_sampling_client(base_model=base_model)

            future = sampling_client.sample(prompt=prompt, sampling_params=sampling_params, num_samples=1)
            result = future.result(timeout=timeout) if timeout else future.result()

            sequences = getattr(result, "sequences", None) or []
            if not sequences:
                return LLMResponse(
                    success=False,
                    error_info={"type": "provider_error", "message": "Tinker sampling returned no sequences"},
                    raw_provider_response={"result": repr(result)},
                    is_retryable=True,
                    context=context,
                )

            seq0 = sequences[0]
            tokens = getattr(seq0, "tokens", None) or []

            assistant_message, parse_success = renderer.parse_response(tokens)
            if isinstance(assistant_message, dict):
                content = assistant_message.get("content", "")
            else:
                content = str(assistant_message)

            prompt_tokens = None
            try:
                prompt_tokens = len(prompt.to_ints())
            except Exception:
                prompt_tokens = None

            completion_tokens = len(tokens)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (prompt_tokens or 0) + completion_tokens,
            }

            standardized = {
                "id": None,
                "created": int(time.time()),
                "model": model_id,
                "provider": self.provider_name,
                "content": content,
                "usage": usage,
                "finish_reason": "stop" if parse_success else "length",
            }

            return LLMResponse(
                success=True,
                standardized_response=standardized,
                raw_provider_response={
                    "sequences": [{"tokens": tokens}],
                    "tinker": {"base_model": base_model, "model_path": tinker_path, "renderer": renderer_name},
                },
                is_retryable=False,
                context=context,
            )
        except Exception as e:
            return LLMResponse(
                success=False,
                error_info={"type": "provider_error", "message": str(e), "exception": str(e)},
                raw_provider_response=None,
                is_retryable=False,
                context=context,
            )

    def _lazy_imports(self):
        try:
            import tinker  # type: ignore
        except Exception as e:
            raise ImportError(
                "Missing dependency: 'tinker'. Install it to use llm_client's 'tinker' provider."
            ) from e

        try:
            from tinker_cookbook import renderers  # type: ignore
            from tinker_cookbook.tokenizer_utils import get_tokenizer  # type: ignore
        except Exception as e:
            raise ImportError(
                "Missing dependency: 'tinker_cookbook'. Install it to use llm_client's 'tinker' provider."
            ) from e

        # SamplingParams is present on modern tinker versions; rely on it directly.
        if not hasattr(tinker, "SamplingParams"):
            raise ImportError("Installed 'tinker' package does not expose SamplingParams")

        return tinker, renderers, get_tokenizer

    def _parse_model_id(self, model_id: str, options: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
        default_renderer = options.pop("renderer", None) or options.pop("renderer_name", None) or "role_colon"

        if "::" in model_id:
            parts = model_id.split("::")
            if len(parts) == 2:
                base_model, tinker_path = parts
                return base_model, str(default_renderer), tinker_path
            if len(parts) == 3:
                base_model, renderer_name, tinker_path = parts
                return base_model, renderer_name, tinker_path
            raise ValueError("Invalid tinker model_id. Expected '<base>::<path>' or '<base>::<renderer>::<path>'")

        # If the model_id itself is a tinker path, base_model must be provided in options.
        if model_id.startswith("tinker://"):
            base_model = options.pop("base_model", None) or options.pop("base_model_name", None)
            if not base_model:
                raise ValueError("For model_id like 'tinker://...', pass base_model='...' in options")
            return str(base_model), str(default_renderer), model_id

        # Otherwise treat model_id as a base model name and sample it directly.
        return model_id, str(default_renderer), None
