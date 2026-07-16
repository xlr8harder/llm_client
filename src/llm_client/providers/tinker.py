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

import os
import time
import inspect
import threading
from typing import Any, Dict, Optional, Tuple

from ..base import LLMProvider, LLMResponse, with_finish_reason_metadata


class TinkerProvider(LLMProvider):
    """Provider implementation for Tinker Sampling API."""

    provider_name = "tinker"
    _EFFORT_ALIASES = {
        "none": 0.0,
        "minimal": 0.1,
        "low": 0.2,
        "medium": 0.7,
        "high": 0.9,
        "xhigh": 0.99,
        "max": 0.99,
    }
    _client_lock = threading.Lock()
    _service_clients: Dict[Tuple[Optional[str]], Any] = {}
    _sampling_clients: Dict[Tuple[Optional[str], str, Optional[str]], Any] = {}

    def __init__(self):
        super().__init__()
        # Hugging Face tokenizers can warn (and potentially deadlock) when a process
        # forks after tokenizers parallelism has been used. Default to disabling
        # tokenizers parallelism for the Tinker provider unless the user explicitly
        # configured it.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def _get_api_key_env_var(self) -> str:
        # Tinker auth and configuration is handled by the `tinker` client itself
        # (e.g. via its own env vars). llm_client does not require an API key here.
        return "TINKER_API_KEY"

    def get_api_key(self):
        # Override to avoid forcing any environment variable for local usage.
        return None

    def make_chat_completion_request(
        self, messages, model_id, context=None, **options
    ) -> LLMResponse:
        try:
            tinker, renderers, get_tokenizer = self._lazy_imports()

            timeout = options.pop("timeout", None)
            base_url = options.pop("base_url", None) or options.pop(
                "tinker_base_url", None
            )

            base_model, renderer_name, tinker_path = self._parse_model_id(
                model_id=model_id, options=options
            )

            tokenizer = get_tokenizer(base_model)
            renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

            prompt_kwargs = self._renderer_prompt_kwargs(renderer, renderer_name, options)
            prompt = renderer.build_generation_prompt(messages, **prompt_kwargs)
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

            sampling_client = self._get_sampling_client(
                tinker=tinker,
                base_url=base_url,
                base_model=base_model,
                tinker_path=tinker_path,
            )

            future = sampling_client.sample(
                prompt=prompt, sampling_params=sampling_params, num_samples=1
            )
            result = future.result(timeout=timeout) if timeout else future.result()

            sequences = getattr(result, "sequences", None) or []
            if not sequences:
                return LLMResponse(
                    success=False,
                    error_info={
                        "type": "provider_error",
                        "message": "Tinker sampling returned no sequences",
                    },
                    raw_provider_response={"result": repr(result)},
                    is_retryable=True,
                    context=context,
                )

            seq0 = sequences[0]
            tokens = getattr(seq0, "tokens", None) or []

            assistant_message, parse_termination = renderer.parse_response(tokens)
            parse_finish_reason, parse_observed = self._normalize_parse_termination(
                parse_termination=parse_termination,
                completion_tokens=len(tokens),
                max_tokens=max_tokens,
            )
            raw_response = {
                "sequences": [{"tokens": tokens}],
                "tinker": {
                    "base_model": base_model,
                    "model_path": tinker_path,
                    "renderer": renderer_name,
                    "parse_termination": parse_observed,
                    "renderer_prompt_kwargs": prompt_kwargs,
                },
            }
            if parse_finish_reason is None:
                return LLMResponse(
                    success=False,
                    error_info={
                        "type": "provider_error",
                        "message": (
                            "Tinker renderer returned an unclassified parse "
                            f"termination: {parse_observed!r}"
                        ),
                        "parse_termination": parse_observed,
                    },
                    raw_provider_response=raw_response,
                    is_retryable=False,
                    context=context,
                )

            if isinstance(assistant_message, dict):
                content = assistant_message.get("content", "")
            else:
                content = str(assistant_message)

            # Extract thinking from structured content for reasoning display
            thinking = None
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "thinking":
                        thinking = item.get("thinking", "")
                        break

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
            }
            with_finish_reason_metadata(
                standardized,
                source="renderer.parse_response[1]",
                value=parse_observed,
                normalized=parse_finish_reason,
            )
            if thinking:
                raw_response["thinking"] = thinking

            return LLMResponse(
                success=True,
                standardized_response=standardized,
                raw_provider_response=raw_response,
                is_retryable=False,
                context=context,
            )
        except Exception as e:
            return LLMResponse(
                success=False,
                error_info={
                    "type": "provider_error",
                    "message": str(e),
                    "exception": str(e),
                },
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
            raise ImportError(
                "Installed 'tinker' package does not expose SamplingParams"
            )

        return tinker, renderers, get_tokenizer

    def _get_sampling_client(
        self,
        *,
        tinker: Any,
        base_url: Optional[str],
        base_model: str,
        tinker_path: Optional[str],
    ) -> Any:
        """Reuse Tinker sessions; the service rejects excessive active clients."""
        service_key = (base_url,)
        sampling_key = (base_url, base_model, tinker_path)

        with self._client_lock:
            sampling_client = self._sampling_clients.get(sampling_key)
            if sampling_client is not None:
                return sampling_client

            service_client = self._service_clients.get(service_key)
            if service_client is None:
                service_client = (
                    tinker.ServiceClient(base_url=base_url)
                    if base_url
                    else tinker.ServiceClient()
                )
                self._service_clients[service_key] = service_client

            if tinker_path:
                sampling_client = service_client.create_sampling_client(
                    model_path=tinker_path,
                    base_model=base_model,
                )
            else:
                sampling_client = service_client.create_sampling_client(
                    base_model=base_model
                )
            self._sampling_clients[sampling_key] = sampling_client
            return sampling_client

    @classmethod
    def _clear_client_cache_for_tests(cls) -> None:
        with cls._client_lock:
            cls._service_clients.clear()
            cls._sampling_clients.clear()

    def _parse_model_id(
        self, model_id: str, options: Dict[str, Any]
    ) -> Tuple[str, str, Optional[str]]:
        default_renderer = (
            options.pop("renderer", None)
            or options.pop("renderer_name", None)
            or "role_colon"
        )

        if "::" in model_id:
            parts = model_id.split("::")
            if len(parts) == 2:
                base_model, tinker_path = parts
                return base_model, str(default_renderer), tinker_path
            if len(parts) == 3:
                base_model, renderer_name, tinker_path = parts
                return base_model, renderer_name, tinker_path
            raise ValueError(
                "Invalid tinker model_id. Expected '<base>::<path>' or '<base>::<renderer>::<path>'"
            )

        # If the model_id itself is a tinker path, base_model must be provided in options.
        if model_id.startswith("tinker://"):
            base_model = options.pop("base_model", None) or options.pop(
                "base_model_name", None
            )
            if not base_model:
                raise ValueError(
                    "For model_id like 'tinker://...', pass base_model='...' in options"
                )
            return str(base_model), str(default_renderer), model_id

        # Otherwise treat model_id as a base model name and sample it directly.
        return model_id, str(default_renderer), None

    def _renderer_prompt_kwargs(
        self, renderer: Any, renderer_name: str, options: Dict[str, Any]
    ) -> Dict[str, Any]:
        effort = self._extract_effort(options)
        if effort is None:
            return {}

        signature = inspect.signature(renderer.build_generation_prompt)
        if "effort" not in signature.parameters:
            raise ValueError(
                f"Tinker renderer '{renderer_name}' does not support reasoning effort"
            )
        return {"effort": effort}

    def _extract_effort(self, options: Dict[str, Any]) -> Optional[float]:
        effort = None
        if "effort" in options:
            effort = options.pop("effort")
        if "reasoning_effort" in options:
            if effort is not None:
                raise ValueError("Specify only one of effort or reasoning_effort")
            effort = options.pop("reasoning_effort")

        reasoning = options.pop("reasoning", None)
        if reasoning is None:
            return self._coerce_effort(effort) if effort is not None else None
        if not isinstance(reasoning, dict):
            raise ValueError("Tinker reasoning option must be an object")

        enabled = reasoning.get("enabled")
        reasoning_effort = reasoning.get("effort")
        reasoning_tokens = reasoning.get("max_tokens")

        if enabled is False:
            if effort is not None and self._coerce_effort(effort) != 0.0:
                raise ValueError("Cannot set non-zero effort when reasoning is disabled")
            if reasoning_effort is not None and self._coerce_effort(reasoning_effort) != 0.0:
                raise ValueError("Cannot set non-zero reasoning.effort when reasoning is disabled")
            return 0.0

        if reasoning_tokens is not None:
            raise ValueError(
                "Tinker reasoning uses renderer effort, not reasoning.max_tokens"
            )
        if reasoning_effort is not None:
            if effort is not None:
                raise ValueError("Specify only one of effort or reasoning.effort")
            effort = reasoning_effort

        return self._coerce_effort(effort) if effort is not None else None

    def _coerce_effort(self, effort: Any) -> float:
        if isinstance(effort, bool):
            return 0.9 if effort else 0.0
        if isinstance(effort, str):
            normalized = effort.strip().lower()
            if normalized in self._EFFORT_ALIASES:
                return self._EFFORT_ALIASES[normalized]
            effort = normalized
        value = float(effort)
        if not 0.0 <= value < 1.0:
            raise ValueError("Tinker effort must be in the range [0.0, 1.0)")
        return value

    def _normalize_parse_termination(
        self,
        *,
        parse_termination: Any,
        completion_tokens: int,
        max_tokens: int,
    ) -> Tuple[Optional[str], Optional[str]]:
        if isinstance(parse_termination, bool):
            return ("stop" if parse_termination else "length"), str(parse_termination)

        observed = self._parse_termination_value(parse_termination)
        if observed is None:
            return None, None

        normalized = observed.strip().lower()
        if normalized in {"stop", "stop_sequence", "eos", "end_turn"}:
            return "stop", observed
        if normalized in {"length", "max_tokens", "max_token", "truncated"}:
            return "length", observed
        if normalized == "malformed":
            if completion_tokens >= max_tokens:
                return "length", observed
            return None, observed

        return None, observed

    def _parse_termination_value(self, parse_termination: Any) -> Optional[str]:
        if parse_termination is None:
            return None
        value = getattr(parse_termination, "value", None)
        if value is not None:
            return str(value)
        name = getattr(parse_termination, "name", None)
        if name is not None:
            return str(name)
        return str(parse_termination)
