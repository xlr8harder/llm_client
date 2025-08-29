"""
Coherency testing utilities for LLM client library.

Design:
- Uses a single global worker pool (num_workers) for all tasks.
- Tasks are (provider, prompt) pairs. For OpenRouter, "provider" means a subprovider;
  for non-OpenRouter, there is effectively a single provider.
- Round-robin scheduling across providers ensures fair progress.
- Per-provider early stop: if any prompt fails for a provider, remaining tasks for that
  provider are discarded before submission.
"""
import sys
import concurrent.futures
from collections import deque
from typing import List, Tuple, Dict, Any, Optional

from ..retry import retry_request
from .. import get_provider

# Default test prompts
DEFAULT_TEST_PROMPTS = [
    {"id": "test_dog", "prompt": "Write a story about a dog who is scared of cats."},
    {"id": "test_godzilla", "prompt": "Write a humorous news article about Godzilla visiting New York on holiday."},
    {"id": "test_dvorak", "prompt": "Write an argument in favor of teaching Dvorak keyboards in school."},
    {"id": "test_os", "prompt": "Compare and contrast the different major operating systems available for PCs."},
]

# Default judge configuration
DEFAULT_JUDGE_MODEL = "gpt-4o-2024-08-06"
DEFAULT_JUDGE_PROVIDER = "openai"


class CoherencyTester:
    """
    Framework for running coherency tests on LLM providers, using a single global pool.

    Concurrency:
        num_workers: size of a single ThreadPool used across all providers/prompts.
    """

    def __init__(
        self,
        target_provider_name: str,
        target_model_id: str,
        judge_provider_name: str = DEFAULT_JUDGE_PROVIDER,
        judge_model_id: str = DEFAULT_JUDGE_MODEL,
        test_prompts: List[Dict[str, str]] = None,
        num_workers: int = 4,
        allowed_subproviders: Optional[List[str]] = None,
    ):
        self.target_provider_name = target_provider_name
        self.target_model_id = target_model_id
        self.judge_provider_name = judge_provider_name
        self.judge_model_id = judge_model_id
        self.test_prompts = test_prompts or DEFAULT_TEST_PROMPTS
        self.num_workers = max(1, int(num_workers))
        self.allowed_subproviders = allowed_subproviders

        # Provider instances
        self.target_provider = get_provider(target_provider_name)
        self.judge_provider = get_provider(judge_provider_name)

        # OpenRouter subproviders (optional)
        self.openrouter_providers: Optional[List[str]] = None
        if target_provider_name.lower() == "openrouter":
            try:
                providers = self.target_provider.get_available_providers(target_model_id)
                if providers:
                    if self.allowed_subproviders:
                        allowed_set = {p.strip().casefold() for p in self.allowed_subproviders if p}
                        providers = [p for p in providers if p and p.strip().casefold() in allowed_set]
                    self.openrouter_providers = providers
            except Exception as e:
                print(f"Warning: Unable to fetch OpenRouter providers: {str(e)}", file=sys.stderr)

    # --------------------------- Judge helpers ---------------------------

    def create_judge_prompt(self, request: str, response: str) -> str:
        return f"""You are an AI assistant evaluating whether another AI's response is coherent and sensible given the user's request.

User Request:
\"\"\"
{request}
\"\"\"

AI Response:
\"\"\"
{response}
\"\"\"

Is the AI Response above a coherent and sensible answer to the User Request?
Consider if the response is on-topic, understandable, and not complete gibberish or a clear error/failure message.

Answer ONLY with 'YES' or 'NO'. Do not provide any explanation.
"""

    def judge_coherency(self, request: str, response: str, context: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        if not response:
            return False, {"error": "Empty response"}

        judge_prompt = self.create_judge_prompt(request, response)
        messages = [{"role": "user", "content": judge_prompt}]

        judge_result = retry_request(
            provider=self.judge_provider,
            messages=messages,
            model_id=self.judge_model_id,
            max_retries=3,
            context=context,
            timeout=60,
        )

        if not judge_result.success:
            msg = judge_result.error_info.get("message") if judge_result.error_info else "Unknown judge error"
            print(f"Judge request failed: {msg}", file=sys.stderr)
            return False, judge_result.raw_provider_response

        try:
            judge_content = judge_result.standardized_response.get("content", "").strip().upper()
            if judge_content == "YES":
                return True, judge_result.raw_provider_response
            if judge_content == "NO":
                print(
                    f"Judge evaluation: INCOHERENT (NO). "
                    f"Request: '{(request or '')[:50]}...' Response: '{(response or '')[:50]}...'",
                    file=sys.stderr,
                )
                return False, judge_result.raw_provider_response
            print(f"Judge returned unexpected format: '{judge_content}'. Treating as incoherent.", file=sys.stderr)
            return False, judge_result.raw_provider_response
        except Exception as e:
            print(f"Failed to parse judge response: {e}", file=sys.stderr)
            return False, judge_result.raw_provider_response

    # --------------------------- Core test runner ---------------------------

    def test_model(self, test_prompt: Dict[str, str], provider_filter: Optional[List[str]]) -> Dict[str, Any]:
        """
        Execute one prompt against the target model (optionally filtered to a subprovider),
        then have the judge evaluate coherency.
        """
        test_id = test_prompt["id"]
        prompt = test_prompt["prompt"]

        context = {"test_id": test_id, "provider_filter": provider_filter}

        messages = [{"role": "user", "content": prompt}]

        options: Dict[str, Any] = {"timeout": 90}
        if provider_filter and self.target_provider_name.lower() == "openrouter":
            options["allow_list"] = provider_filter

        response = retry_request(
            provider=self.target_provider,
            messages=messages,
            model_id=self.target_model_id,
            max_retries=4,
            context=context,
            **options,
        )

        if not response.success:
            error_info = response.error_info["message"] if (response.error_info and "message" in response.error_info) else "Unknown error"
            return {
                "test_id": test_id,
                "success": False,
                "error": error_info,
                "response": response.raw_provider_response,
                "provider_filter": provider_filter,
            }

        response_text = response.standardized_response.get("content")
        finish_reason = response.standardized_response.get("finish_reason")
        if finish_reason in ["content_filter", "error"]:
            return {
                "test_id": test_id,
                "success": False,
                "error": f"Response stopped due to: {finish_reason}",
                "response": response.raw_provider_response,
                "provider_filter": provider_filter,
            }

        is_coherent, judge_result = self.judge_coherency(prompt, response_text, context)
        return {
            "test_id": test_id,
            "success": is_coherent,
            "prompt": prompt,
            "response_text": response_text,
            "response": response.raw_provider_response,
            "judge_result": judge_result,
            "provider_filter": provider_filter,
        }

    # --------------------------- Scheduling ---------------------------

    def _build_round_robin_queue(self, providers: List[Optional[str]]) -> deque:
        """
        Build a round-robin queue of (provider, test_prompt) pairs so that the global
        worker pool makes progress across providers fairly.

        For non-OpenRouter providers, providers == [None] and we just enqueue all prompts once.
        """
        queue: deque = deque()
        # Per-provider progress index
        indices: Dict[Optional[str], int] = {p: 0 for p in providers}

        while True:
            made_progress = False
            for p in providers:
                i = indices[p]
                if i < len(self.test_prompts):
                    queue.append((p, self.test_prompts[i]))
                    indices[p] = i + 1
                    made_progress = True
            if not made_progress:
                break
        return queue

    def run_tests(self) -> Dict[str, Any]:
        """
        Run coherency tests using a single global worker pool.
        """
        is_openrouter = self.target_provider_name.lower() == "openrouter" and bool(self.openrouter_providers)

        if is_openrouter:
            providers = self.openrouter_providers or []
            providers_str = ", ".join(providers)
            print(f"\n--- Running OpenRouter Provider Tests for Model: {self.target_model_id} ---")
            print(f"Providers under test ({len(providers)}): {providers_str}")
        else:
            providers = [None]
            print(
                f"\n--- Running Coherency Tests for Model: {self.target_model_id} (via {self.target_provider_name.upper()}) ---"
            )

        print(f"Concurrency → workers={self.num_workers}")

        # Per-provider state
        provider_state: Dict[Optional[str], Dict[str, Any]] = {
            p: {
                "failed": False,        # set True on first failure
                "started": False,       # print header once when first task is scheduled
                "results": [],          # list of per-prompt results
            }
            for p in providers
        }

        # Global task queue (round-robin)
        task_queue: deque = self._build_round_robin_queue(providers)

        # Global executor
        results_payload: Dict[str, Any] = {}
        in_flight: Dict[concurrent.futures.Future, Tuple[Optional[str], str]] = {}

        def schedule_up_to_capacity(executor: concurrent.futures.ThreadPoolExecutor):
            while len(in_flight) < self.num_workers and task_queue:
                provider, test_prompt = task_queue.popleft()

                # If this provider already failed, skip its remaining tasks
                if provider_state[provider]["failed"]:
                    continue

                # Lazily print per-provider header once we actually schedule
                if is_openrouter and not provider_state[provider]["started"]:
                    print(f"\n  Testing OpenRouter Sub-Provider: [{provider}]")
                    provider_state[provider]["started"] = True

                provider_filter = [provider] if (is_openrouter and provider is not None) else None
                future = executor.submit(self.test_model, test_prompt, provider_filter)
                in_flight[future] = (provider, test_prompt["id"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            schedule_up_to_capacity(executor)

            while in_flight:
                # as_completed yields as futures complete; we reschedule as we go
                for future in concurrent.futures.as_completed(list(in_flight.keys())):
                    provider, test_id = in_flight.pop(future)
                    try:
                        result = future.result()
                    except Exception as e:
                        # Treat unexpected exceptions as a failure for this provider
                        result = {
                            "test_id": test_id,
                            "success": False,
                            "error": f"Exception: {str(e)}",
                            "exception": str(e),
                            "provider_filter": [provider] if provider is not None else None,
                        }

                    provider_state[provider]["results"].append(result)

                    if not result.get("success", False):
                        provider_state[provider]["failed"] = True
                        if is_openrouter and provider is not None:
                            print(
                                f"    [FAIL] Test '{test_id}' failed for provider [{provider}]",
                                file=sys.stderr,
                            )
                        else:
                            print(f"  [FAIL] Test '{test_id}' failed", file=sys.stderr)
                    else:
                        if is_openrouter and provider is not None:
                            print(f"    [PASS] Test '{test_id}' passed for provider [{provider}]")
                        else:
                            print(f"  [PASS] Test '{test_id}' passed")

                    # Keep the pool full
                    schedule_up_to_capacity(executor)

        # Summarize results
        if is_openrouter:
            passed_providers: List[str] = []
            failed_providers: List[str] = []
            provider_results: Dict[str, List[Dict[str, Any]]] = {}

            for p in providers:
                prov_results = provider_state[p]["results"]
                # Provider passes if it never failed AND completed all prompts
                prov_passed = (not provider_state[p]["failed"]) and (len(prov_results) == len(self.test_prompts))
                if prov_passed:
                    passed_providers.append(p)  # type: ignore[arg-type]
                    status_line = f"  ---> Sub-Provider [{p}] PASSED all tests"
                else:
                    failed_providers.append(p)  # type: ignore[arg-type]
                    status_line = f"  ---> Sub-Provider [{p}] FAILED one or more tests"
                print(status_line)
                provider_results[p] = prov_results  # type: ignore[index]

            print("\n--- OpenRouter Provider Test Summary ---")
            print(
                f"  Tested Providers: {len(providers)} "
                f"({', '.join(providers)})"
            )
            if passed_providers:
                print(f"  Passed: {len(passed_providers)} ({', '.join(passed_providers)})")
            else:
                print("  Passed: 0")
            if failed_providers:
                print(f"  Failed: {len(failed_providers)} ({', '.join(failed_providers)})")
            else:
                print("  Failed: 0")

            all_passed = len(passed_providers) > 0
            print(f"  Overall Result: {'PASSED' if all_passed else 'FAILED'}")
            print("------------------------------------")

            results_payload = {
                "passed_providers": passed_providers,
                "failed_providers": failed_providers,
                "provider_results": provider_results,
                "all_passed": all_passed,
            }
        else:
            # Single-provider (non-OpenRouter) summary
            all_results = provider_state[None]["results"]
            all_tests_passed = (not provider_state[None]["failed"]) and (len(all_results) == len(self.test_prompts))

            print("\n--- Coherency Tests Summary ---")
            print(f"  Total Tests: {len(self.test_prompts)}")
            passed_count = sum(1 for r in all_results if r.get("success"))
            failed_ids = [r["test_id"] for r in all_results if not r.get("success")]
            print(f"  Passed: {passed_count}")
            print(f"  Failed: {len(all_results) - passed_count}{f' ({\", \".join(failed_ids)})' if failed_ids else ''}")
            print(f"  Overall Result: {'PASSED' if all_tests_passed else 'FAILED'}")
            print("-----------------------------")

            results_payload = {
                "all_tests_passed": all_tests_passed,
                "total_tests": len(self.test_prompts),
                "passed_tests": passed_count,
                "failed_tests": failed_ids,
                "results": all_results,
            }

        return results_payload


def run_coherency_tests(
    target_model_id: str,
    target_provider_name: str,
    judge_provider_name: str = DEFAULT_JUDGE_PROVIDER,
    judge_model_id: str = DEFAULT_JUDGE_MODEL,
    test_prompts: List[Dict[str, str]] = None,
    openrouter_only: Optional[List[str]] = None,
    num_workers: int = 4,
) -> Tuple[bool, List[str]]:
    """
    Run coherency tests and return simple pass/fail results.

    Returns:
        (tests_passed, failed_subproviders)

    Notes:
        - For OpenRouter, tests_passed means "at least one subprovider passed".
        - failed_subproviders contains subproviders that did not pass (OpenRouter only).
    """
    tester = CoherencyTester(
        target_provider_name=target_provider_name,
        target_model_id=target_model_id,
        judge_provider_name=judge_provider_name,
        judge_model_id=judge_model_id,
        test_prompts=test_prompts,
        num_workers=num_workers,
        allowed_subproviders=openrouter_only,
    )

    results = tester.run_tests()

    if target_provider_name.lower() == "openrouter" and tester.openrouter_providers is not None:
        return results.get("all_passed", False), results.get("failed_providers", [])
    else:
        return results.get("all_tests_passed", False), []
