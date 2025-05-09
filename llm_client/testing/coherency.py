"""
Coherency testing utilities for LLM client library.
"""
import sys
import concurrent.futures
from typing import List, Tuple, Dict, Any, Optional

from ..base import LLMResponse
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
    Framework for running coherency tests on LLM providers.
    """
    
    def __init__(self, 
                 target_provider_name: str,
                 target_model_id: str,
                 judge_provider_name: str = DEFAULT_JUDGE_PROVIDER,
                 judge_model_id: str = DEFAULT_JUDGE_MODEL,
                 test_prompts: List[Dict[str, str]] = None,
                 num_workers: int = 4):
        """
        Initialize the coherency tester.
        
        Args:
            target_provider_name: Name of the provider to test
            target_model_id: Model ID to test
            judge_provider_name: Provider to use for judging coherency
            judge_model_id: Model ID to use for judging
            test_prompts: List of test prompts (dicts with 'id' and 'prompt' keys)
            num_workers: Number of concurrent workers for testing
        """
        self.target_provider_name = target_provider_name
        self.target_model_id = target_model_id
        self.judge_provider_name = judge_provider_name
        self.judge_model_id = judge_model_id
        self.test_prompts = test_prompts or DEFAULT_TEST_PROMPTS
        self.num_workers = num_workers
        
        # Create provider instances
        self.target_provider = get_provider(target_provider_name)
        self.judge_provider = get_provider(judge_provider_name)
        
        # For OpenRouter, we'll track providers
        self.openrouter_providers = None
        if target_provider_name.lower() == "openrouter":
            try:
                self.openrouter_providers = self.target_provider.get_available_providers(target_model_id)
            except Exception as e:
                print(f"Warning: Unable to fetch OpenRouter providers: {str(e)}", file=sys.stderr)
    
    def create_judge_prompt(self, request: str, response: str) -> str:
        """
        Create a prompt for the judging model to evaluate coherency.
        
        Args:
            request: The original request text
            response: The response to evaluate
            
        Returns:
            Prompt for the judge
        """
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
        """
        Use the judge to evaluate if a response is coherent.
        
        Args:
            request: The original request text
            response: The response to evaluate
            context: Optional context for request tracking
            
        Returns:
            Tuple (is_coherent, judge_result)
        """
        # Handle empty or None responses
        if not response:
            return False, {"error": "Empty response"}
            
        # Create the judge prompt
        judge_prompt = self.create_judge_prompt(request, response)
        
        # Prepare messages for the judge
        messages = [{"role": "user", "content": judge_prompt}]
        
        # Make the judge request with retries
        judge_result = retry_request(
            provider=self.judge_provider,
            messages=messages,
            model_id=self.judge_model_id,
            max_retries=3,
            context=context,
            timeout=60
        )
        
        # Check for success
        if not judge_result.success:
            error_info = "Unknown error"
            if judge_result.error_info and "message" in judge_result.error_info:
                error_info = judge_result.error_info["message"]
            print(f"Judge request failed: {error_info}", file=sys.stderr)
            return False, judge_result.raw_provider_response
            
        # Extract the judge's verdict
        try:
            judge_content = judge_result.standardized_response.get("content", "").strip().upper()
            if judge_content == "YES":
                return True, judge_result.raw_provider_response
            elif judge_content == "NO":
                print(f"Judge evaluation: INCOHERENT (NO). Request: '{request[:50]}...' Response: '{response[:50]}...'", file=sys.stderr)
                return False, judge_result.raw_provider_response
            else:
                print(f"Judge returned unexpected format: '{judge_content}'. Treating as incoherent.", file=sys.stderr)
                return False, judge_result.raw_provider_response
        except (KeyError, IndexError, AttributeError, TypeError) as e:
            print(f"Failed to parse judge response: {e}", file=sys.stderr)
            return False, judge_result.raw_provider_response
    
    def test_model(self, test_prompt: Dict[str, str], provider_filter: List[str] = None) -> Dict[str, Any]:
        """
        Test a single prompt with the target model.
        
        Args:
            test_prompt: Dictionary with 'id' and 'prompt' keys
            provider_filter: For OpenRouter, list of providers to test with
            
        Returns:
            Dictionary with test results
        """
        test_id = test_prompt["id"]
        prompt = test_prompt["prompt"]
        
        # Prepare context for tracking
        context = {
            "test_id": test_id,
            "provider_filter": provider_filter
        }
        
        # Prepare messages for the target model
        messages = [{"role": "user", "content": prompt}]
        
        # Prepare provider-specific options
        options = {"timeout": 90}
        if provider_filter and self.target_provider_name.lower() == "openrouter":
            if len(provider_filter) == 1:
                options["allow_list"] = provider_filter
            else:
                options["ignore_list"] = provider_filter
        
        # Make the target model request with retries
        response = retry_request(
            provider=self.target_provider,
            messages=messages,
            model_id=self.target_model_id,
            max_retries=4,
            context=context,
            **options
        )
        
        # Check for success
        if not response.success:
            error_info = "Unknown error"
            if response.error_info and "message" in response.error_info:
                error_info = response.error_info["message"]
            
            return {
                "test_id": test_id,
                "success": False,
                "error": error_info,
                "response": response.raw_provider_response,
                "provider_filter": provider_filter
            }
        
        # Extract the response content
        response_text = response.standardized_response.get("content")
        
        # Check for content filter or other issues
        finish_reason = response.standardized_response.get("finish_reason")
        if finish_reason in ["content_filter", "error"]:
            return {
                "test_id": test_id,
                "success": False,
                "error": f"Response stopped due to: {finish_reason}",
                "response": response.raw_provider_response,
                "provider_filter": provider_filter
            }
        
        # Judge the coherency
        is_coherent, judge_result = self.judge_coherency(prompt, response_text, context)
        
        return {
            "test_id": test_id,
            "success": is_coherent,
            "prompt": prompt,
            "response_text": response_text,
            "response": response.raw_provider_response,
            "judge_result": judge_result,
            "provider_filter": provider_filter
        }
    
    def run_tests(self) -> Dict[str, Any]:
        """
        Run all coherency tests, potentially in parallel.
        
        Returns:
            Dictionary with test results
        """
        # OpenRouter-specific handling
        if self.target_provider_name.lower() == "openrouter" and self.openrouter_providers:
            return self._run_openrouter_tests()
        else:
            return self._run_standard_tests()
    
    def _run_standard_tests(self) -> Dict[str, Any]:
        """Run tests for non-OpenRouter providers or OpenRouter without provider filtering"""
        results = []
        failed_tests = []
        all_tests_passed = True
        
        print(f"\n--- Running Coherency Tests for Model: {self.target_model_id} (via {self.target_provider_name.upper()}) ---")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all test prompts for processing
            future_to_test = {
                executor.submit(self.test_model, test_prompt): test_prompt
                for test_prompt in self.test_prompts
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_test):
                test_prompt = future_to_test[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    test_id = test_prompt["id"]
                    if not result.get("success", False):
                        all_tests_passed = False
                        failed_tests.append(test_id)
                        print(f"  [FAIL] Test '{test_id}' failed", file=sys.stderr)
                    else:
                        print(f"  [PASS] Test '{test_id}' passed")
                        
                except Exception as e:
                    all_tests_passed = False
                    test_id = test_prompt["id"]
                    failed_tests.append(test_id)
                    print(f"  [ERROR] Test '{test_id}' raised exception: {str(e)}", file=sys.stderr)
                    results.append({
                        "test_id": test_id,
                        "success": False,
                        "error": f"Exception: {str(e)}",
                        "exception": str(e)
                    })
        
        # Summarize results
        print("\n--- Coherency Tests Summary ---")
        print(f"  Total Tests: {len(self.test_prompts)}")
        print(f"  Passed: {len(self.test_prompts) - len(failed_tests)}")
        print(f"  Failed: {len(failed_tests)} {f'({", ".join(failed_tests)})' if failed_tests else ''}")
        print(f"  Overall Result: {'PASSED' if all_tests_passed else 'FAILED'}")
        print("-----------------------------")
        
        return {
            "all_tests_passed": all_tests_passed,
            "total_tests": len(self.test_prompts),
            "passed_tests": len(self.test_prompts) - len(failed_tests),
            "failed_tests": failed_tests,
            "results": results
        }
    
    def _run_openrouter_tests(self) -> Dict[str, Any]:
        """Run tests for OpenRouter with provider filtering"""
        overall_results = {
            "passed_providers": [],
            "failed_providers": [],
            "provider_results": {},
            "all_passed": False
        }
        
        print(f"\n--- Running OpenRouter Provider Tests for Model: {self.target_model_id} ---")
        print(f"Available providers: {', '.join(self.openrouter_providers)}")
        
        # Test each provider individually
        for provider in self.openrouter_providers:
            print(f"\n  Testing OpenRouter Sub-Provider: [{provider}]")
            provider_passed = True
            provider_results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all test prompts for this provider
                future_to_test = {
                    executor.submit(self.test_model, test_prompt, [provider]): test_prompt
                    for test_prompt in self.test_prompts
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_test):
                    test_prompt = future_to_test[future]
                    try:
                        result = future.result()
                        provider_results.append(result)
                        
                        test_id = test_prompt["id"]
                        if not result.get("success", False):
                            provider_passed = False
                            print(f"    [FAIL] Test '{test_id}' failed for provider [{provider}]", file=sys.stderr)
                            # Stop testing this provider if one test fails
                            break
                        else:
                            print(f"    [PASS] Test '{test_id}' passed for provider [{provider}]")
                            
                    except Exception as e:
                        provider_passed = False
                        test_id = test_prompt["id"]
                        print(f"    [ERROR] Test '{test_id}' raised exception for provider [{provider}]: {str(e)}", file=sys.stderr)
                        provider_results.append({
                            "test_id": test_id,
                            "success": False,
                            "error": f"Exception: {str(e)}",
                            "exception": str(e),
                            "provider": provider
                        })
                        # Stop testing this provider if one test fails
                        break
            
            # Store results for this provider
            if provider_passed:
                print(f"  ---> Sub-Provider [{provider}] PASSED all tests")
                overall_results["passed_providers"].append(provider)
            else:
                print(f"  ---> Sub-Provider [{provider}] FAILED one or more tests")
                overall_results["failed_providers"].append(provider)
                
            overall_results["provider_results"][provider] = provider_results
        
        # Summarize results
        print("\n--- OpenRouter Provider Test Summary ---")
        print(f"  Tested Providers: {len(self.openrouter_providers)} ({', '.join(self.openrouter_providers)})")
        print(f"  Passed: {len(overall_results['passed_providers'])} ({', '.join(overall_results['passed_providers'])})")
        print(f"  Failed: {len(overall_results['failed_providers'])} ({', '.join(overall_results['failed_providers'])})")
        
        # Overall success if at least one provider passed
        overall_results["all_passed"] = len(overall_results["passed_providers"]) > 0
        print(f"  Overall Result: {'PASSED' if overall_results['all_passed'] else 'FAILED'}")
        print("------------------------------------")
        
        return overall_results


def run_coherency_tests(
    target_model_id: str,
    target_provider_name: str,
    judge_provider_name: str = DEFAULT_JUDGE_PROVIDER,
    judge_model_id: str = DEFAULT_JUDGE_MODEL,
    test_prompts: List[Dict[str, str]] = None,
    num_workers: int = 4
) -> Tuple[bool, List[str]]:
    """
    Run coherency tests and return simple pass/fail results.
    
    Args:
        target_model_id: Model ID to test
        target_provider_name: Name of the provider to test
        judge_provider_name: Provider to use for judging coherency
        judge_model_id: Model ID to use for judging
        test_prompts: List of test prompts (dicts with 'id' and 'prompt' keys)
        num_workers: Number of concurrent workers for testing
        
    Returns:
        Tuple (overall_success, failed_providers_list)
    """
    tester = CoherencyTester(
        target_provider_name=target_provider_name,
        target_model_id=target_model_id,
        judge_provider_name=judge_provider_name,
        judge_model_id=judge_model_id,
        test_prompts=test_prompts,
        num_workers=num_workers
    )
    
    results = tester.run_tests()
    
    # For OpenRouter, return the list of failed providers
    if target_provider_name.lower() == "openrouter" and tester.openrouter_providers:
        return results["all_passed"], results.get("failed_providers", [])
    else:
        return results["all_tests_passed"], []
