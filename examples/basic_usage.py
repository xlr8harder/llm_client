#!/usr/bin/env python3
"""
Basic usage example for the LLM client library.
"""
import os
import sys
import json

# Add the parent directory to sys.path to import the library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_client import get_provider, retry_request
from llm_client.testing import run_coherency_tests

def make_simple_request(provider_name, model_id, prompt):
    """
    Make a simple request to an LLM provider.
    
    Args:
        provider_name: Name of the provider to use
        model_id: Model ID to use
        prompt: Prompt text to send
        
    Returns:
        Response text or error message
    """
    # Get the provider instance
    provider = get_provider(provider_name)
    
    # Format the message
    messages = [{"role": "user", "content": prompt}]
    
    # Make the request with retry logic
    response = retry_request(
        provider=provider,
        messages=messages,
        model_id=model_id,
        max_retries=3,
        timeout=60
    )
    
    # Check for success
    if response.success:
        # Access the content from the standardized response
        return response.standardized_response.get("content", "No content returned")
    else:
        # Handle error
        error_info = "Unknown error"
        if response.error_info and "message" in response.error_info:
            error_info = response.error_info["message"]
        return f"Error: {error_info}"

def test_provider_coherency(provider_name, model_id):
    """
    Run coherency tests for a provider/model combination.
    
    Args:
        provider_name: Name of the provider to test
        model_id: Model ID to test
        
    Returns:
        None
    """
    # Run the coherency tests
    success, failed_providers = run_coherency_tests(
        target_model_id=model_id,
        target_provider_name=provider_name,
        num_workers=2  # Use 2 workers for demonstration
    )
    
    # Report results
    if success:
        print(f"Coherency tests PASSED for {provider_name}/{model_id}")
        if failed_providers:
            print(f"However, these OpenRouter providers failed: {', '.join(failed_providers)}")
    else:
        print(f"Coherency tests FAILED for {provider_name}/{model_id}")
        
def main():
    """Main function to demonstrate library usage"""
    # Simple request example
    
    # Choose provider and model - change these as needed
    provider_name = "openai"  # or "openrouter", "fireworks", "chutes", "google"
    
    if provider_name == "openai":
        model_id = "gpt-4o-2024-08-06"
    elif provider_name == "openrouter":
        model_id = "openai/gpt-4o-2024-08-06"
    elif provider_name == "fireworks":
        model_id = "accounts/fireworks/models/llama-v3-70b-instruct"
    elif provider_name == "chutes":
        model_id = "mistral-7b-instruct-v0.2"
    elif provider_name == "google":
        model_id = "gemini-1.5-pro"
    else:
        print(f"Unknown provider: {provider_name}")
        return
    
    # Make a simple request
    prompt = "Explain quantum computing in simple terms."
    print(f"\nMaking request to {provider_name}/{model_id}...\n")
    response_text = make_simple_request(provider_name, model_id, prompt)
    print(f"Response from {provider_name}/{model_id}:\n")
    print(response_text)
    print("\n" + "-" * 50 + "\n")
    
    # Run coherency tests (optional)
    run_tests = input("Would you like to run coherency tests? (y/n): ").lower().strip() == 'y'
    if run_tests:
        test_provider_coherency(provider_name, model_id)

if __name__ == "__main__":
    main()
