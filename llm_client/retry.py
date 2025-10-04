"""Retry logic for LLM API requests."""
import logging
import random
import threading
import time


def retry_request(
    provider,
    messages,
    model_id,
    max_retries=5,
    initial_delay=1,
    backoff_factor=2,
    jitter=0.1,
    context=None,
    verbose=False,
    logger=None,
    **options,
):
    """
    Higher-level function that handles retries
    
    Args:
        provider: LLMProvider instance
        messages: Messages to send
        model_id: Model ID to use
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Factor by which to increase delay after each retry
        jitter: Random factor to add to delay to prevent synchronized retries
        context: Optional context object
        verbose: When True, emit retry progress messages via the provided logger
        logger: Optional ``logging.Logger`` instance to use when ``verbose`` is True
        **options: Additional provider-specific options
        
    Returns:
        Final LLMResponse object (either successful or last failed attempt)
    """
    retries = 0
    last_response = None
    log = logger or logging.getLogger(__name__)

    def _log(message):
        if verbose:
            log.info(message)
    
    while retries <= max_retries:
        thread_id = threading.get_ident()
        if retries > 0:
            _log(f"[Thread-{thread_id}] Retry attempt {retries}/{max_retries}")
            
        response = provider.make_chat_completion_request(
            messages=messages, 
            model_id=model_id,
            context=context,
            **options
        )
        
        if response.success or not response.is_retryable:
            return response
            
        last_response = response
        retries += 1
        
        if retries <= max_retries:
            # Calculate backoff with jitter
            delay = min(initial_delay * (backoff_factor ** (retries - 1)), 60)
            actual_delay = delay * (1 + random.uniform(-jitter, jitter))
            
            error_msg = "Unknown error"
            if response.error_info and "message" in response.error_info:
                error_msg = response.error_info["message"]
                
            _log(
                f"[Thread-{thread_id}] Request failed with retryable error: {error_msg}. "
                f"Waiting {actual_delay:.2f}s before retry."
            )
                  
            time.sleep(actual_delay)
    
    # We've exhausted retries
    if last_response:
        # Mark as no longer retryable since we've exhausted attempts
        last_response.is_retryable = False
        
        if last_response.error_info:
            last_response.error_info["max_retries_exceeded"] = True
    
    return last_response
