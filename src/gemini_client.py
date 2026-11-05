"""Gemini API client: asynchronous calls with rate limiting, retries and cost.

``AsyncRateLimiter`` keeps requests within per-minute request and token limits;
``process_batch_async`` sends a list of prompts concurrently. Calls are retried
on HTTP 429, 500 and 503. Costs are computed from token counts (thinking
tokens included) with LiteLLM's price table. The API key is read from the
``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) environment variable.
"""

import os
import time
import asyncio
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import tenacity
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from litellm import completion_cost, cost_per_token


# ---------------------------------------------------------------------------
# Retry helpers for transient Gemini API errors (429, 500, 503, etc.)
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS_CODES = {429, 500, 503}


def _is_retryable_error(exc: BaseException) -> bool:
    """Return True for transient Gemini API errors worth retrying."""
    if isinstance(exc, genai_errors.ServerError):
        return True  # covers 500, 503
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "status_code", 0) in _RETRYABLE_STATUS_CODES
    return False


def _log_retry(retry_state: tenacity.RetryCallState) -> None:
    """Print a warning message before each retry sleep."""
    exc = retry_state.outcome.exception()
    sleep = retry_state.next_action.sleep if retry_state.next_action else 0
    print(
        f"  [retry] {type(exc).__name__}: {exc} — "
        f"attempt {retry_state.attempt_number}, "
        f"sleeping {sleep:.1f}s before next try"
    )


# Shared retry policy: exponential back-off 2 s → 60 s, up to 6 attempts.
_RETRY_POLICY = dict(
    retry=tenacity.retry_if_exception(_is_retryable_error),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
    stop=tenacity.stop_after_attempt(6),
    before_sleep=_log_retry,
    reraise=True,
)


class AsyncRateLimiter:
    """
    Asynchronous rate limiter for API calls.
    """

    def __init__(self, rpm_limit: int, tpm_limit: int, window_size: int = 60):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.window_size = window_size
        self.requests = deque()
        self.tokens = deque()
        self._lock: Optional[asyncio.Lock] = None
        self._lock_loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_lock(self) -> asyncio.Lock:
        """Return an asyncio.Lock bound to the current running event loop,
        creating a new one if the loop has changed (e.g. between asyncio.run calls)."""
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def _clean_old_entries(self):
        """Remove entries older than the window size."""
        current_time = time.monotonic()
        while self.requests and current_time - self.requests[0][1] > self.window_size:
            self.requests.popleft()
        while self.tokens and current_time - self.tokens[0][1] > self.window_size:
            self.tokens.popleft()

    async def wait_if_needed(self, token_count: int):
        """Wait until a request can be made within rate limits."""
        async with self._get_lock():
            while True:
                await self._clean_old_entries()

                # Check request rate
                if len(self.requests) < self.rpm_limit:
                    # Check token rate
                    current_tokens = sum(t[0] for t in self.tokens)
                    if current_tokens + token_count <= self.tpm_limit:
                        # Record request
                        current_time = time.monotonic()
                        self.requests.append((1, current_time))
                        self.tokens.append((token_count, current_time))
                        return

                # Wait before checking again
                await asyncio.sleep(1)


@tenacity.retry(**_RETRY_POLICY)
def _generate_content_sync(
    client: genai.Client,
    model: str,
    prompt: str,
    generation_config: types.GenerateContentConfig,
) -> Any:
    """Synchronous generate_content with retry on transient errors."""
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=generation_config,
    )


async def generate_response_async(
    client: genai.Client,
    prompt: str,
    model: str,
    rate_limiter: AsyncRateLimiter,
    generation_config: types.GenerateContentConfig,
) -> Any:
    """
    Generate a response from Gemini asynchronously with rate limiting.

    Both ``count_tokens`` and ``generate_content`` are retried automatically
    on transient Gemini errors (429 / 500 / 503).
    """
    try:
        # count_tokens is retried via its own @tenacity.retry decorator
        token_count = count_tokens(client, prompt, model)

        await rate_limiter.wait_if_needed(token_count)

        # Run the (synchronous, retried) generate call in a thread
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: _generate_content_sync(client, model, prompt, generation_config),
        )
        return response
    except Exception as e:
        print(f"Error in generate_response_async: {e}")
        return None


async def process_batch_async(
    prompts: List[str],
    model: str,
    rate_limiter: AsyncRateLimiter,
    temperature: float = 0.7,
    num_samples: int = 1,
    thinking_budget: int = 0,
    thinking_level: Optional[str] = None,
    response_mime_type: Optional[str] = None,
    response_schema: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """
    Process a batch of prompts asynchronously using Gemini API.
    """
    client = setup_gemini_client()

    # Prepare generation config
    config_kwargs = {
        "temperature": temperature,
        "candidate_count": num_samples,
    }

    # Configure thinking
    if thinking_level:
        t_level = (
            thinking_level.upper()
            if isinstance(thinking_level, str)
            else thinking_level
        )
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            include_thoughts=True,
            thinking_level=t_level,
        )
    elif thinking_budget > 0:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=thinking_budget,
        )

    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type

    if response_schema:
        config_kwargs["response_json_schema"] = response_schema

    generation_config = types.GenerateContentConfig(**config_kwargs)

    # Create tasks
    tasks = []
    for prompt in prompts:
        task = generate_response_async(
            client=client,
            prompt=prompt,
            model=model,
            rate_limiter=rate_limiter,
            generation_config=generation_config,
        )
        tasks.append(task)

    # Run tasks concurrently
    results = await asyncio.gather(*tasks)
    return results


def setup_gemini_client():
    """Initialize the Gemini client."""
    # GEMINI_API_KEY takes precedence; GOOGLE_API_KEY is the google-genai fallback.
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "API key not found. Set it in your shell first: "
            "export GEMINI_API_KEY=<your-key>"
        )
    return genai.Client(api_key=api_key)


def extract_completion_text_for_cost(response: Any) -> str:
    """
    Extract the full completion string (thoughts + output text) from a Gemini
    GenerateContentResponse for LiteLLM cost calculation. Includes thinking
    tokens so cost reflects actual usage.

    Gemini parts: each Part has .text (the content) and optionally .thought=True
    (boolean flag). Thought content is in part.text when thought=True, not in
    part.thought. So we collect all part.text in order (thought blocks then
    final output).
    """
    if not response:
        return ""
    parts_text: List[str] = []
    if getattr(response, "candidates", None):
        for c in response.candidates:
            if not getattr(c, "content", None) or not getattr(c.content, "parts", None):
                continue
            for part in c.content.parts:
                if getattr(part, "text", None) and part.text:
                    parts_text.append(part.text)
    if not parts_text and getattr(response, "text", None):
        parts_text.append(response.text)
    return "\n".join(parts_text) if parts_text else ""


def calculate_cost(
    prompt: str, response: Any, model: str
) -> float:
    """
    Calculate cost (USD) using LiteLLM's completion_cost with prompt + completion
    strings so LiteLLM uses its live pricing (api.litellm.ai). Completion text
    includes thought tokens when present.
    """
    completion_text = extract_completion_text_for_cost(response)
    try:
        return completion_cost(model=model, prompt=prompt, completion=completion_text)
    except Exception as e:
        print(f"Warning: Could not calculate cost with LiteLLM - {e}")
        return 0.0


def calculate_cost_from_tokens(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    batch_mode: bool = False,
) -> float:
    """
    Calculate cost (USD) using LiteLLM's cost_per_token with token counts.
    Batch mode applies a 50% discount.

    See: https://docs.litellm.ai/docs/completion/token_usage#5-cost_per_token
    """
    try:
        prompt_cost, completion_cost_val = cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        total_cost = prompt_cost + completion_cost_val
        if batch_mode:
            total_cost *= 0.5  # Batch API is 50% cheaper
        return total_cost
    except Exception as e:
        print(f"Warning: Could not calculate cost with LiteLLM - {e}")
        return 0.0


@tenacity.retry(**_RETRY_POLICY)
def count_tokens(client: genai.Client, text: str, model: str) -> int:
    """Count tokens for text using Gemini count_tokens API."""
    result = client.models.count_tokens(model=model, contents=text)
    if hasattr(result, "total_tokens"):
        return result.total_tokens
    if isinstance(result, int):
        return result
    # Some SDKs return an object with total_token_count
    return getattr(result, "total_token_count", int(result))


def truncate_to_token_limit(
    client: genai.Client, doc: str, limit: int, model: str
) -> Tuple[str, int, bool]:
    """
    Truncate document to stay under token limit. Returns (truncated_doc, token_count, was_truncated).
    Uses binary search on character position to find the longest prefix under the limit.
    """
    n = len(doc)
    if n == 0:
        return doc, 0, False

    count = count_tokens(client, doc, model)
    if count <= limit:
        return doc, count, False

    # Binary search: find largest hi such that doc[:hi] is under limit
    lo, hi = 0, n
    while lo < hi - 1:
        mid = (lo + hi) // 2
        candidate = doc[:mid]
        count_mid = count_tokens(client, candidate, model)
        if count_mid <= limit:
            lo = mid
        else:
            hi = mid

    truncated = doc[:lo]
    final_count = count_tokens(client, truncated, model)
    return truncated, final_count, True
