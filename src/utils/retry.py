import logging

import groq
import litellm.exceptions
import redis.exceptions
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# groq.RateLimitError: raised by direct Groq SDK calls (retrieval.py, guardrails.py)
# litellm.exceptions.RateLimitError: raised by CrewAI which routes through LiteLLM
GROQ_TRANSIENT = (
    groq.RateLimitError,
    groq.InternalServerError,
    groq.APIConnectionError,
    groq.APITimeoutError,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.ServiceUnavailableError,
)


# it a pre-configured decorator, apply to any funtion makes a groq api, call raise of the groq_transient
groq_retry = retry(
    # retry if the exception is one of those
    retry=retry_if_exception_type(GROQ_TRANSIENT),

    # it means after 3 retries, (the first call is excluded)
    # 4 attempts (1 original + 3 retries)
    stop=stop_after_attempt(4),

    # how long to pause, it doubles the base each attempt plus some jitter(random noise)
    # cz without jitter, if many users send request may have rate limit at the same time
    # the max wait is at 30 seconds
    wait=wait_exponential_jitter(initial=1, max=60),

    # call it before each wait
    before_sleep=before_sleep_log(logger, logging.WARNING),

    # after ALL attempts fail, re-raise the original exception
    reraise=True,
)


REDIS_TRANSIENT = (
    redis.exceptions.ConnectionError,
    redis.exceptions.TimeoutError,
)

redis_retry = retry(
    retry=retry_if_exception_type(REDIS_TRANSIENT),

    # 3 attempts (1 original + 2 retries)
    stop=stop_after_attempt(3),

    wait=wait_exponential_jitter(initial=0.5, max=5),

    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
