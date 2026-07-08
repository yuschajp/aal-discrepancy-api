#!/usr/bin/env python3
"""
test_llm_retry.py — unit tests for retry-with-backoff around the Anthropic
LLM call in main.extract_fields / main._messages_create_with_retry.

No live API calls. pytest-asyncio is NOT installed, so async code is driven
via asyncio.run(...) inside plain sync test functions.
"""
import os
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

import asyncio
from unittest.mock import Mock, AsyncMock, patch

import httpx
import anthropic
from fastapi import HTTPException

import main


def _req():
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _rate_limit_err(retry_after="2"):
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return anthropic.RateLimitError(
        "rate limited",
        response=httpx.Response(429, headers=headers, request=_req()),
        body=None,
    )


def _server_err():
    return anthropic.InternalServerError(
        "overloaded", response=httpx.Response(529, request=_req()), body=None
    )


def _timeout_err():
    return anthropic.APITimeoutError(request=_req())


def _connection_err():
    return anthropic.APIConnectionError(request=_req())


def _bad_request_err():
    return anthropic.BadRequestError(
        "bad", response=httpx.Response(400, request=_req()), body=None
    )


def _auth_err():
    return anthropic.AuthenticationError(
        "unauthorized", response=httpx.Response(401, request=_req()), body=None
    )


def _success_msg(text='{"counterparty": "X", "extraction_confidence": 0.9}'):
    return Mock(content=[Mock(text=text)])


def test_retries_ratelimit_then_succeeds():
    success = _success_msg()
    with patch.object(main, "client") as mock_client, \
         patch("main.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        mock_client.messages.create = Mock(
            side_effect=[_rate_limit_err(), _rate_limit_err(), success]
        )
        result = asyncio.run(main._messages_create_with_retry(model="m", max_tokens=1, system="s", messages=[]))
        assert result is success
        assert mock_client.messages.create.call_count == 3
        assert mock_sleep.await_count == 2


def test_retries_timeout_then_succeeds():
    with patch.object(main, "client") as mock_client, \
         patch("main.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        mock_client.messages.create = Mock(side_effect=[_timeout_err(), _success_msg()])
        result = asyncio.run(main._messages_create_with_retry(model="m", max_tokens=1, system="s", messages=[]))
        assert result is not None
        assert mock_client.messages.create.call_count == 2
        assert mock_sleep.await_count == 1


def test_retries_connection_error_then_succeeds():
    with patch.object(main, "client") as mock_client, \
         patch("main.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        mock_client.messages.create = Mock(side_effect=[_connection_err(), _success_msg()])
        result = asyncio.run(main._messages_create_with_retry(model="m", max_tokens=1, system="s", messages=[]))
        assert result is not None
        assert mock_client.messages.create.call_count == 2
        assert mock_sleep.await_count == 1


def test_retries_529_then_succeeds():
    with patch.object(main, "client") as mock_client, \
         patch("main.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        mock_client.messages.create = Mock(side_effect=[_server_err(), _success_msg()])
        result = asyncio.run(main._messages_create_with_retry(model="m", max_tokens=1, system="s", messages=[]))
        assert result is not None
        assert mock_client.messages.create.call_count == 2
        assert mock_sleep.await_count == 1


def test_no_retry_on_bad_request():
    with patch.object(main, "client") as mock_client, \
         patch("main.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        mock_client.messages.create = Mock(side_effect=[_bad_request_err()])
        try:
            asyncio.run(main._messages_create_with_retry(model="m", max_tokens=1, system="s", messages=[]))
            assert False, "expected BadRequestError to propagate"
        except anthropic.BadRequestError:
            pass
        assert mock_client.messages.create.call_count == 1
        mock_sleep.assert_not_awaited()


def test_no_retry_on_auth_error():
    with patch.object(main, "client") as mock_client, \
         patch("main.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        mock_client.messages.create = Mock(side_effect=[_auth_err()])
        try:
            asyncio.run(main._messages_create_with_retry(model="m", max_tokens=1, system="s", messages=[]))
            assert False, "expected AuthenticationError to propagate"
        except anthropic.AuthenticationError:
            pass
        assert mock_client.messages.create.call_count == 1
        mock_sleep.assert_not_awaited()


def test_exhausts_after_max_attempts():
    with patch.object(main, "client") as mock_client, \
         patch("main.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        mock_client.messages.create = Mock(side_effect=[_server_err() for _ in range(main.AAL_LLM_MAX_ATTEMPTS)])
        try:
            asyncio.run(main._messages_create_with_retry(model="m", max_tokens=1, system="s", messages=[]))
            assert False, "expected InternalServerError to propagate"
        except anthropic.InternalServerError:
            pass
        assert mock_client.messages.create.call_count == main.AAL_LLM_MAX_ATTEMPTS
        assert mock_sleep.await_count == main.AAL_LLM_MAX_ATTEMPTS - 1


def test_honors_retry_after_header():
    with patch.object(main, "client") as mock_client, \
         patch("main.asyncio.sleep", new=AsyncMock()) as mock_sleep, \
         patch("main.random.uniform", return_value=0.0):
        mock_client.messages.create = Mock(side_effect=[_rate_limit_err(retry_after="2"), _success_msg()])
        asyncio.run(main._messages_create_with_retry(model="m", max_tokens=1, system="s", messages=[]))
        mock_sleep.assert_awaited_once_with(2.0)


def test_full_jitter_bounds():
    delays = []

    async def _record_sleep(d):
        delays.append(d)

    with patch.object(main, "client") as mock_client, \
         patch("main.asyncio.sleep", new=AsyncMock(side_effect=_record_sleep)):
        mock_client.messages.create = Mock(
            side_effect=[_server_err(), _server_err(), _server_err(), _success_msg()]
        )
        asyncio.run(main._messages_create_with_retry(model="m", max_tokens=1, system="s", messages=[]))

    assert len(delays) == 3
    for i, d in enumerate(delays):
        cap = min(main.AAL_LLM_BACKOFF_MAX_S, main.AAL_LLM_BACKOFF_BASE_S * main.AAL_LLM_BACKOFF_MULTIPLIER ** i)
        assert 0 <= d <= cap


def test_extract_fields_maps_exhausted_to_502():
    with patch.object(main, "_messages_create_with_retry", new=AsyncMock(side_effect=_rate_limit_err())):
        try:
            asyncio.run(main.extract_fields("txt", "IRS"))
            assert False, "expected HTTPException"
        except HTTPException as e:
            assert e.status_code == 502
            assert "after 4 attempts" in e.detail
            assert "RateLimitError" in e.detail


def test_extract_fields_non_retryable_maps_to_502():
    with patch.object(main, "_messages_create_with_retry", new=AsyncMock(side_effect=_bad_request_err())):
        try:
            asyncio.run(main.extract_fields("txt", "IRS"))
            assert False, "expected HTTPException"
        except HTTPException as e:
            assert e.status_code == 502
            assert "BadRequestError" in e.detail
            assert "attempts" not in e.detail


def test_extract_fields_success_parses():
    msg = _success_msg('{"counterparty": "GS", "extraction_confidence": 0.9}')
    with patch.object(main, "_messages_create_with_retry", new=AsyncMock(return_value=msg)):
        result = asyncio.run(main.extract_fields("txt", "IRS"))
        assert result == ({"counterparty": "GS"}, 0.9)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
