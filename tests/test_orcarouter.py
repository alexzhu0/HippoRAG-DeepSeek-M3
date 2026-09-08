import httpx
import pytest
from openai import APIStatusError


def test_retry_policy():
    from orcarouter_provider import retry_delay

    def error(status, code, headers=None):
        return APIStatusError(
            "sensitive upstream message",
            response=httpx.Response(
                status, headers=headers, request=httpx.Request("POST", "https://example.com")
            ),
            body={"code": code},
        )

    assert retry_delay(error(429, "free_rate_limited"), 0) is None
    assert retry_delay(error(429, "free_rate_limited", {"Retry-After": "2"}), 0) == 2
    assert retry_delay(error(429, "free_rate_limited", {"Retry-After": "86400"}), 0) is None
    assert retry_delay(error(401, "invalid_api_key"), 0) is None
    assert retry_delay(error(503, "server_error"), 0) == 0.5
    assert retry_delay(error(503, "server_error"), 2) is None


@pytest.mark.parametrize(
    "model,thinking", [("deepseek/deepseek-v4-flash-free", True), ("qwen/test-model", False)]
)
def test_orca_client_cache_and_errors(tmp_path, monkeypatch, model, thinking):
    from hipporag.utils.config_utils import BaseConfig
    from openai.resources.chat.completions import Completions
    from openai.types.chat import ChatCompletion

    from orcarouter_provider import ProjectOrcaRouterLLM

    monkeypatch.setenv("ORCAROUTER_API_KEY", "test-orca-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    llm = ProjectOrcaRouterLLM(
        BaseConfig(
            save_dir=str(tmp_path),
            llm_name="orcarouter/" + model,
            llm_base_url="https://api.orcarouter.ai/v1",
            max_retry_attempts=0,
        )
    )
    calls = []

    def create(self, **kwargs):
        calls.append(kwargs)
        assert self._client.api_key == "test-orca-key"
        assert kwargs["model"] == model
        assert str(self._client.base_url) == "https://api.orcarouter.ai/v1/"
        assert "max_completion_tokens" in kwargs
        assert "max_tokens" not in kwargs
        assert ("extra_body" in kwargs) == thinking
        return ChatCompletion.model_validate(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    monkeypatch.setattr(Completions, "create", create)
    try:
        assert llm.infer([{"role": "user", "content": "hello"}])[0] == "hello"
        assert llm.infer([{"role": "user", "content": "hello"}])[2] is True
        assert len(calls) == 1

        def fail(self, **kwargs):
            calls.append(kwargs)
            raise APIStatusError(
                "secret upstream data",
                response=httpx.Response(401, request=httpx.Request("POST", "https://example.com")),
                body={"code": "invalid_api_key"},
            )

        monkeypatch.setattr(Completions, "create", fail)
        with pytest.raises(RuntimeError, match="ORCAROUTER_API_KEY") as error:
            llm.infer([{"role": "user", "content": "new question"}])
        assert "secret" not in str(error.value)
        assert len(calls) == 2
    finally:
        llm.close()


@pytest.mark.parametrize(
    "content,usage",
    [
        ("", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
        ("   ", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
        ("hello", None),
        ("hello", {"prompt_tokens": None, "completion_tokens": None, "total_tokens": 0}),
    ],
)
def test_invalid_response_is_not_cached(tmp_path, monkeypatch, content, usage):
    from hipporag.utils.config_utils import BaseConfig
    from openai.resources.chat.completions import Completions
    from openai.types.chat import ChatCompletion
    from openai.types.completion_usage import CompletionUsage

    from orcarouter_provider import ProjectOrcaRouterLLM

    monkeypatch.setenv("ORCAROUTER_API_KEY", "test-key")
    llm = ProjectOrcaRouterLLM(
        BaseConfig(save_dir=str(tmp_path), llm_name="orcarouter/qwen/test", max_retry_attempts=0)
    )
    calls = []

    def create(self, **kwargs):
        calls.append(kwargs)
        response = ChatCompletion.model_validate(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen/test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": None,
            }
        )

        response.usage = CompletionUsage.model_construct(**usage) if usage is not None else None
        return response

    monkeypatch.setattr(Completions, "create", create)
    try:
        for _ in range(2):
            with pytest.raises(ValueError):
                llm.infer([{"role": "user", "content": "test"}])
        assert len(calls) == 2
    finally:
        llm.close()


@pytest.mark.parametrize(
    "status,code,hint",
    [
        (413, "", "缩短"),
        (403, "free_quota_exhausted", "每日"),
        (429, "free_rate_limited", "限额"),
        (503, "model_not_found", "模型"),
    ],
)
def test_safe_diagnostics(status, code, hint):
    from orcarouter_provider import safe_error

    error = APIStatusError(
        "secret",
        response=httpx.Response(status, request=httpx.Request("POST", "https://example.com")),
        body={"error": {"code": code}},
    )
    assert hint in safe_error(error)
    assert "secret" not in safe_error(error)


def test_orca_failed_constructor_closes_resources(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import hipporag

    import backend
    import orcarouter_provider

    closed = []
    monkeypatch.setenv("ORCAROUTER_API_KEY", "test-orca")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "untouched")
    monkeypatch.setattr(
        backend, "create_embedding", lambda *args: SimpleNamespace(close=lambda: closed.append("embedding"))
    )
    monkeypatch.setattr(
        orcarouter_provider,
        "ProjectOrcaRouterLLM",
        lambda *args: SimpleNamespace(close=lambda: closed.append("llm")),
    )

    def fail(**kwargs):
        assert kwargs["extraction_llm"] is kwargs["qa_llm"]
        raise RuntimeError("construction failed")

    monkeypatch.setattr(hipporag, "HippoRAG", fail)
    with pytest.raises(RuntimeError, match="construction failed"):
        backend.create_backend(save_dir=tmp_path, provider="orcarouter", llm_model="qwen/test", device="cpu")
    assert closed == ["llm", "embedding"]
    import os

    assert os.environ["OPENAI_API_KEY"] == "untouched"
