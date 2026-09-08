"""Provider selection must never mix models, credentials or defaults."""

import importlib
from pathlib import Path

import pytest

import backend


def resolve(**kwargs):
    assert hasattr(backend, "resolve_provider"), "Provider-specific configuration is missing"
    return backend.resolve_provider(**kwargs)


@pytest.fixture(autouse=True)
def clean_provider_environment(monkeypatch):
    for name in [
        "LLM_PROVIDER",
        "ORCAROUTER_MODEL",
        "ORCAROUTER_API_URL",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_API_URL",
        "ORCAROUTER_API_KEY",
        "DEEPSEEK_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_deepseek_defaults_remain_compatible():
    config = resolve()
    assert (config.provider, config.model, config.base_url) == (
        "deepseek",
        "deepseek-v4-flash",
        "https://api.deepseek.com/v1",
    )


def test_orca_environment_is_independent(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "orcarouter")
    monkeypatch.setenv("ORCAROUTER_MODEL", "deepseek/deepseek-v4-flash-free")
    monkeypatch.setenv("DEEPSEEK_MODEL", "wrong-model")
    config = resolve()
    assert config.model == "deepseek/deepseek-v4-flash-free"
    assert config.base_url == "https://api.orcarouter.ai/v1"
    assert config.hipporag_model == "orcarouter/deepseek/deepseek-v4-flash-free"


def test_explicit_config_overrides_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("ORCAROUTER_MODEL", "deepseek/deepseek-v4-flash-free")
    config = resolve(provider="orcarouter", model="qwen/test-model", base_url="https://gateway.example/v1/")
    assert config.model == "qwen/test-model"
    assert config.base_url == "https://gateway.example/v1"


@pytest.mark.parametrize(
    "model",
    [
        "",
        " ",
        "deepseek-chat",
        "orcarouter/deepseek/model",
        "orcarouter/auto",
        "deepseek/",
        "../model",
        "vendor/model/extra",
    ],
)
def test_orca_requires_an_explicit_vendor_model(model):
    with pytest.raises(ValueError):
        resolve(provider="orcarouter", model=model)


def test_direct_provider_cannot_dispatch_another_backend():
    with pytest.raises(ValueError):
        resolve(provider="deepseek", model="orcarouter/deepseek/deepseek-v4-flash")


@pytest.mark.parametrize(
    "url",
    [
        "ftp://gateway.example",
        "https://user:secret@gateway.example",
        "https://gateway.example/v1?key=secret",
        "https://gateway.example/chat/completions",
    ],
)
def test_invalid_base_url_is_rejected(url):
    with pytest.raises(ValueError):
        resolve(provider="orcarouter", model="deepseek/model", base_url=url)


def test_only_selected_provider_key_is_required(monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "test-orca")
    assert backend.validate_key("orcarouter") == "test-orca"
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        backend.validate_key("deepseek")


@pytest.mark.parametrize("entry", ["app", "advanced_app"])
@pytest.mark.parametrize("single_query", [False, True])
def test_cli_selects_orca_without_deepseek_credentials(monkeypatch, tmp_path, entry, single_query):
    import cli
    import demo_core

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setenv("ORCAROUTER_API_KEY", "test-orca")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("ORCAROUTER_MODEL", "qwen/env-model")
    observed = {}

    def initialize(self, **kwargs):
        observed.update(kwargs)
        observed["provider"] = self.provider
        observed["save_dir"] = self.save_dir
        return True

    monkeypatch.setattr(demo_core.HippoRAGDemo, "initialize", initialize)
    monkeypatch.setattr(demo_core.HippoRAGDemo, "index_documents", lambda self: True)
    monkeypatch.setattr(
        demo_core.HippoRAGDemo, "ask", lambda self, q: {"query": q, "answer": "ok", "retrieved_docs": []}
    )
    monkeypatch.setattr("builtins.input", lambda *a: "exit")
    args = ["--provider", "orcarouter", "--model", "deepseek/deepseek-v4-flash-free"]
    if single_query:
        args += ["--query", "hello"]
    assert importlib.import_module(entry).main(args) == 0
    assert observed["provider"] == "orcarouter"
    assert observed["llm_model"] == "deepseek/deepseek-v4-flash-free"
    assert Path(observed["save_dir"]).name == "orcarouter"


def test_invalid_provider_in_environment_fails_before_loading(monkeypatch, tmp_path):
    import app
    import cli

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    with pytest.raises(SystemExit) as failure:
        app.main([])
    assert failure.value.code == 2
