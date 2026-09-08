"""Device selection, credential routing and real local embedding regression tests."""

import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import pytest


def backend_module():
    assert importlib.util.find_spec("backend") is not None, "An explicit device/credential backend is missing"
    import backend

    return backend


@pytest.mark.parametrize(
    "mps,cuda,want", [(True, False, "mps"), (False, True, "cuda"), (False, False, "cpu")]
)
def test_auto_selects_available_device(monkeypatch, mps, cuda, want):
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    assert backend_module().select_device("auto") == want


def test_explicit_unavailable_device_is_an_error(monkeypatch):
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="mps"):
        backend_module().select_device("mps")


@pytest.fixture
def tiny_model(tmp_path):
    from transformers import BertConfig, BertModel, BertTokenizerFast

    model_dir = tmp_path / "tiny-contriever"
    model_dir.mkdir()
    (model_dir / "vocab.txt").write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n", encoding="utf-8"
    )
    BertTokenizerFast(vocab_file=str(model_dir / "vocab.txt")).save_pretrained(model_dir)
    BertModel(
        BertConfig(
            vocab_size=7,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=512,
        )
    ).save_pretrained(model_dir)
    return model_dir


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_embeddings_use_real_batches_and_handle_empty_input(tiny_model, device):
    import torch

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this runner")
    from hipporag.utils.config_utils import BaseConfig

    config = BaseConfig(
        embedding_model_name=str(tiny_model),
        embedding_batch_size=2,
        embedding_max_seq_len=8,
        embedding_model_dtype="float32",
    )
    embedding = backend_module().create_embedding(config, device)
    try:
        texts = ["hello", "world", "hello world " * 20]
        actual = embedding.batch_encode(texts)
        assert actual.shape == (3, 8)
        assert actual.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), [1, 1, 1], atol=1e-5)
        np.testing.assert_allclose(actual, np.vstack([embedding.batch_encode(t) for t in texts]), atol=1e-5)
        assert embedding.embedding_model.device.type == device
        assert embedding.batch_encode([]).shape == (0, 8)
    finally:
        embedding.close()


def test_deepseek_key_is_used_and_other_key_restored(monkeypatch, tiny_model, tmp_path):
    import hipporag

    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        captured["key"] = os.environ.get("OPENAI_API_KEY")
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(hipporag, "HippoRAG", factory)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek")
    monkeypatch.setenv("OPENAI_API_KEY", "other-provider")
    rag, embedding = backend_module().create_backend(
        save_dir=tmp_path, embedding_model=str(tiny_model), device="cpu", batch_size=2
    )
    embedding.close()
    assert captured["key"] == "test-deepseek"
    assert os.environ["OPENAI_API_KEY"] == "other-provider"
    assert captured["global_config"].llm_base_url == "https://api.deepseek.com/v1"
    assert captured["global_config"].embedding_batch_size == 2
    assert captured["index_identity"]


@pytest.mark.parametrize("key", ["", "your_deepseek_api_key_here", "您的DeepSeek_API密钥"])
def test_bad_key_rejected_before_loading_models(monkeypatch, tmp_path, key):
    monkeypatch.setenv("DEEPSEEK_API_KEY", key)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        backend_module().create_backend(save_dir=tmp_path)


def test_invalid_batch_size_fails_before_loading_models(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek")
    with pytest.raises(ValueError, match="batch_size"):
        backend_module().create_backend(save_dir=tmp_path, batch_size=0)


def test_failed_backend_construction_restores_key_and_closes_embedding(monkeypatch, tmp_path):
    import hipporag

    backend = backend_module()
    closed = []
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek")
    monkeypatch.setattr(
        backend, "create_embedding", lambda *a: SimpleNamespace(close=lambda: closed.append(True))
    )

    def fail(**kwargs):
        assert os.environ["OPENAI_API_KEY"] == "test-deepseek"
        raise RuntimeError("constructor failed")

    monkeypatch.setattr(hipporag, "HippoRAG", fail)
    with pytest.raises(RuntimeError, match="constructor failed"):
        backend.create_backend(save_dir=tmp_path, device="cpu")
    assert "OPENAI_API_KEY" not in os.environ
    assert closed == [True]


def test_real_hipporag_persists_and_reopens_index(monkeypatch, tiny_model, tmp_path):
    import json

    from openai.resources.chat.completions import Completions
    from openai.types.chat import ChatCompletion

    from demo_core import parse_result

    calls = []

    def completion(self, **kwargs):
        calls.append(kwargs)
        message = kwargs["messages"][-1]["content"]
        if "fact_before_filter" in message:
            content = '[[ ## fact_after_filter ## ]]\n{"fact": [["hello", "knows", "world"]]}'
        elif "Question:" in message:
            content = "Answer: world"
        else:
            content = json.dumps(
                {"named_entities": ["hello", "world"], "triples": [["hello", "knows", "world"]]}
            )
        return ChatCompletion.model_validate(
            {
                "id": "offline-test",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-chat",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    monkeypatch.setattr(Completions, "create", completion)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek")
    monkeypatch.setenv("OPENAI_API_KEY", "other-provider")
    options = dict(save_dir=tmp_path / "index", embedding_model=str(tiny_model), device="cpu", batch_size=2)
    rag, embedding = backend_module().create_backend(**options)
    try:
        assert rag.llm_model.openai_client.api_key == "test-deepseek"
        assert os.environ["OPENAI_API_KEY"] == "other-provider"
        rag.index(docs=["hello world", "world hello"])
        assert len(rag.chunk_embedding_store.get_all_ids()) == 2
        assert len(rag.fact_embedding_store.get_all_ids()) == 1
        answer, docs = parse_result(rag.rag_qa(queries=["hello"]))
        assert answer == "world"
        assert set(docs) == {"hello world", "world hello"}
    finally:
        rag.close()
        embedding.close()
    reopened, embedding = backend_module().create_backend(**options)
    try:
        assert len(reopened.chunk_embedding_store.get_all_ids()) == 2
        assert len(reopened.fact_embedding_store.get_all_ids()) == 1
        before = len(calls)
        reopened.index(docs=["hello world", "world hello"])
        assert len(calls) == before
    finally:
        reopened.close()
        embedding.close()
