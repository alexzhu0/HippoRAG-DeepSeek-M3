"""Regression coverage for the original CLI and HippoRAG 2 contract."""

import csv
import importlib
import json
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def advanced():
    return importlib.import_module("advanced_app")


def test_advanced_cli_initializes_and_indexes_before_questions(advanced, monkeypatch):
    events = []
    monkeypatch.setattr(
        advanced.HippoRAGDemo, "initialize", lambda *a, **k: events.append("initialize") or True
    )
    monkeypatch.setattr(
        advanced.HippoRAGDemo, "load_documents", lambda *a, **k: events.append("load") or True
    )
    monkeypatch.setattr(
        advanced.HippoRAGDemo, "index_documents", lambda *a, **k: events.append("index") or True
    )
    monkeypatch.setattr("builtins.input", lambda *a: "exit")
    monkeypatch.setattr(sys, "argv", ["advanced_app.py"])
    advanced.main()
    assert {"initialize", "load", "index"}.issubset(events)
    assert events.index("initialize") < events.index("index")
    assert events.index("load") < events.index("index")


def test_advanced_extracts_upstream_tuple_answer(advanced):
    demo = advanced.HippoRAGDemo()
    solution = SimpleNamespace(
        question="问题",
        answer="答案",
        docs=["来源"],
        doc_scores=None,
        gold_answers=None,
        gold_docs=None,
        thoughts=None,
        doc_metadata=None,
        graph_seeds=None,
    )
    demo.hipporag = SimpleNamespace(rag_qa=lambda **k: ([solution], ["答案"], [{}]))
    assert demo.ask("问题") == {"query": "问题", "answer": "答案", "retrieved_docs": ["来源"]}
    assert demo.history[0]["answer"] == "答案"


def test_document_objects_become_strings(advanced, tmp_path):
    source = tmp_path / "docs.json"
    source.write_text(json.dumps([{"id": "a", "content": " 文档 "}, " 文档 ", "另一篇"]), encoding="utf-8")
    demo = advanced.HippoRAGDemo()
    assert demo.load_documents(source)
    assert demo.docs == ["文档", "另一篇"]


@pytest.mark.parametrize("contents", [[], {}, [" "], [{"id": "a"}], [4]])
def test_rejects_invalid_documents(advanced, tmp_path, contents):
    source = tmp_path / "docs.json"
    source.write_text(json.dumps(contents), encoding="utf-8")
    assert advanced.HippoRAGDemo().load_documents(source) is False


def test_export_includes_sources_without_pandas(advanced, tmp_path):
    demo = advanced.HippoRAGDemo()
    demo.history = [
        {
            "timestamp": "2026-09-08",
            "query": "问题",
            "answer": "答案",
            "retrieved_docs": ["来源"],
            "memory_usage": 10,
        }
    ]
    target = tmp_path / "history.csv"
    assert demo.export_history(target)
    with target.open(encoding="utf-8-sig", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["回答"] == "答案"
    assert json.loads(row["参考资料"]) == ["来源"]


@pytest.mark.parametrize("value", [None, [], (), {"answers": []}, [{"answer": None}]])
def test_missing_answer_does_not_create_history(advanced, value):
    demo = advanced.HippoRAGDemo()
    demo.hipporag = SimpleNamespace(rag_qa=lambda **k: value)
    assert demo.ask("问题") is None
    assert demo.history == []


def test_blank_query_never_calls_backend(advanced):
    demo = advanced.HippoRAGDemo()

    def unexpected(**kwargs):
        raise AssertionError("blank query reached API")

    demo.hipporag = SimpleNamespace(rag_qa=unexpected)
    assert demo.ask("   ") is None
    assert demo.history == []


def test_invalid_reload_discards_stale_documents(advanced, tmp_path):
    demo = advanced.HippoRAGDemo()
    demo.docs = ["old document"]
    assert demo.load_documents(tmp_path / "missing.json") is False
    assert demo.docs == []


def test_startup_failure_never_enters_prompt(advanced, monkeypatch):
    monkeypatch.setattr(advanced.HippoRAGDemo, "load_documents", lambda *a, **k: False)

    def unexpected(*args):
        raise AssertionError("prompt reached after startup failed")

    monkeypatch.setattr("builtins.input", unexpected)
    assert advanced.main([]) == 1
