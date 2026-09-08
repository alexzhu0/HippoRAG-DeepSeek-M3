"""Shared document, QA and history handling; importing this module does no I/O."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "v2"
DEFAULT_DOCUMENTS = ROOT / "data" / "documents.json"


def load_documents(file_path):
    with Path(file_path).open(encoding="utf-8") as stream:
        documents = json.load(stream)
    if not isinstance(documents, list) or not documents:
        raise ValueError("文档必须是非空 JSON 数组")
    texts = []
    seen = set()
    for index, document in enumerate(documents, 1):
        text = document.get("content") if isinstance(document, dict) else document
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"第 {index} 篇文档必须是非空字符串或含 content 的对象")
        text = text.strip()
        if text not in seen:
            seen.add(text)
            texts.append(text)
    return texts


def parse_result(results):
    """Accept HippoRAG 2 QuerySolution tuples and legacy dictionary responses."""
    if isinstance(results, tuple):
        results = results[0] if results else []
    if isinstance(results, list):
        results = results[0] if results else None
    if isinstance(results, dict):
        answer = results.get("answer")
        if answer is None:
            answers = results.get("answers", [])
            answer = answers[0] if isinstance(answers, list) and answers else None
        docs = results.get("docs", results.get("retrieved_nodes", []))
    else:
        answer = getattr(results, "answer", None)
        docs = getattr(results, "docs", [])
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("HippoRAG 未返回有效答案")
    sources = []
    if docs is None:
        docs = []
    if not isinstance(docs, (list, tuple)):
        raise ValueError("HippoRAG 返回的参考资料格式无效")
    for doc in docs:
        if isinstance(doc, dict):
            doc = doc.get("node", doc)
        if isinstance(doc, dict):
            doc = doc.get("content", "")
        if isinstance(doc, str) and doc.strip():
            sources.append(doc.strip())
    return answer.strip(), sources


class HippoRAGDemo:
    """Shared implementation, retaining the original advanced app's public methods."""

    def __init__(self, save_dir=None, *, provider=None, device="auto", batch_size=8):
        import os

        self.provider = provider if provider is not None else os.getenv("LLM_PROVIDER", "deepseek")
        self.save_dir = str(
            save_dir
            if save_dir is not None
            else (DEFAULT_OUTPUT / "orcarouter" if self.provider == "orcarouter" else DEFAULT_OUTPUT)
        )
        self.device = device
        self.batch_size = batch_size
        self.hipporag = None
        self.docs = []
        self.history = []
        self.last_error = None
        self._resources = None

    def _failed(self, operation, error):
        self.last_error = str(error)
        logger.error("%s: %s", operation, error)
        return False

    def initialize(self, llm_model=None, embedding_model="facebook/contriever", llm_base_url=None):
        try:
            from backend import create_backend

            self.close()
            self.hipporag, self._resources = create_backend(
                save_dir=self.save_dir,
                provider=self.provider,
                llm_model=llm_model,
                embedding_model=embedding_model,
                llm_base_url=llm_base_url,
                device=self.device,
                batch_size=self.batch_size,
            )
            return True
        except Exception as error:
            return self._failed("初始化失败", error)

    def load_documents(self, file_path):
        self.docs = []
        try:
            self.docs = load_documents(file_path)
            logger.info("加载 %d 篇去重后的文档", len(self.docs))
            return True
        except (OSError, ValueError) as error:
            return self._failed("加载文档失败", error)

    def index_documents(self):
        if self.hipporag is None or not self.docs:
            return self._failed("索引失败", "请先初始化并加载文档")
        try:
            self.hipporag.index(docs=self.docs)
            return True
        except Exception as error:
            return self._failed("索引失败", error)

    def ask(self, query):
        if self.hipporag is None or not isinstance(query, str) or not query.strip():
            self._failed("提问失败", "请先初始化，并输入非空问题")
            return None
        query = query.strip()
        try:
            answer, sources = parse_result(self.hipporag.rag_qa(queries=[query]))
            record = {"query": query, "answer": answer, "retrieved_docs": sources}
            self.history.append(
                {
                    **record,
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "memory_usage": self.get_memory_usage(),
                }
            )
            return record
        except Exception as error:
            self._failed("提问失败", error)
            return None

    def export_history(self, file_path="qa_history.csv"):
        if not self.history:
            return self._failed("导出失败", "没有历史记录")
        try:
            with Path(file_path).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["时间", "问题", "回答", "参考资料", "内存使用(MB)"]
                )
                writer.writeheader()
                for record in self.history:
                    writer.writerow(
                        {
                            "时间": record["timestamp"],
                            "问题": record["query"],
                            "回答": record["answer"],
                            "参考资料": json.dumps(record["retrieved_docs"], ensure_ascii=False),
                            "内存使用(MB)": record["memory_usage"],
                        }
                    )
            return True
        except (OSError, ValueError) as error:
            return self._failed("导出失败", error)

    def get_memory_usage(self):
        try:
            import psutil

            return psutil.Process().memory_info().rss / 1024**2
        except (ImportError, OSError, AttributeError):
            return 0.0

    def close(self):
        try:
            if self.hipporag is not None:
                self.hipporag.close()
        finally:
            self.hipporag = None
            if self._resources is not None:
                self._resources.close()
                self._resources = None
