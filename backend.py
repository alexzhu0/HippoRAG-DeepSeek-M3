"""HippoRAG integration with explicit device and client-local DeepSeek credentials."""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)
_CREDENTIAL_LOCK = threading.RLock()


def validate_key():
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key or key.lower().startswith(("your_", "your-")) or "您的" in key:
        raise ValueError("请在项目 .env 或环境变量中设置有效的 DEEPSEEK_API_KEY")
    return key


def select_device(requested="auto"):
    import torch

    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("device 必须是 auto/cpu/mps/cuda")
    available = {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
    if requested == "auto":
        return next(device for device in ("mps", "cuda", "cpu") if available[device])
    if not available[requested]:
        raise ValueError(f"设备 {requested} 不可用，请使用 --device cpu 或 auto")
    return requested


@contextmanager
def _deepseek_credentials(key):
    # Upstream constructs its OpenAI-compatible client from OPENAI_API_KEY.
    # Restore the caller's environment as soon as the client captures the key.
    with _CREDENTIAL_LOCK:
        previous = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = key
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous


def create_embedding(config, device):
    from local_embedding import LocalContriever

    return LocalContriever(config, device)


def create_backend(
    *,
    save_dir,
    llm_model="deepseek-chat",
    embedding_model="facebook/contriever",
    llm_base_url=None,
    device="auto",
    batch_size=8,
):
    key = validate_key()
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size 必须是正整数")
    url = (llm_base_url or os.getenv("DEEPSEEK_API_URL") or "https://api.deepseek.com/v1").rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("DEEPSEEK_API_URL 必须是有效的 HTTP(S) API 基础地址")
    if parsed.query or parsed.fragment or parsed.path.endswith("/chat/completions"):
        raise ValueError("DEEPSEEK_API_URL 应使用基础地址，如 https://api.deepseek.com/v1")
    selected = select_device(device)
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    config = BaseConfig(
        save_dir=str(save_dir),
        llm_name=llm_model,
        llm_base_url=url,
        embedding_model_name=embedding_model,
        embedding_batch_size=batch_size,
        embedding_max_seq_len=512,
        embedding_model_dtype="float32",
        synonymy_edge_query_batch_size=128,
        synonymy_edge_key_batch_size=1024,
        max_retry_attempts=2,
    )
    embedding = create_embedding(config, selected)
    try:
        with _deepseek_credentials(key):
            rag = HippoRAG(
                global_config=config,
                embedding_model=embedding,
                index_identity="deepseek-m3-contriever-mean-pooling-v1",
            )
    except BaseException:
        embedding.close()
        raise
    logger.info("嵌入设备：%s；批大小：%d；索引目录：%s", selected, batch_size, save_dir)
    return rag, embedding
