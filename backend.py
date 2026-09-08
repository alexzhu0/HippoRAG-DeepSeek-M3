"""HippoRAG integration with explicit device and client-local DeepSeek credentials."""

from __future__ import annotations

import logging
import os
import re
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)
_CREDENTIAL_LOCK = threading.RLock()


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    base_url: str

    @property
    def hipporag_model(self):
        return f"orcarouter/{self.model}" if self.provider == "orcarouter" else self.model


def resolve_provider(provider=None, model=None, base_url=None):
    provider = provider if provider is not None else os.getenv("LLM_PROVIDER", "deepseek")
    if provider not in {"deepseek", "orcarouter"}:
        raise ValueError("provider 必须是 deepseek/orcarouter")
    prefix = provider.upper()
    if model is None:
        model = os.getenv(f"{prefix}_MODEL", "deepseek-v4-flash" if provider == "deepseek" else "")
    model = model.strip()
    pattern = r"[A-Za-z0-9][A-Za-z0-9._:-]*"
    valid = re.fullmatch(f"{pattern}/{pattern}" if provider == "orcarouter" else pattern, model)
    if not valid or model.startswith("orcarouter/"):
        raise ValueError(
            f"请设置有效的 {prefix}_MODEL 或 --model"
            + ("（vendor/model）" if provider == "orcarouter" else "")
        )
    url = (
        (
            base_url
            if base_url is not None
            else os.getenv(
                f"{prefix}_API_URL",
                f"https://api.{provider}.{'ai' if provider == 'orcarouter' else 'com'}/v1",
            )
        )
        .strip()
        .rstrip("/")
    )
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{prefix}_API_URL 必须是有效的 HTTP(S) API 基础地址")
    if parsed.query or parsed.fragment or parsed.path.endswith("/chat/completions"):
        raise ValueError(f"{prefix}_API_URL 应使用 API 基础地址")
    return ProviderConfig(provider, model, url)


def validate_key(provider="deepseek"):
    if provider not in {"deepseek", "orcarouter"}:
        raise ValueError("provider 必须是 deepseek/orcarouter")
    name = f"{provider.upper()}_API_KEY"
    key = os.getenv(name, "").strip()
    if not key or key.lower().startswith(("your_", "your-")) or "您的" in key:
        raise ValueError(f"请在项目 .env 或环境变量中设置有效的 {name}")
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
def _deepseek_credentials(key, name="OPENAI_API_KEY"):
    # Upstream constructs its OpenAI-compatible client from OPENAI_API_KEY.
    # Restore the caller's environment as soon as the client captures the key.
    with _CREDENTIAL_LOCK:
        previous = os.environ.get(name)
        os.environ[name] = key
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def create_embedding(config, device):
    from local_embedding import LocalContriever

    return LocalContriever(config, device)


def create_backend(
    *,
    save_dir,
    llm_model=None,
    provider=None,
    embedding_model="facebook/contriever",
    llm_base_url=None,
    device="auto",
    batch_size=8,
):
    settings = resolve_provider(provider, llm_model, llm_base_url)
    key = validate_key(settings.provider)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size 必须是正整数")
    selected = select_device(device)
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    config = BaseConfig(
        save_dir=str(save_dir),
        llm_name=settings.hipporag_model,
        llm_base_url=settings.base_url,
        embedding_model_name=embedding_model,
        embedding_batch_size=batch_size,
        embedding_max_seq_len=512,
        embedding_model_dtype="float32",
        synonymy_edge_query_batch_size=128,
        synonymy_edge_key_batch_size=1024,
        max_retry_attempts=0 if settings.provider == "orcarouter" else 2,
    )
    resources = ExitStack()
    try:
        embedding = create_embedding(config, selected)
        resources.callback(embedding.close)
        kwargs = {}
        identity = "deepseek-m3-contriever-nonthinking-v2"
        if settings.provider == "orcarouter":
            from orcarouter_provider import ProjectOrcaRouterLLM

            config.temperature = None
            with _deepseek_credentials(key, "ORCAROUTER_API_KEY"):
                llm = ProjectOrcaRouterLLM(config)
            resources.callback(llm.close)
            kwargs = {"extraction_llm": llm, "qa_llm": llm}
            identity = "deepseek-m3-orcarouter-v1"
        with _deepseek_credentials(key) if settings.provider == "deepseek" else ExitStack():
            rag = HippoRAG(global_config=config, embedding_model=embedding, index_identity=identity, **kwargs)
        if settings.provider == "deepseek":
            rag.llm_model.llm_config.generate_params["extra_body"] = {"thinking": {"type": "disabled"}}
    except BaseException:
        resources.close()
        raise
    logger.info(
        "Provider：%s；模型：%s；嵌入设备：%s；批大小：%d；索引目录：%s",
        settings.provider,
        settings.model,
        selected,
        batch_size,
        save_dir,
    )
    return rag, resources
