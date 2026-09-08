"""Fixed-model OrcaRouter requests with bounded retries and safe diagnostics."""

import math
import time
from email.utils import parsedate_to_datetime

from hipporag.llm.openai_gpt import cache_response
from hipporag.llm.orcarouter_llm import OrcaRouterLLM
from openai import APIConnectionError, APIStatusError


def retry_delay(error, attempt):
    if attempt >= 2:
        return None
    if isinstance(error, APIConnectionError):
        return 0.5 * (2**attempt)
    if isinstance(error, APIStatusError):
        if error.status_code == 429:
            # No header can mean a per-request token cap: never retry unchanged.
            header = error.response.headers.get("Retry-After")
            if header is None or attempt > 0:
                return None
            try:
                seconds = float(header)
            except ValueError:
                try:
                    seconds = parsedate_to_datetime(header).timestamp() - time.time()
                except (ValueError, TypeError, OverflowError):
                    return None
            return max(0, seconds) if math.isfinite(seconds) and 0 <= seconds <= 30 else None
        if error.status_code in {500, 502, 503, 504}:
            return 0.5 * (2**attempt)
    return None


def safe_error(error):
    if isinstance(error, APIConnectionError):
        return "OrcaRouter 连接失败或超时，请检查网络和 ORCAROUTER_API_URL 后重试。"
    status = error.status_code
    body = error.body if isinstance(error.body, dict) else {}
    detail = body.get("error", body)
    code = detail.get("code") if isinstance(detail, dict) else None
    code_hints = {
        "free_quota_exhausted": "免费每日额度已耗尽，请等待额度恢复；不会自动切换付费模型。",
        "insufficient_user_quota": "账户额度不足，请检查账户可用额度。",
        "pre_consume_token_quota_failed": "当前额度无法覆盖本次请求，请缩短输入或检查账户额度。",
        "model_not_found": "模型不可用，请核对 ORCAROUTER_MODEL 和当前模型目录。",
    }
    if isinstance(code, str) and code in code_hints:
        return f"OrcaRouter HTTP {status}：" + code_hints[code]
    hints = {
        400: "请求无效，请检查模型参数或缩短文档/问题。",
        401: "认证失败，请检查 ORCAROUTER_API_KEY。",
        403: "权限或额度不足，请检查账户、模型权限及免费日额度。",
        404: "模型或接口不存在，请核对 ORCAROUTER_MODEL 和 ORCAROUTER_API_URL。",
        413: "请求过长，请缩短文档段落、问题或检索上下文。",
        425: "模型尚未开放，请稍后重试或手动选择其他模型。",
        429: "达到限额；有 Retry-After 时按服务端提示稍后重试，无该字段时请缩短输入并检查额度。",
    }
    return f"OrcaRouter HTTP {status}：" + hints.get(status, "服务暂不可用，请稍后重试并检查模型是否可用。")


class ProjectOrcaRouterLLM(OrcaRouterLLM):
    def __init__(self, global_config):
        super().__init__(global_config)
        # Disable SDK retries: gateway free-model 429 errors need special handling.
        self.openai_client.max_retries = 0
        params = self.llm_config.generate_params
        if params["model"].startswith("deepseek/deepseek-v4-"):
            params["extra_body"] = {"thinking": {"type": "disabled"}}

    @cache_response
    def infer(self, messages, **kwargs):
        for attempt in range(3):
            try:
                # Reuse pinned upstream request/metadata handling, validating before
                # the outer cache decorator records a successful response.
                text, metadata = OrcaRouterLLM.infer.__wrapped__(self, messages, **kwargs)
                if not text.strip():
                    raise ValueError("OrcaRouter 返回了空白文本，请更换模型或检查输出预算。")
                for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = metadata.get(field)
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        raise ValueError("OrcaRouter 返回了无效 usage，无法记录 token 用量。")
                return text, metadata
            except (APIConnectionError, APIStatusError) as error:
                delay = retry_delay(error, attempt)
                if delay is None:
                    raise RuntimeError(safe_error(error)) from None
                time.sleep(delay)
        raise AssertionError("unreachable")
