#!/usr/bin/env python3
"""Offline diagnostics by default; no network request or model download is made."""

import argparse
import importlib
import os
import platform
import sys
from importlib.metadata import version

from demo_core import DEFAULT_DOCUMENTS, ROOT, load_documents


def main(argv=None):
    parser = argparse.ArgumentParser(description="检查依赖、文档与配置，不调用 API")
    parser.add_argument("--offline", action="store_true", help="不要求配置 API 密钥，适合安装检查和 CI")
    args = parser.parse_args(argv)
    # LiteLLM otherwise fetches its price map while HippoRAG is imported.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    errors = []
    print(f"Python {platform.python_version()} / {platform.platform()}")
    if not (3, 10) <= sys.version_info < (3, 13):
        errors.append("需要 Python 3.10–3.12")
    for module, distribution in [
        ("torch", "torch"),
        ("hipporag", "hipporag"),
        ("dotenv", "python-dotenv"),
        ("psutil", "psutil"),
    ]:
        try:
            importlib.import_module(module)
            print(f"✓ {distribution} {version(distribution)}")
        except Exception as error:
            errors.append(f"{distribution} 导入失败：{error}")
    try:
        from backend import select_device

        print(f"✓ 可用嵌入设备：{select_device('auto')}")
        print(f"✓ 示例文档：{len(load_documents(DEFAULT_DOCUMENTS))} 篇")
    except Exception as error:
        errors.append(str(error))
    if not args.offline:
        try:
            from dotenv import load_dotenv

            from backend import validate_key

            load_dotenv(ROOT / ".env", override=False)
            validate_key()
            print("✓ DEEPSEEK_API_KEY 已配置（未验证 API 有效性）")
        except Exception as error:
            errors.append(str(error))
    for error in errors:
        print(f"✗ {error}")
    if errors:
        print("检查未通过；请参考 README 或重新运行 setup_env.sh")
        return 1
    print("离线检查通过；未下载嵌入模型，也未调用 DeepSeek API。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
