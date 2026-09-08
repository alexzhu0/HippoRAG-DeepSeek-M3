#!/usr/bin/env bash
# Reusable project-local installation, without altering system Python or Conda.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info < (3, 13) else "需要 Python 3.10–3.12，建议 3.11；可设置 PYTHON_BIN=python3.11")'
command -v git >/dev/null || { echo "需要先安装 Git" >&2; exit 1; }
if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
    "$python_bin" -m venv "$project_dir/.venv"
fi
venv_python="$project_dir/.venv/bin/python"
"$venv_python" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info < (3, 13) else "现有 .venv 的 Python 版本不受支持，请改名保存后重新运行安装")'
"$venv_python" -m pip install --upgrade pip
"$venv_python" -m pip install -r "$project_dir/requirements.txt"
"$venv_python" -m pip check
if [[ ! -f "$project_dir/.env" ]]; then
    cp "$project_dir/.env.template" "$project_dir/.env"
    chmod 600 "$project_dir/.env"
fi
"$venv_python" "$project_dir/test_env.py" --offline
printf '\n安装完成。请填写项目 .env 中的 DEEPSEEK_API_KEY，然后运行：\n'
printf 'source "%s/.venv/bin/activate"\n' "$project_dir"
printf 'python "%s/advanced_app.py"\n' "$project_dir"
