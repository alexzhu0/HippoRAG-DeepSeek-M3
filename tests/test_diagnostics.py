import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_offline_diagnostics_work_outside_project(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "test_env.py"), "--offline"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "离线检查通过" in result.stdout


def test_missing_key_is_a_nonzero_diagnostic(tmp_path):
    env = {**os.environ, "DEEPSEEK_API_KEY": ""}
    result = subprocess.run(
        [sys.executable, str(ROOT / "test_env.py")], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "DEEPSEEK_API_KEY" in result.stdout


def test_model_listing_needs_no_hipporag_import(tmp_path):
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "list_models.py")], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "facebook/contriever" in result.stdout


@pytest.mark.parametrize("entry", ["app.py", "advanced_app.py"])
def test_help_outside_project_does_not_load_models(tmp_path, entry):
    result = subprocess.run(
        [sys.executable, str(ROOT / entry), "--help"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--batch-size" in result.stdout


def test_imports_need_no_dependencies_or_credentials(tmp_path):
    code = f"import sys; sys.path.insert(0, {str(ROOT)!r}); import app, advanced_app; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-S", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_diagnostics_do_not_fetch_provider_metadata(tmp_path):
    env = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": ""}
    code = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import httpx
calls = []
def blocked(*args, **kwargs):
    calls.append(args)
    raise RuntimeError('network disabled for test')
httpx.get = blocked
from test_env import main
result = main(['--offline'])
assert not calls, 'diagnostics attempted to fetch provider metadata'
sys.exit(result)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_orca_diagnostics_only_require_selected_key(tmp_path):
    env = {
        **os.environ,
        "LLM_PROVIDER": "orcarouter",
        "ORCAROUTER_API_KEY": "test-orca",
        "ORCAROUTER_MODEL": "qwen/test",
        "DEEPSEEK_API_KEY": "",
    }
    result = subprocess.run(
        [sys.executable, str(ROOT / "test_env.py"), "--provider", "orcarouter"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "orcarouter / qwen/test" in result.stdout
    assert "test-orca" not in result.stdout
