"""Tests never download models or make live API requests."""

import os
import tempfile

_cache = tempfile.TemporaryDirectory(prefix="hipporag-tests-hf-")
os.environ["HF_HOME"] = _cache.name
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
