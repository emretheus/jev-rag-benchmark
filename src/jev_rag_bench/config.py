from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 13,
    "paths": {
        "data_dir": "data",
        "results_dir": "results",
        "reports_dir": "reports",
    },
    "retrieval": {
        "candidate_depth": 20,
        "bm25_depth": 100,
        "dense_depth": 100,
        "rrf_k": 60,
        "bm25": {"k1": 1.5, "b": 0.75},
    },
    "models": {
        "embedding": {
            "provider": "nvidia",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key_env": "NVIDIA_API_KEY",
            "name": "nvidia/nemotron-3-embed-1b",
            "input_type": "query_passage",
            "batch_size": 32,
            "requests_per_minute": 40,
        },
        "reranker_nvidia": {
            "base_url": "https://ai.api.nvidia.com/v1",
            "api_key_env": "NVIDIA_API_KEY",
            "name": "nvidia/llama-nemotron-rerank-vl-1b-v2",
            "requests_per_minute": 40,
        },
        "systemone": {
            "provider": "vercel",
            "base_url": "https://ai-gateway.vercel.sh/typesafe",
            "decisions_path": "/v1/systemone",
            "api_key_env": "AI_GATEWAY_API_KEY",
            "model": "typesafe-ai/jev",
            "instruction_template": (
                "Passage [{index}] contains information that is needed to answer the question"
            ),
            "max_passage_chars": 4000,
            "max_state_chars": 200000,
            "requests_per_minute": 60,
        },
        "generator": {
            "provider": "codiv",
            "base_url": "https://api.codiv.ai/v1",
            "api_key_env": "TYPESAFE_API_KEY",
            "name": "diffusiongemma-26b",
            "max_tokens": 400,
            "temperature": 0.0,
            "requests_per_minute": 600,
        },
    },
    "generation": {"top_k": 5, "max_context_chars": 6000},
    "pricing": {
        "jev_usd_per_mtok": 0.042,
        "other_usd_per_mtok": 0.0,
    },
    "safety": {
        "max_systemone_requests": 5000,
        "max_systemone_input_tokens": 16000000,
    },
}


def default_config() -> dict[str, Any]:
    return deep_merge({}, DEFAULT_CONFIG)


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    cfg = default_config()
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    elif Path("configs/default.yaml").exists():
        candidates.append(Path("configs/default.yaml"))
    for candidate in candidates:
        if not candidate.exists():
            raise FileNotFoundError(f"config not found: {candidate}")
        with open(candidate, encoding="utf-8") as handle:
            user = yaml.safe_load(handle) or {}
        cfg = deep_merge(cfg, user)
    return cfg


def api_key(section: dict[str, Any]) -> str | None:
    env_name = section.get("api_key_env")
    if not env_name:
        return None
    return os.environ.get(str(env_name))


def load_dotenv(path: str | os.PathLike[str] = ".env") -> None:
    """Load KEY=VALUE pairs from a .env file without overriding real environment vars."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


def data_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["data_dir"])


def results_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["results_dir"])


def reports_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["reports_dir"])
