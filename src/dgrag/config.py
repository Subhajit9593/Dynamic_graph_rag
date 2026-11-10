"""Typed configuration loaded from config.yaml.

Relative paths in the YAML are resolved against the project root (the directory containing config.yaml),
so commands behave the same regardless of the current working directory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class PathsConfig:
    raw_dir: Path = Path("data/raw/multihoprag")
    db_path: Path = Path("data/db/dgrag.sqlite")
    faiss_dir: Path = Path("data/faiss")
    results_dir: Path = Path("results")


@dataclass
class DatasetConfig:
    hf_repo: str = "yixuantt/MultiHopRAG"
    corpus_file: str = "corpus.json"
    questions_file: str = "MultiHopRAG.json"


@dataclass
class LLMConfig:
    model_path: Path = Path("models/qwen2.5-7b-instruct-q4_k_m/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf")
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    temperature: float = 0.0
    seed: int = 42
    server_bin: Path = Path("third_party/llama.cpp/build/bin/llama-server")
    server_port: int = 8090
    parallel: int = 16
    ctx_per_slot: int = 3072
    max_tokens: int = 1024


@dataclass
class EmbedderConfig:
    model_name: str = "BAAI/bge-base-en-v1.5"
    dim: int = 768
    device: str = "cuda"
    batch_size: int = 64


@dataclass
class ChunkingConfig:
    chunk_tokens: int = 400
    overlap_tokens: int = 50
    min_chunk_tokens: int = 40


@dataclass
class ExtractionConfig:
    prompt_version: str = "v1"
    max_entities: int = 20
    max_relations: int = 20


@dataclass
class ResolutionConfig:
    tau_high: float = 0.90
    tau_low: float = 0.80
    top_k: int = 5


@dataclass
class CommunityConfig:
    staleness_threshold: float = 0.2
    superseded_weight: float = 3.0
    drift_threshold: float = 0.05


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    community: CommunityConfig = field(default_factory=CommunityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    root: Path = PROJECT_ROOT

    def resolve(self, p: Path) -> Path:
        """Absolute path for a (possibly relative) config path."""
        return p if p.is_absolute() else (self.root / p)


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    """Recursively build a dataclass from a dict, rejecting unknown keys (catches typos in config.yaml)."""
    data = data or {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    defaults = cls()
    for name, value in data.items():
        default_value = getattr(defaults, name)
        if is_dataclass(default_value):
            kwargs[name] = _build(type(default_value), value)
        elif isinstance(default_value, Path):
            kwargs[name] = Path(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    cfg = _build(Config, raw)
    cfg.root = path.resolve().parent if path.exists() else PROJECT_ROOT
    return cfg


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # faiss probes for a separate AVX2 .so that modern wheels no longer ship; the single library already
    # uses runtime CPU dispatch (compile options "DD AVX2"), so the INFO message is noise (PLAN.md Issue #5).
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)
    for noisy in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
