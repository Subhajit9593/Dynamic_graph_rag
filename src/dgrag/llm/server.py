"""llama-server process manager + minimal HTTP client (PLAN.md §3, D8).

Why a separate server process instead of llama-cpp-python in-process:
* llama-server runs N parallel slots with continuous batching -> ~5.5x throughput on our GPU.
* All KV-cache memory for all slots is allocated at start-up: if the server starts, it cannot OOM later.
* It is a separate process, so its CUDA runtime never clashes with torch's (PLAN.md Issue #1).

Usage:
    with LlamaServer.from_config(cfg) as server:
        client = server.client()
        obj, info = client.chat_json(messages, schema)
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dgrag.config import Config

log = logging.getLogger(__name__)


@dataclass
class ChatResult:
    content: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class LlamaClient:
    def __init__(self, base_url: str, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(self.base_url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def chat(self, messages: list[dict[str, str]], *, schema: dict | None = None, max_tokens: int = 1024,
             temperature: float = 0.0, seed: int = 42) -> ChatResult:
        payload: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                                   "seed": seed, "cache_prompt": True}
        if schema is not None:
            # grammar-constrained decoding: output is guaranteed to follow the schema (until max_tokens)
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
        t = time.perf_counter()
        r = self._post("/v1/chat/completions", payload)
        dt = time.perf_counter() - t
        ch = r["choices"][0]
        usage = r.get("usage", {})
        return ChatResult(ch["message"]["content"], ch.get("finish_reason"), usage.get("prompt_tokens", 0),
                          usage.get("completion_tokens", 0), dt)

    def chat_json(self, messages: list[dict[str, str]], schema: dict, **kw: Any) -> tuple[Any | None, ChatResult]:
        res = self.chat(messages, schema=schema, **kw)
        try:
            return json.loads(res.content), res
        except json.JSONDecodeError:
            return None, res  # e.g. cut off by max_tokens (finish_reason == "length")

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=2) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False


class LlamaServer:
    def __init__(self, server_bin: Path, model_path: Path, *, port: int = 8090, parallel: int = 16,
                 ctx_per_slot: int = 3072, n_gpu_layers: int = 99, log_path: Path | None = None,
                 startup_timeout: float = 180.0, kv_cache_type: str | None = None):
        self.server_bin = Path(server_bin)
        self.model_path = Path(model_path)
        self.port = port
        self.parallel = parallel
        self.ctx_per_slot = ctx_per_slot
        self.n_gpu_layers = 99 if n_gpu_layers < 0 else n_gpu_layers
        self.log_path = log_path
        self.startup_timeout = startup_timeout
        self.kv_cache_type = kv_cache_type  # e.g. "q8_0": halves KV-cache memory for long prompts
        self.proc: subprocess.Popen | None = None
        self._log_file = None

    @classmethod
    def from_config(cls, cfg: Config, **overrides: Any) -> LlamaServer:
        kw: dict[str, Any] = dict(
            port=cfg.llm.server_port, parallel=cfg.llm.parallel, ctx_per_slot=cfg.llm.ctx_per_slot,
            n_gpu_layers=cfg.llm.n_gpu_layers, log_path=cfg.resolve(Path("data/logs/llama_server.log")),
        )
        kw.update(overrides)
        return cls(cfg.resolve(cfg.llm.server_bin), cfg.resolve(cfg.llm.model_path), **kw)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def client(self, timeout: float = 180.0) -> LlamaClient:
        return LlamaClient(self.base_url, timeout)

    def start(self) -> None:
        if not self.server_bin.exists():
            raise FileNotFoundError(f"llama-server not found at {self.server_bin} (see PLAN.md D8)")
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found at {self.model_path}")
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(f"port {self.port} already in use (another llama-server running?)")
        cmd = [str(self.server_bin), "-m", str(self.model_path), "--host", "127.0.0.1", "--port", str(self.port),
               "-ngl", str(self.n_gpu_layers), "-c", str(self.ctx_per_slot * self.parallel),
               "--parallel", str(self.parallel), "-fa", "on", "--no-webui"]
        if self.kv_cache_type:
            cmd += ["-ctk", self.kv_cache_type, "-ctv", self.kv_cache_type]
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.log_path, "w")
        log.info("Starting llama-server: %d slots x %d ctx on port %d", self.parallel, self.ctx_per_slot, self.port)
        # start_new_session: the server gets its own process group, so a terminal Ctrl-C (SIGINT to the whole
        # foreground group) reaches only Python, which then shuts the server down in order (PLAN.md Issue #8).
        self.proc = subprocess.Popen(cmd, stdout=self._log_file or subprocess.DEVNULL, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        client = self.client()
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}; see {self.log_path}")
            if client.health():
                log.info("llama-server ready")
                return
            time.sleep(0.5)
        self.stop()
        raise TimeoutError("llama-server did not become healthy in time")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)  # stateless server: force-kill is safe
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            log.info("llama-server stopped")
        self.proc = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> LlamaServer:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()
