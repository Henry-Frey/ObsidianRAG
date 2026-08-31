"""Embedding client for llama-server. Stdlib only -- no `requests`, no SDK.

llama-server speaks an OpenAI-compatible API on loopback, so this is ~100 lines
of urllib. We target /v1/embeddings rather than llama.cpp's native /embedding
because the OpenAI shape has stayed stable across llama.cpp releases and takes
a batch of inputs in one call, while the native endpoint's response shape has
changed more than once.

The whole network surface of this project is the two functions below and the
hard-coded 127.0.0.1 hosts in config.py.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import PurePath

import numpy as np

from . import config


class ServerError(RuntimeError):
    pass


class ModelLoading(ServerError):
    """The server is up, but the weights are not in memory yet.

    A distinct type because it is not a misconfiguration and needs no fix --
    it needs a wait. A 22 GB GGUF off a cold page cache can take minutes, and
    reporting that as FAILED sends you looking for a problem that does not
    exist.
    """


def http_error(url: str, code: int, body: str) -> ServerError:
    """Map an HTTP failure onto the most specific exception we have."""
    if code == 503 and "loading" in body.lower():
        return ModelLoading(
            f"{url} -> the server is running but is still loading the model.\n"
            f"Large GGUFs take a while on a cold page cache. Wait and retry, or\n"
            f"run  index.py --doctor --wait 600  to poll until it is ready.")
    if code in (404, 501) or "embedding" in body.lower():
        return ServerError(
            f"{url} -> HTTP {code}: {body}\n"
            f"If this is the embedding server, it must be started with "
            f"--embedding (and --pooling cls for BGE models).")
    return ServerError(f"{url} -> HTTP {code}: {body}")


def wait_until_ready(host: str, timeout: int = 600, on_wait=None) -> dict:
    """Poll /props until the model is loaded. Returns the props payload."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return props(host)
        except ModelLoading:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            if on_wait:
                on_wait(int(remaining))
            time.sleep(3)


def _request(url: str, payload: dict | None, timeout: int | None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout or config.REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise http_error(url, exc.code, body) from exc
    except urllib.error.URLError as exc:
        raise ServerError(
            f"Cannot reach llama-server at {url} ({exc.reason}).\n"
            f"Start it, or check EMBED_HOST / CHAT_HOST in config.py."
        ) from exc


def props(host: str | None = None) -> dict:
    """GET /props -- what the server has loaded."""
    host = host or config.EMBED_HOST
    return _request(f"{host}/props", None, timeout=30)


def model_id(host: str | None = None) -> str:
    """A stable cache key naming the model this server actually loaded.

    Deriving it from the server rather than from a constant means swapping the
    GGUF invalidates cached vectors automatically. Mixing vectors from two
    embedding models in one index does not raise anything -- it just makes
    retrieval quietly and inexplicably worse, which is far harder to notice.
    """
    if config.EMBED_MODEL_ID:
        return config.EMBED_MODEL_ID
    try:
        info = props(host)
    except ServerError:
        return "unknown"
    for key in ("model_path", "model"):
        value = info.get(key)
        if isinstance(value, str) and value:
            return PurePath(value).name
    settings = info.get("default_generation_settings") or {}
    value = settings.get("model") if isinstance(settings, dict) else None
    return PurePath(value).name if isinstance(value, str) and value else "unknown"


def check(host: str | None = None) -> dict:
    """Fail early and legibly, and report what we found."""
    host = host or config.EMBED_HOST
    info = props(host)
    probe = embed(["probe"], host=host)
    return {
        "host": host,
        "model": model_id(host),
        "dim": int(probe.shape[1]),
        "n_ctx": info.get("n_ctx") or (info.get("default_generation_settings") or {}).get("n_ctx"),
    }


def embed(texts: list[str], host: str | None = None) -> np.ndarray:
    """Embed a batch of texts. Returns an L2-normalised float32 array (n, dim).

    Normalising here means cosine similarity is a plain dot product everywhere
    downstream, and the stored vectors are already in the form search uses.
    We normalise regardless of what the server does, so this stays correct
    whether or not the model applies its own normalisation.
    """
    if not texts:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)

    host = host or config.EMBED_HOST
    out = _request(f"{host}/v1/embeddings", {"input": texts, "model": "embedding"}, None)

    rows = out.get("data")
    if not rows:
        raise ServerError(f"No embeddings returned for {len(texts)} texts: {str(out)[:300]}")
    if len(rows) != len(texts):
        raise ServerError(f"Asked for {len(texts)} embeddings, got {len(rows)}.")

    # Sort by index: the spec allows any order, and llama.cpp has returned
    # out-of-order batches. Silently mispairing vectors with chunks would be
    # invisible until retrieval started returning nonsense.
    rows = sorted(rows, key=lambda r: r.get("index", 0))
    vectors = [r["embedding"] for r in rows]

    # Some builds wrap each embedding in an extra list (one row per token
    # position when pooling is off). Unwrap a single-element nesting.
    vectors = [v[0] if v and isinstance(v[0], list) and len(v) == 1 else v for v in vectors]

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ServerError(
            f"Expected 2-D embeddings, got shape {arr.shape}. "
            f"Start the server with --pooling cls so each input yields one vector."
        )
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_batched(texts: list[str], host: str | None = None, progress=None) -> np.ndarray:
    """Embed in batches so one huge request cannot time out the whole run."""
    chunks = []
    for start in range(0, len(texts), config.EMBED_BATCH):
        batch = texts[start:start + config.EMBED_BATCH]
        chunks.append(embed(batch, host=host))
        if progress:
            progress(min(start + len(batch), len(texts)), len(texts))
    if not chunks:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)
    return np.vstack(chunks)
