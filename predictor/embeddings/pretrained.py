"""Pretrained local semantic embeddings that work at every dataset size.

``PretrainedBackend`` uses the WordLlama static token embeddings (256 dimensions, MIT
licence) that ship inside the ``wordllama`` pip package. The weights are read straight from
the installed package, so nothing is downloaded and nothing leaves the machine. A sentence
vector is the average of its token vectors, L2-normalised. The model is general English
knowledge distilled from large language models: it knows that "capillary rise" and "liquid
climbing a thin tube" are related even when a course has only one past paper.

``HybridBackend`` concatenates the pretrained vector with the syllabus-fitted TF-IDF vector
(both unit length, weighted by ``a`` and ``1 - a``), so its cosine is exactly
``a * pretrained + (1 - a) * tfidf``. With ``a = 0.3`` the TF-IDF alignment thresholds still
separate in-syllabus from out-of-syllabus questions (measured on the demo course and a set of
paraphrased questions; see docs/LOW_DATA_INFERENCE.md).

Neither backend fits anything on exam text, so similarities for exam ``t`` never depend on
later exams.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config.settings import Settings
from ..preprocessing.textnorm import clean_text, normalize_math
from ..utils.logging import get_logger, log_event

log = get_logger("embeddings")

WEIGHTS = ("weights", "l2_supercat_256.safetensors")
TOKENIZER = ("tokenizers", "l2_supercat_tokenizer_config.json")
MODEL_NAME = "wordllama-l2-supercat-256"


def _package_root() -> Path | None:
    # find_spec does not import the package (its __init__ reconfigures the root logger).
    try:
        spec = importlib.util.find_spec("wordllama")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def pretrained_available() -> tuple[bool, str]:
    root = _package_root()
    if root is None:
        return False, "the 'wordllama' package is not installed (pip install wordllama)"
    for parts in (WEIGHTS, TOKENIZER):
        if not root.joinpath(*parts).exists():
            return False, f"the bundled WordLlama file {'/'.join(parts)} is missing"
    try:
        import safetensors  # noqa: F401
        import tokenizers  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the installation
        return False, f"a WordLlama dependency is missing ({exc})"
    return True, ""


@lru_cache(maxsize=1)
def _load():
    from safetensors.numpy import load_file
    from tokenizers import Tokenizer

    root = _package_root()
    tok = Tokenizer.from_file(str(root.joinpath(*TOKENIZER)))
    tok.no_padding()
    tok.no_truncation()
    emb = load_file(str(root.joinpath(*WEIGHTS)))["embedding.weight"].astype(np.float32)
    log_event(log, "pretrained_loaded", model=MODEL_NAME, vocab=int(emb.shape[0]), dim=int(emb.shape[1]))
    return tok, np.ascontiguousarray(emb)


def _prepare(text: str) -> str:
    return normalize_math(clean_text(text or "")).strip()


class PretrainedBackend:
    """Average-pooled pretrained token embeddings. Stateless: ``fit`` does nothing."""

    kind = "pretrained"
    name = f"pretrained:{MODEL_NAME}"
    dim = 256

    def __init__(self, cache_size: int = 20000):
        self._cache: dict[str, np.ndarray] = {}
        self._cache_size = cache_size

    def fit(self, corpus: list[str]) -> None:
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prepared = [_prepare(t) for t in texts]
        missing = sorted({p for p in prepared if p not in self._cache})
        if missing:
            tok, emb = _load()
            for enc, text in zip(tok.encode_batch(missing, add_special_tokens=False), missing):
                ids = np.clip(np.asarray(enc.ids, dtype=np.int64), 0, emb.shape[0] - 1)
                vec = emb[ids].mean(axis=0) if ids.size else np.zeros(self.dim, dtype=np.float32)
                norm = float(np.linalg.norm(vec))
                self._cache[text] = (vec / norm if norm > 0 else vec).astype(np.float32)
            if len(self._cache) > self._cache_size:
                for key in list(self._cache)[: len(self._cache) - self._cache_size]:
                    self._cache.pop(key, None)
        return np.vstack([self._cache[p] for p in prepared]).astype(np.float32)


class HybridBackend:
    """Pretrained semantics plus syllabus-fitted TF-IDF (used for syllabus alignment)."""

    kind = "hybrid"

    def __init__(self, settings: Settings, pretrained: PretrainedBackend | None = None):
        from .backends import TfidfBackend

        self.pretrained = pretrained or PretrainedBackend()
        self.tfidf = TfidfBackend(settings)
        self.a = float(settings.embeddings.hybrid_pretrained_weight)
        self.name = f"hybrid:{MODEL_NAME}"

    def fit(self, corpus: list[str]) -> None:
        self.tfidf.fit(corpus)
        self.name = f"hybrid:{MODEL_NAME}:{self.tfidf.name}"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        p = self.pretrained.encode(texts)
        t = self.tfidf.encode(texts)
        x = np.hstack([np.sqrt(self.a) * p, np.sqrt(1.0 - self.a) * t])
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return (x / np.where(n == 0, 1.0, n)).astype(np.float32)
