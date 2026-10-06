"""Embedding backends (spec sections 10 and 35).

* ``TfidfBackend`` (default, no download): stemmed word 1-2 grams plus character 3-5
  grams, fitted on the syllabus text only. Character n-grams absorb OCR errors and word
  variants ("derive"/"derivation"). Fitting on the syllabus keeps exam text, including
  future exams during backtests, out of the representation.
* ``SentenceTransformerBackend`` (optional): a pretrained local model, used only when it
  is already on disk. ``predictor models download`` fetches it explicitly.

All vectors are L2-normalised, so a dot product is a cosine similarity.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from ..config.settings import Settings
from ..preprocessing.textnorm import STOPWORDS, normalize_for_matching, stem, tokenize
from ..utils.logging import get_logger, log_event

log = get_logger("embeddings")


class EmbeddingBackend(Protocol):
    name: str
    kind: str  # "tfidf" or "neural"

    def fit(self, corpus: list[str]) -> None: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


def stem_analyzer(text: str) -> list[str]:
    toks = [stem(t) for t in tokenize(normalize_for_matching(text)) if t not in STOPWORDS and not t[0].isdigit()]
    toks = [t for t in toks if len(t) > 1]
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


class TfidfBackend:
    kind = "tfidf"

    def __init__(self, settings: Settings):
        cfg = settings.embeddings
        self.word_weight = float(cfg.tfidf_word_weight)
        self.char_weight = float(cfg.tfidf_char_weight)
        self.word = TfidfVectorizer(analyzer=stem_analyzer, sublinear_tf=True, min_df=1)
        self.char = TfidfVectorizer(analyzer="char_wb", preprocessor=normalize_for_matching,
                                    ngram_range=(int(cfg.tfidf_char_ngram_min), int(cfg.tfidf_char_ngram_max)),
                                    sublinear_tf=True, min_df=1)
        self.fitted = False
        self.name = "tfidf"

    def fit(self, corpus: list[str]) -> None:
        corpus = [c for c in corpus if c and c.strip()] or ["empty"]
        self.word.fit(corpus)
        self.char.fit(corpus)
        self.fitted = True
        digest = hashlib.sha256("\x00".join(corpus).encode("utf-8")).hexdigest()[:12]
        self.name = f"tfidf:{digest}"

    def _sparse(self, texts: list[str]) -> sparse.csr_matrix:
        if not self.fitted:
            raise RuntimeError("TfidfBackend.fit() must be called before encode()")
        w = normalize(self.word.transform(texts)) * np.sqrt(self.word_weight)
        c = normalize(self.char.transform(texts)) * np.sqrt(self.char_weight)
        return normalize(sparse.hstack([w, c]).tocsr())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return self._sparse(texts).toarray().astype(np.float32)


class SentenceTransformerBackend:
    kind = "neural"

    def __init__(self, settings: Settings, model_path: Path, cache: "EmbeddingCacheLike | None" = None):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(str(model_path), device="cpu")
        self.batch_size = int(settings.embeddings.batch_size)
        self.name = f"st:{settings.embeddings.model_name}"
        self.cache = cache

    def fit(self, corpus: list[str]) -> None:  # pretrained: nothing to fit
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        keys = [hashlib.sha256(f"{self.name}\x00{t}".encode("utf-8")).hexdigest() for t in texts]
        cached = self.cache.get_many(keys) if self.cache else {}
        missing = [i for i, k in enumerate(keys) if k not in cached]
        if missing:
            vecs = self.model.encode([texts[i] for i in missing], batch_size=self.batch_size,
                                     normalize_embeddings=True, show_progress_bar=False)
            vecs = np.asarray(vecs, dtype=np.float32)
            new = {keys[i]: vecs[j] for j, i in enumerate(missing)}
            cached.update(new)
            if self.cache:
                self.cache.put_many(self.name, new)
        return np.vstack([cached[k] for k in keys]).astype(np.float32)


class EmbeddingCacheLike(Protocol):
    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]: ...

    def put_many(self, backend: str, vectors: dict[str, np.ndarray]) -> None: ...


def model_dir(settings: Settings) -> Path:
    configured = settings.embeddings.model_dir
    base = Path(configured) if configured else Path(settings.app.data_dir) / "models"
    return base / settings.embeddings.model_name.replace("/", "__")


def neural_available(settings: Settings) -> tuple[bool, str]:
    try:
        import sentence_transformers  # noqa: F401  # type: ignore
    except Exception:
        return False, "sentence-transformers is not installed (pip install 'exam-predictor[neural]')"
    path = model_dir(settings)
    if not (path.exists() and any(path.iterdir())):
        return False, f"model not downloaded yet (run: predictor models download); expected at {path}"
    return True, ""


def get_backend(settings: Settings, cache: EmbeddingCacheLike | None = None,
                factory: Callable[[], EmbeddingBackend] | None = None) -> tuple[EmbeddingBackend, str]:
    """Return (backend, note). ``note`` explains any fallback for the analysis report."""
    if factory is not None:
        return factory(), ""
    choice = str(settings.embeddings.backend).lower()
    if choice in ("auto", "sentence-transformers", "neural"):
        ok, reason = neural_available(settings)
        if ok:
            try:
                backend = SentenceTransformerBackend(settings, model_dir(settings), cache)
                log_event(log, "embedding_backend", backend=backend.name)
                return backend, ""
            except Exception as exc:  # pragma: no cover - depends on local model files
                reason = f"failed to load the local model: {exc}"
        if choice != "auto":
            note = f"Neural embeddings requested but unavailable ({reason}); using TF-IDF instead."
            log_event(log, "embedding_fallback", reason=reason)
            return TfidfBackend(settings), note
    return TfidfBackend(settings), ""


def download_model(settings: Settings) -> Path:
    """Explicit, user-initiated download of the configured sentence-transformers model."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from sentence_transformers import SentenceTransformer  # type: ignore

    target = model_dir(settings)
    target.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(settings.embeddings.model_name, device="cpu")
    model.save(str(target))
    return target


_HASHER = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 18, alternate_sign=False,
                            norm="l2", preprocessor=normalize_for_matching)


def stateless_char_vectors(texts: list[str]) -> sparse.csr_matrix:
    """Character n-gram vectors with no fitted statistics (no leakage between exams)."""
    if not texts:
        return sparse.csr_matrix((0, 2 ** 18))
    return _HASHER.transform(texts)
