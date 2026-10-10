"""Sentence embedders.

``MiniLMEmbedder`` runs the quantised all-MiniLM-L6-v2 ONNX export with onnxruntime and
the Hugging Face ``tokenizers`` runtime (mean pooling, L2 normalisation, as in
sentence-transformers). ``HashingEmbedder`` is a dependency-free signed feature-hashing
embedder: it has no semantics beyond shared words, and exists so tests, the demo and the
Docker smoke run work without downloading a model.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32 array of L2-normalised rows."""
        ...


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = _WORD.findall(text.lower())
            features = words + [f"{a}_{b}" for a, b in itertools.pairwise(words)]
            for feature in features:
                col, sign = self._bucket(feature)
                out[row, col] += sign
        return _normalise(out)


class MiniLMEmbedder:
    dim = 384

    def __init__(
        self,
        model_dir: Path,
        max_length: int = 256,
        batch_size: int = 1,
        threads: int | None = None,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # The exported tokenizer pads every input to 128; padding to the longest item in
        # each batch is cheaper, and 256 matches the sentence-transformers config.
        self._tok.enable_truncation(max_length=max_length)
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or os.cpu_count() or 1
        self._session = ort.InferenceSession(
            str(model_dir / "model_quantized.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._inputs = {i.name for i in self._session.get_inputs()}
        # The quantised export computes activation scales per input tensor, so a text's
        # vector shifts with whatever else shares its batch (a few hundredths of cosine
        # on LoCoMo). One text per call keeps vectors reproducible and was not slower on
        # CPU; larger batches remain available when that drift is acceptable.
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        # Length-sorted batches waste less compute on padding.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            encs = self._tok.encode_batch([texts[i] for i in idx])
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, feeds)[0]
            weights = mask[:, :, None].astype(np.float32)
            pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
            out[idx] = pooled
        return _normalise(out)


class CachedEmbedder:
    """Serves precomputed vectors and embeds misses with the wrapped embedder.

    The replay benchmark embeds every message once up front so that per-turn latency
    measures the memory manager, not the encoder (encoder cost is reported separately).
    """

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self.dim = inner.dim
        self._cache: dict[str, np.ndarray] = {}
        self.misses = 0
        # Texts asked for through embed(), cached or not: what a live deployment would
        # have to encode.
        self.requested = 0

    def prime(self, texts: Sequence[str]) -> int:
        """Embed the texts not cached yet, in one batch call; return how many that was."""
        todo = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if todo:
            for text, vec in zip(todo, self._inner.embed(todo), strict=True):
                self._cache[text] = vec
        return len(todo)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.requested += len(texts)
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            self.misses += len(missing)
            self.prime(missing)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._cache[t] for t in texts])
