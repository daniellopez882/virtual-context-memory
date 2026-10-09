"""Checks against the downloaded LoCoMo file and MiniLM model (``pytest -m integration``)."""

from __future__ import annotations

import numpy as np
import pytest

from vcmem.assets import ASSETS, sha256_of
from vcmem.config import data_dir, locomo_path, minilm_dir
from vcmem.embed import MiniLMEmbedder
from vcmem.locomo import load_locomo
from vcmem.tokens import WordPieceTokenCounter

pytestmark = pytest.mark.integration

needs_data = pytest.mark.skipif(not locomo_path().exists(), reason="run `vcmem fetch` first")
needs_model = pytest.mark.skipif(
    not (minilm_dir() / "model_quantized.onnx").exists(), reason="run `vcmem fetch` first"
)


@needs_data
@needs_model
def test_downloaded_assets_match_pinned_checksums() -> None:
    for asset in ASSETS:
        assert sha256_of(data_dir() / asset.relpath) == asset.sha256


@needs_data
def test_locomo_release_shape() -> None:
    convs = load_locomo(locomo_path())
    assert len(convs) == 10
    assert sum(len(c.messages()) for c in convs) == 5882
    assert sum(len(c.qa) for c in convs) == 1986
    assert sum(c.dropped_evidence for c in convs) == 2
    assert sum(1 for c in convs for q in c.qa if not q.evidence) == 4


@needs_model
def test_minilm_embeddings_are_normalised_and_semantic() -> None:
    emb = MiniLMEmbedder(minilm_dir())
    vecs = emb.embed(
        ["A man is eating food.", "A man is eating a piece of bread.", "A man rides a horse."]
    )
    assert vecs.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert vecs[0] @ vecs[1] > vecs[0] @ vecs[2]
    # with the default batch size a text's vector does not depend on its neighbours
    single = emb.embed(["A man rides a horse."])[0]
    assert np.allclose(single, vecs[2], atol=1e-5)


@needs_model
def test_wordpiece_counter_is_additive_and_truncates() -> None:
    counter = WordPieceTokenCounter(minilm_dir() / "tokenizer.json")
    a, b = "Caroline (8 May 2023): I went to an LGBTQ support group!", "Melanie: Wow, amazing."
    assert counter.count(a + "\n" + b) == counter.count(a) + counter.count(b)
    for n in range(0, 20):
        assert counter.count(counter.truncate(a, n)) <= n
