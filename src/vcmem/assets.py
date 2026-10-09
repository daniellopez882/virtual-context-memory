"""Pinned downloads: the LoCoMo release and the quantised MiniLM ONNX export.

URLs point at fixed commits. The sha256 values were recorded on the first download and
every fetch is verified against them; a mismatch deletes the file and fails loudly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

LOCOMO_COMMIT = "cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc"
MINILM_REVISION = "751bff37182d3f1213fa05d7196b954e230abad9"


@dataclass(frozen=True, slots=True)
class Asset:
    name: str
    relpath: str
    url: str
    sha256: str
    size: int
    license: str


ASSETS: tuple[Asset, ...] = (
    Asset(
        name="locomo10",
        relpath="locomo/locomo10.json",
        url=(
            "https://raw.githubusercontent.com/snap-research/locomo/"
            f"{LOCOMO_COMMIT}/data/locomo10.json"
        ),
        sha256="79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
        size=2_805_274,
        license="CC BY-NC 4.0",
    ),
    Asset(
        name="minilm-onnx-quantized",
        relpath="models/all-MiniLM-L6-v2/model_quantized.onnx",
        url=(
            "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/"
            f"{MINILM_REVISION}/onnx/model_quantized.onnx"
        ),
        sha256="afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1",
        size=22_972_370,
        license="Apache-2.0",
    ),
    Asset(
        name="minilm-tokenizer",
        relpath="models/all-MiniLM-L6-v2/tokenizer.json",
        url=(
            "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/"
            f"{MINILM_REVISION}/tokenizer.json"
        ),
        sha256="da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
        size=711_661,
        license="Apache-2.0",
    ),
)


class ChecksumError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(
    dest: Path,
    assets: Sequence[Asset] = ASSETS,
    client: httpx.Client | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Path]:
    """Download missing assets into ``dest`` and verify all of them; return their paths."""
    own_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=120.0)
    paths: dict[str, Path] = {}
    try:
        for asset in assets:
            path = dest / asset.relpath
            if path.exists() and sha256_of(path) == asset.sha256:
                log(f"ok        {asset.name} ({path})")
                paths[asset.name] = path
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".part")
            log(f"download  {asset.name} <- {asset.url}")
            with http.stream("GET", asset.url) as response:
                response.raise_for_status()
                with partial.open("wb") as fh:
                    for chunk in response.iter_bytes(1 << 20):
                        fh.write(chunk)
            got = sha256_of(partial)
            if got != asset.sha256:
                partial.unlink()
                raise ChecksumError(f"{asset.name}: sha256 {got} != expected {asset.sha256}")
            partial.replace(path)
            log(f"verified  {asset.name} ({path})")
            paths[asset.name] = path
    finally:
        if own_client:
            http.close()
    return paths
