"""Core memory: small named blocks that are always in context (MemGPT's persona/human)."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from vcmem.stores.sqlite import Database
    from vcmem.tokens import TokenCounter


class CoreMemoryError(ValueError):
    """Raised for a full block, an unknown block, or a replace target that is absent."""


@dataclass(slots=True)
class CoreLine:
    text: str
    sources: frozenset[str] = field(default_factory=frozenset)


@dataclass(slots=True)
class CoreBlock:
    name: str
    limit: int
    lines: list[CoreLine] = field(default_factory=list)

    def render(self) -> str:
        body = "\n".join(line.text for line in self.lines)
        return f"[{self.name}]\n{body}" if body else f"[{self.name}]"

    def sources(self) -> frozenset[str]:
        out: set[str] = set()
        for line in self.lines:
            out |= line.sources
        return frozenset(out)


class CoreMemory:
    def __init__(self, db: Database, counter: TokenCounter, limits: Mapping[str, int]) -> None:
        self._db = db
        self._counter = counter
        self.blocks: dict[str, CoreBlock] = {}
        with db.lock:
            rows = db.conn.execute("SELECT name, limit_tokens, lines FROM core_blocks").fetchall()
        for name, limit, lines in rows:
            parsed = [CoreLine(t, frozenset(s)) for t, s in json.loads(lines)]
            self.blocks[name] = CoreBlock(name, int(limit), parsed)
        for name, limit in limits.items():
            if name not in self.blocks:
                self.blocks[name] = CoreBlock(name, limit)
                self._save(name)

    def _block(self, name: str) -> CoreBlock:
        try:
            return self.blocks[name]
        except KeyError:
            raise CoreMemoryError(
                f"unknown block {name!r}; blocks are {sorted(self.blocks)}"
            ) from None

    def _save(self, name: str) -> None:
        block = self.blocks[name]
        payload = json.dumps([[line.text, sorted(line.sources)] for line in block.lines])
        with self._db.lock:
            self._db.conn.execute(
                "INSERT INTO core_blocks(name, limit_tokens, lines) VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET limit_tokens = excluded.limit_tokens, "
                "lines = excluded.lines",
                (name, block.limit, payload),
            )
            self._db.conn.commit()

    def append(self, name: str, content: str, sources: Iterable[str] = ()) -> None:
        block = self._block(name)
        line = CoreLine(content, frozenset(sources))
        candidate = CoreBlock(name, block.limit, [*block.lines, line])
        used = self._counter.count(candidate.render())
        if used > block.limit:
            raise CoreMemoryError(
                f"block {name!r} would hold {used} tokens, limit is {block.limit}; "
                "use core_memory_replace to make room"
            )
        block.lines = candidate.lines
        self._save(name)

    def replace(self, name: str, old: str, new: str, sources: Iterable[str] = ()) -> None:
        block = self._block(name)
        for i, line in enumerate(block.lines):
            if old and old in line.text:
                updated = line.text.replace(old, new, 1).strip()
                lines = list(block.lines)
                if updated:
                    lines[i] = CoreLine(updated, line.sources | frozenset(sources))
                else:
                    del lines[i]
                candidate = CoreBlock(name, block.limit, lines)
                used = self._counter.count(candidate.render())
                if used > block.limit:
                    raise CoreMemoryError(
                        f"block {name!r} would hold {used} tokens, limit is {block.limit}"
                    )
                block.lines = lines
                self._save(name)
                return
        raise CoreMemoryError(f"{old!r} not found in block {name!r}")

    def snapshot(self) -> dict[str, CoreBlock]:
        return copy.deepcopy(self.blocks)

    def restore(self, blocks: dict[str, CoreBlock]) -> None:
        # In-memory only: the replay benchmark rewinds between questions, which never
        # write core memory, so the persisted rows are already identical.
        self.blocks = copy.deepcopy(blocks)
