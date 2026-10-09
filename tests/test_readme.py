"""The README's Python examples must run as written (the LLM example needs a server)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).parent.parent / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.S)
OFFLINE = [b for b in BLOCKS if "OpenAICompatibleProvider(" not in b]


def test_readme_has_examples() -> None:
    assert len(OFFLINE) >= 2


@pytest.mark.parametrize("code", OFFLINE, ids=[f"block{i}" for i in range(len(OFFLINE))])
def test_readme_example_runs(code: str, capsys: pytest.CaptureFixture[str]) -> None:
    exec(compile(code, str(README), "exec"), {"__name__": "readme_example"})
    out = capsys.readouterr().out
    if "print(page.paged_in)" in code:
        assert "True False" in out
        assert "['t0']" in out
