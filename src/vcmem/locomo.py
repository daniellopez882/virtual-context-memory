"""Loader for LoCoMo (Maharana et al., ACL 2024), ``locomo10.json``.

Ten two-person conversations of 19 to 32 sessions. Each QA item lists evidence dialog ids
such as ``D3:12`` (session 3, turn 12). A few evidence strings are malformed in the
release (``"D8:6; D9:17"``, ``"D:11:26"``); they are split and normalised here, and ids that
do not exist in the conversation are dropped and counted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vcmem.messages import Message

if TYPE_CHECKING:
    from pathlib import Path

# Integer codes in the file; names as used by follow-up work (e.g. the Mem0 paper).
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

_EVIDENCE = re.compile(r"D:?(\d+):(\d+)")
_DATE = re.compile(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")
_SESSION_KEY = re.compile(r"session_(\d+)")


@dataclass(frozen=True, slots=True)
class QAItem:
    index: int
    question: str
    answer: str
    category: int
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Session:
    index: int
    date: str
    messages: tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class Conversation:
    sample_id: str
    speakers: tuple[str, str]
    sessions: tuple[Session, ...]
    qa: tuple[QAItem, ...]
    dropped_evidence: int

    def messages(self) -> list[Message]:
        return [m for s in self.sessions for m in s.messages]


def parse_evidence(raw: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    for item in raw:
        for session, turn in _EVIDENCE.findall(str(item)):
            ref = f"D{int(session)}:{int(turn)}"
            if ref not in out:
                out.append(ref)
    return tuple(out)


def short_date(raw: str) -> str:
    """``"1:56 pm on 8 May, 2023"`` -> ``"8 May 2023"``; unknown formats pass through."""
    match = _DATE.search(raw)
    if match is None:
        return raw.strip()
    day, month, year = match.groups()
    return f"{int(day)} {month} {year}"


def _turn_text(turn: dict[str, object]) -> str:
    text = str(turn.get("text", "")).strip()
    caption = turn.get("blip_caption")
    if caption:
        text = f"{text} [shared image: {caption}]"
    return text


def parse_conversation(sample: dict[str, object]) -> Conversation:
    conv = sample["conversation"]
    if not isinstance(conv, dict):
        raise ValueError("conversation must be an object")
    session_numbers = sorted(
        int(m.group(1)) for k in conv if (m := _SESSION_KEY.fullmatch(k)) is not None
    )
    sessions: list[Session] = []
    known: set[str] = set()
    for n in session_numbers:
        date = short_date(str(conv.get(f"session_{n}_date_time", "")))
        messages = []
        for turn in conv[f"session_{n}"]:
            mid = str(turn["dia_id"])
            known.add(mid)
            messages.append(
                Message(
                    id=mid,
                    text=_turn_text(turn),
                    speaker=str(turn["speaker"]),
                    role="user",
                    timestamp=date,
                    session=n,
                )
            )
        sessions.append(Session(n, date, tuple(messages)))
    qa: list[QAItem] = []
    dropped = 0
    raw_qa = sample.get("qa", [])
    if not isinstance(raw_qa, list):
        raise ValueError("qa must be a list")
    for i, item in enumerate(raw_qa):
        evidence = parse_evidence(item.get("evidence", []))
        kept = tuple(e for e in evidence if e in known)
        dropped += len(evidence) - len(kept)
        answer = item.get("answer", item.get("adversarial_answer", ""))
        qa.append(QAItem(i, str(item["question"]), str(answer), int(item["category"]), kept))
    return Conversation(
        sample_id=str(sample.get("sample_id", "")),
        speakers=(str(conv["speaker_a"]), str(conv["speaker_b"])),
        sessions=tuple(sessions),
        qa=tuple(qa),
        dropped_evidence=dropped,
    )


def load_locomo(path: Path) -> list[Conversation]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [parse_conversation(sample) for sample in data]
