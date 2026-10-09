"""A self-contained run on a short synthetic chat, for the quick start and the Docker smoke run.

No downloads: regex token counts and the hashing embedder. The chat is written for this
demo; it is not LoCoMo data.
"""

from __future__ import annotations

from dataclasses import dataclass

from vcmem.agent import HeartbeatAgent
from vcmem.build import build_manager
from vcmem.messages import Message
from vcmem.policies import policy_names
from vcmem.scoring import HeuristicImportance
from vcmem.scripted import ScriptedMemoryAgent

_SESSIONS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "3 March 2024",
        (
            ("Ana", "Hi Ben! Long time no see."),
            ("Ben", "Hey Ana, good to hear from you. How have you been?"),
            ("Ana", "I adopted a greyhound called Pixel from the shelter last week."),
            ("Ben", "That is lovely. Greyhounds are so calm indoors."),
            ("Ana", "She sleeps most of the day, then sprints around the park at dusk."),
            ("Ben", "I started training for the Lisbon half marathon in May."),
            ("Ana", "Wow, how far are you running now?"),
            ("Ben", "About twelve kilometres on Sundays, slowly."),
            ("Ana", "Talk soon, Pixel needs her walk."),
        ),
    ),
    (
        "17 March 2024",
        (
            ("Ben", "Morning Ana! Quick hello before work."),
            ("Ana", "Morning! Busy week here, lots of meetings."),
            ("Ben", "Same. My sister Clara moved to Porto for a nursing job."),
            ("Ana", "Porto is beautiful, she will love it."),
            ("Ben", "She already found a flat near the river."),
            ("Ana", "I signed up for a pottery class on Thursday evenings."),
            ("Ben", "Nice, you always liked making things."),
            ("Ana", "We will see if my first bowl survives the kiln."),
            ("Ben", "Fingers crossed. Have a good day."),
        ),
    ),
    (
        "2 April 2024",
        (
            ("Ana", "Hey Ben, how is the marathon training going?"),
            ("Ben", "Still going, I hurt my ankle but it is healing."),
            ("Ana", "Take it easy. Pixel says hi."),
            ("Ben", "Ha, give her a pat from me."),
            ("Ana", "My pottery bowl came out lopsided but I love it."),
            ("Ben", "Handmade things are the best gifts."),
            ("Ana", "Exactly. I will make one for Clara's new flat."),
            ("Ben", "She would like that a lot."),
            ("Ana", "Okay, off to the park now. Bye!"),
        ),
    ),
)

QUESTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("What is the name of Ana's dog?", ("S1:3",)),
    ("Which race is Ben training for?", ("S1:6",)),
    ("Where did Ben's sister move?", ("S2:3",)),
    ("When is Ana's pottery class?", ("S2:6",)),
)


def demo_messages() -> list[Message]:
    out = []
    for s, (date, turns) in enumerate(_SESSIONS, 1):
        for t, (speaker, text) in enumerate(turns, 1):
            out.append(Message(f"S{s}:{t}", text, speaker, "user", date, s))
    return out


@dataclass(frozen=True, slots=True)
class DemoRow:
    policy: str
    found: int
    total: int
    evictions: int
    page_ins: int


def run_demo(budget: int = 320, show_context_for: str | None = "arc") -> list[DemoRow]:
    rows = []
    for policy in policy_names():
        manager = build_manager(budget=budget, policy=policy, core_blocks=("Ana", "Ben"))
        agent = HeartbeatAgent(manager, ScriptedMemoryAgent(manager, HeuristicImportance()))
        for message in demo_messages():
            agent.observe(message)
        state = manager.checkpoint()
        found = total = 0
        for question, evidence in QUESTIONS:
            manager.restore(state)
            agent.ask(question)
            context = manager.assemble()
            present = context.full_sources()
            found += sum(1 for e in evidence if e in present)
            total += len(evidence)
            if policy == show_context_for and question == QUESTIONS[0][0]:
                print(f"--- context for {question!r} ({policy}, {context.tokens}/{budget} tokens)")
                print(context.render())
                print("---")
        rows.append(DemoRow(policy, found, total, manager.stats.evictions, manager.stats.page_ins))
    return rows
