from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from vcmem.assembler import ContextAssembler, ContextItem, Priority
from vcmem.tokens import CachedCounter, RegexTokenCounter

counter = RegexTokenCounter()
words = st.text(alphabet="abcdefghij .,!?'", min_size=0, max_size=60)


def test_regex_counter_counts_words_and_punctuation() -> None:
    assert counter.count("Hello, world!") == 4
    assert counter.count("") == 0


@given(words, words)
def test_regex_counter_is_additive_across_newlines(a: str, b: str) -> None:
    assert counter.count(a + "\n" + b) == counter.count(a) + counter.count(b)


@given(words, st.integers(min_value=0, max_value=30))
def test_truncate_never_exceeds_limit(text: str, n: int) -> None:
    out = counter.truncate(text, n)
    assert counter.count(out) <= n
    assert text.startswith(out)


def test_cached_counter_matches_inner() -> None:
    cached = CachedCounter(counter, maxsize=2)
    for text in ["a b", "c, d", "a b", "e f g"]:
        assert cached.count(text) == counter.count(text)


items_strategy = st.lists(
    st.builds(
        ContextItem,
        text=words,
        priority=st.sampled_from(list(Priority)),
        position=st.floats(min_value=0, max_value=100, allow_nan=False),
        keep_rank=st.floats(min_value=-100, max_value=100, allow_nan=False),
        pinned=st.booleans(),
    ),
    max_size=25,
)


@settings(max_examples=300)
@given(items_strategy, st.integers(min_value=0, max_value=120))
def test_assembled_context_never_exceeds_budget(items: list[ContextItem], budget: int) -> None:
    ctx = ContextAssembler(counter).assemble(items, budget)
    assert ctx.tokens <= budget
    assert counter.count(ctx.render()) == ctx.tokens  # region breaks cost no tokens
    assert sum(ctx.item_tokens) == ctx.tokens


@given(items_strategy, st.integers(min_value=0, max_value=120))
def test_pinned_items_are_never_dropped(items: list[ContextItem], budget: int) -> None:
    ctx = ContextAssembler(counter).assemble(items, budget)
    assert all(not item.pinned for item in ctx.dropped)
    assert sum(1 for i in ctx.items if i.pinned) == sum(1 for i in items if i.pinned)


def test_lower_priority_is_dropped_first() -> None:
    items = [
        ContextItem("system prompt here", Priority.SYSTEM, pinned=True),
        ContextItem("core fact one two", Priority.CORE),
        ContextItem("old queue message words", Priority.QUEUE, position=1, keep_rank=1),
        ContextItem("new queue message words", Priority.QUEUE, position=2, keep_rank=2),
    ]
    ctx = ContextAssembler(counter).assemble(items, budget=3 + 4 + 4)
    kept = [i.text for i in ctx.items]
    assert kept == ["system prompt here", "core fact one two", "new queue message words"]


def test_render_order_puts_input_last_and_queue_chronological() -> None:
    items = [
        ContextItem("question?", Priority.INPUT, pinned=True),
        ContextItem("second", Priority.QUEUE, position=2),
        ContextItem("sys", Priority.SYSTEM, pinned=True),
        ContextItem("first", Priority.RECALL, position=1),
        ContextItem("summary", Priority.SUMMARY),
    ]
    ctx = ContextAssembler(counter).assemble(items, budget=100)
    assert [i.text for i in ctx.items] == ["sys", "summary", "first", "second", "question?"]
    chat = ctx.to_chat_messages()
    assert chat[0].role == "system"
    assert [m.content for m in chat[1:]] == ["first", "second", "question?"]


def test_truncated_pinned_item_loses_full_source() -> None:
    item = ContextItem(
        "one two three four five six", Priority.INPUT, pinned=True, full_sources=frozenset({"x"})
    )
    ctx = ContextAssembler(counter).assemble([item], budget=3)
    assert ctx.tokens == 3
    assert ctx.full_sources() == frozenset()
    assert ctx.partial_sources() == frozenset({"x"})
