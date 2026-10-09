"""OS-style virtual context management for long-running agents."""

from vcmem.agent import HeartbeatAgent, StepTrace
from vcmem.assembler import AssembledContext, ContextAssembler, ContextItem, Priority
from vcmem.build import build_manager, make_counter, make_embedder
from vcmem.embed import CachedEmbedder, Embedder, HashingEmbedder, MiniLMEmbedder
from vcmem.manager import BudgetPlan, MemoryManager
from vcmem.messages import Message
from vcmem.policies import (
    ARCPolicy,
    EvictionPolicy,
    FIFOPolicy,
    ImportancePolicy,
    LRUPolicy,
    PageInfo,
    make_policy,
    register_policy,
)
from vcmem.scoring import HeuristicImportance, ProviderImportance
from vcmem.scripted import ScriptedMemoryAgent
from vcmem.summary import ProviderSummarizer, Summary, TextRankSummarizer
from vcmem.tokens import RegexTokenCounter, TokenCounter, WordPieceTokenCounter
from vcmem.tools import MemoryTools, tool_specs

__version__ = "0.1.0"

__all__ = [
    "ARCPolicy",
    "AssembledContext",
    "BudgetPlan",
    "CachedEmbedder",
    "ContextAssembler",
    "ContextItem",
    "Embedder",
    "EvictionPolicy",
    "FIFOPolicy",
    "HashingEmbedder",
    "HeartbeatAgent",
    "HeuristicImportance",
    "ImportancePolicy",
    "LRUPolicy",
    "MemoryManager",
    "MemoryTools",
    "Message",
    "MiniLMEmbedder",
    "PageInfo",
    "Priority",
    "ProviderImportance",
    "ProviderSummarizer",
    "RegexTokenCounter",
    "ScriptedMemoryAgent",
    "StepTrace",
    "Summary",
    "TextRankSummarizer",
    "TokenCounter",
    "WordPieceTokenCounter",
    "build_manager",
    "make_counter",
    "make_embedder",
    "make_policy",
    "register_policy",
    "tool_specs",
]
