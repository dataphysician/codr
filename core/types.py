"""
Core types for the decoupled tree traversal architecture.

These types are shared across all components and must be JSON-serializable
for persistence and rewind functionality.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NewType

# Type aliases
NodeId = NewType('NodeId', str)  # normalized code/range identifier


class Action(str, Enum):
    """Actions that can be performed during traversal."""
    GOTO = "goto"           # traverse to target
    REPORT = "report"       # finalize & emit
    EXIT = "exit"           # terminate without report  
    MEMORY = "memory"       # defer/append for later
    FORK = "fork"           # spawn parallel branch
    
    # Domains may extend with additional actions


class GuardOutcome(str, Enum):
    """Results from guard checks on proposed moves."""
    ALLOW = "allow"
    BLOCK = "block"                  # hard stop
    UPROOT = "uproot"               # hard stop (alternative naming)
    REQUIRE_PREFIX = "require_prefix"    # ordering prerequisite (e.g., ICD codeFirst)
    REQUIRE_SUFFIX = "require_suffix"    # e.g., ICD useAdditionalCode  
    REQUIRE_SEVEN = "require_seven"      # ICD 7th-character lineage requirement
    
    # Domains may extend (e.g., CPT bundling edits)


# Removed SearchStrategy enum - now using int values directly for max_candidates


@dataclass
class GuardResult:
    """Result of a guard check on a proposed traversal move."""
    outcome: GuardOutcome
    message: str
    details: dict[str, Any] | None = None  # optional, machine-readable info


@dataclass  
class RichCandidate:
    """A candidate move with optional metadata from agent reasoning."""
    target: NodeId
    action: Action
    metadata: dict[str, Any] | None = None  # agent reasoning, citations, scores


@dataclass
class RunContext:
    """
    Domain-owned, JSON-serializable context.
    
    Opaque to orchestrator/agents; traversal engine mutates this.
    Contains domain-specific state like pending constraints, collected codes, etc.
    """
    data: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure data is always a dict."""
        if not isinstance(self.data, dict):
            self.data = {}


@dataclass
class QueryState:
    """Current state of a traversal query."""
    current: NodeId
    finalized: bool = False
    step: tuple[NodeId, Action, NodeId] | None = None  # (from, action, to)
    ctx: RunContext = field(default_factory=RunContext)
    
    def __post_init__(self):
        """Ensure ctx is always a RunContext instance."""
        if not isinstance(self.ctx, RunContext):
            self.ctx = RunContext(self.ctx if isinstance(self.ctx, dict) else {})