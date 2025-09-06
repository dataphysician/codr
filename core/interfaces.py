"""
Core interfaces for the decoupled tree traversal architecture.

These abstract base classes define the contracts between different layers:
- TreeIndex: structure + metadata access
- TraversalEngine: domain-specific traversal rules
- Agent interfaces: candidate selection and scoring
"""

from __future__ import annotations
from typing import Any, Dict, List, Protocol

from .types import NodeId, Action, GuardResult, QueryState, RunContext, RichCandidate


class NodeView:
    """View of a single node with its metadata."""
    
    def __init__(
        self, 
        id: NodeId, 
        name: str, 
        element_type: str, 
        notes: dict[str, list[tuple[str, str]]] | None = None
    ):
        self.id = id
        self.name = name  
        self.element_type = element_type  # "chapter" | "block" | "diagnosis" | domain-specific
        self.notes = notes or {}  # {note_type: [(code, desc)]}


class TreeIndex(Protocol):
    """
    Protocol for tree structure and metadata access.
    
    Implementations provide fast lookup of nodes and their relationships
    without domain-specific traversal logic. Works with any hierarchical
    coding system (ICD, CPT, SNOMED, etc.).
    
    Uses structural typing to allow flexible implementations without
    forcing inheritance constraints that could disrupt adaptive agents.
    """
    
    def normalize_id(self, raw: str) -> NodeId:
        """Normalize a raw string to a canonical NodeId."""
        ...
    
    def get(self, node_id: NodeId) -> NodeView | None:
        """Get a node by its ID, or None if not found."""
        ...
    
    def children(self, node_id: NodeId) -> list[NodeId]:
        """Get direct children of a node."""
        ...
    
    def ancestors(self, node_id: NodeId) -> list[NodeId]:
        """Get ancestors from parent to root (ordered parent -> grandparent -> root)."""
        ...
    
    def path_to_root(self, node_id: NodeId) -> list[NodeId]:
        """Get full path from node to root (ordered node -> parent -> root)."""
        ...
    
    def is_leaf(self, node_id: NodeId) -> bool:
        """Check if node has no children."""
        ...
    
    def details(self, node_id: NodeId) -> dict[str, Any]:
        """Get comprehensive details for a node (convenience for agents)."""
        ...
    
    def search(self, text: str, k: int = 50) -> list[NodeView]:
        """Search for nodes matching text query."""
        ...


class TraversalEngine(Protocol):
    """
    Protocol for domain-specific traversal rules and state management.
    
    Implementations encode business logic for different coding systems (ICD-10, CPT, etc.).
    Must be pure and deterministic given (tree, state, move). Works with any hierarchical
    coding system by implementing domain-specific rules.
    
    Uses structural typing to allow flexible implementations and self-optimizing agents
    that can adapt their behavior based on performance feedback.
    """
    
    def dump_state(self, q: QueryState) -> dict[str, Any]:
        """Serialize QueryState to JSON-compatible dict."""
        ...
    
    def load_state(self, d: dict[str, Any]) -> QueryState:
        """Deserialize QueryState from JSON-compatible dict."""
        ...
    
    def dump_context(self, c: RunContext) -> dict[str, Any]:
        """Serialize RunContext to JSON-compatible dict."""
        ...
    
    def load_context(self, d: dict[str, Any]) -> RunContext:
        """Deserialize RunContext from JSON-compatible dict."""
        ...
    
    def ingest(self, tree: TreeIndex, node_id: NodeId, ctx: RunContext) -> RunContext:
        """Fold node notes/constraints into context when entering a node."""
        ...
    
    def candidate_actions(self, tree: TreeIndex, q: QueryState) -> list[tuple[NodeId, Action]]:
        """Get legal next moves from current state."""
        ...
    
    def guard_next(self, tree: TreeIndex, q: QueryState, move: tuple[NodeId, Action]) -> GuardResult:
        """Guard a proposed move against domain rules."""
        ...
    
    def apply(self, tree: TreeIndex, q: QueryState, move: tuple[NodeId, Action]) -> QueryState | GuardResult:
        """Apply move if allowed, returning new QueryState or GuardResult if blocked."""
        ...
    
    def can_attempt_finalize(self, tree: TreeIndex, q: QueryState) -> bool:
        """Check if current state allows attempting finalization (advisory)."""
        ...
    
    def parallel_seeds(self, tree: TreeIndex, q: QueryState) -> list[NodeId]:
        """Get seeds for parallel branches (e.g., ICD codeAlso, CPT add-ons)."""
        ...


class DecisionContext:
    """
    Context provided to agents for decision making.
    
    Contains all information needed for agents to choose among candidates
    without accessing tree internals directly.
    """
    
    def __init__(
        self,
        node: NodeView,
        ancestors: list[NodeView],
        children: list[NodeView], 
        path: list[NodeId],
        allowed_actions: list[Action],
        pending_constraints: dict[str, Any],
        external_context: dict[str, Any] | None = None
    ):
        self.node = node
        self.ancestors = ancestors  
        self.children = children
        self.path = path  # root → current
        self.allowed_actions = allowed_actions
        self.pending_constraints = pending_constraints  # engine-extracted constraints
        self.external_context = external_context or {}  # EHR/document, user intent, etc.


class CandidateAgent(Protocol):
    """
    Protocol for agents that select and rank candidates.
    
    Agents choose among legal candidates, never inventing new moves.
    Works with any domain by using only the DecisionContext interface.
    
    Uses structural typing to enable self-optimizing agents (like DSPy)
    that can modify their behavior and internal structure dynamically.
    """
    
    def select(
        self,
        decision: DecisionContext,
        candidates: list[tuple[NodeId, Action]]
    ) -> list[RichCandidate]:
        """Return ordered subset of candidates with optional metadata."""
        ...


class CandidateScorer(Protocol):
    """
    Protocol for agents that only score candidates.
    
    Alternative to full selection when you only need numerical scores.
    Works with any domain by using only the DecisionContext interface.
    
    Uses structural typing to enable adaptive scoring models.
    """
    
    def score(
        self,
        decision: DecisionContext, 
        candidates: list[tuple[NodeId, Action]]
    ) -> dict[tuple[NodeId, Action], float]:
        """Return scores for candidates. Higher scores are better."""
        ...