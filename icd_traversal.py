from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict, StringConstraints

# ---------------------------------------------------------------------
# Regexes & Normalization (mirror icd_tree.py to avoid drift)
# ---------------------------------------------------------------------

CODE_PATTERN_RE: str = (
    r"\(([A-Z][0-9][0-9](?:\.[0-9A-X-]+)?"
    r"(?:-[A-Z][0-9][0-9](?:\.[0-9A-X-]+)?)?"
    r"(?:, ?[A-Z][0-9][0-9](?:\.[0-9A-X-]+)?)*"
    r"(?:\.?-)?)\)"
)

NORMALIZED_CODE_RE: str = (
    r"^[A-Z][0-9][0-9](?:\.[0-9A-X-]+)?"
    r"(?:-[A-Z][0-9][0-9](?:\.[0-9A-X-]+)?)?"
    r"(?:, ?[A-Z][0-9][0-9](?:\.[0-9A-X-]+)?)*"
    r"(?:\.?-)?$"
)

ICDCode = StringConstraints(pattern=NORMALIZED_CODE_RE)

def to_inner_code(s: str) -> str:
    """
    Accept '(X..)', 'X..' → return normalized inner 'X..'.
    Use this on all external inputs before navigator lookups.
    """
    import re
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if not re.match(NORMALIZED_CODE_RE, s):
        raise ValueError(f"Invalid ICD code/range/list: {s!r}")
    return s

def family_match(candidate: str, family_root: str) -> bool:
    """
    Pragmatic 'family' membership: prefix match ignoring dots.
    Works reasonably for excludes families and range roots.
    """
    a = candidate.replace(".", "")
    b = family_root.replace(".", "")
    return a.startswith(b)

# ---------------------------------------------------------------------
# Actions & Guards
# ---------------------------------------------------------------------

class Action(str, Enum):
    goto = "goto"       # move to target code
    report = "report"   # finalize/report at current code (billable check applies)
    memory = "memory"   # defer current code for end-stage reporting; continue to target
    exit = "exit"       # terminate at current code
    fork = "fork"       # suggest a parallel branch at target (runner schedules)

class GuardOutcome(str, Enum):
    allow = "allow"
    uproot = "uproot"               # excludes1 hard stop
    block = "block"                 # excludes2 family or ordering violation
    require_prefix = "require_prefix"  # codeFirst pending
    require_suffix = "require_suffix"  # useAdditionalCode pending
    require_seven = "require_seven"    # seven-character required for finalization

class GuardResult(BaseModel):
    model_config = ConfigDict(frozen=False)
    outcome: GuardOutcome
    message: str

# ---------------------------------------------------------------------
# Run-level context (owned by your runner, framework-neutral)
# ---------------------------------------------------------------------

class RunContext(BaseModel):
    """
    Long-lived accumulators for a single traversal/run.
    This context is separate from any orchestration engine (Burr, etc.).
    """
    model_config = ConfigDict(frozen=False)

    # Guard accumulators
    blocked_families: set[str] = Field(default_factory=set)  # from excludes2
    pending_prefix: set[str] = Field(default_factory=set)    # from codeFirst
    pending_suffix: set[str] = Field(default_factory=set)    # from useAdditionalCode

    # 7th-character: anchor_code -> allowed set (or None if any 7th is acceptable)
    pending_seven: dict[str, set[str] | None] = Field(default_factory=dict)

    # Visited (for satisfying prefix/suffix over time)
    completed: set[str] = Field(default_factory=set)

    # Side-effect queues
    deferred: list[str] = Field(default_factory=list)        # 'memory' nodes for end-stage reporting
    parallel: list[str] = Field(default_factory=list)        # 'fork' or codeAlso seeds (runner schedules)

    # ---------------- Runner utility methods ----------------

    def mark_progress(self, code: str) -> None:
        """Record a visited code to satisfy prefix/suffix constraints."""
        if code in self.pending_prefix:
            self.pending_prefix.remove(code)
        if code in self.pending_suffix:
            self.pending_suffix.remove(code)
        self.completed.add(code)

    def register_codealso(self, codes: list[str]) -> None:
        """Collect parallel exploration seeds (e.g., from codeAlso notes)."""
        for c in sorted({to_inner_code(x) for x in codes}):
            if c not in self.parallel:
                self.parallel.append(c)

    def add_excludes2(self, families: list[str]) -> None:
        """Add excluded families (normalized)."""
        for fam in families:
            self.blocked_families.add(to_inner_code(fam))

    def add_prefixes(self, codes: list[str]) -> None:
        for c in codes:
            self.pending_prefix.add(to_inner_code(c))

    def add_suffixes(self, codes: list[str]) -> None:
        for c in codes:
            self.pending_suffix.add(to_inner_code(c))

    def add_seven_requirement(self, anchor_code: str, allowed: set[str] | None) -> None:
        """
        Record that a 7th character is required under the lineage of 'anchor_code'.
        'allowed' can be a set like {'A','D','S'} or None if not constrained.
        """
        self.pending_seven[to_inner_code(anchor_code)] = allowed

    def lineage_handoff_ok(self, navigator: Any, anchor_code: str, leaf_code: str) -> bool:
        """
        Only allow satisfying a pending 7th-char requirement if the leaf
        descends from the anchor that introduced that requirement.
        """
        path = navigator.get_path_to_code(leaf_code) or []
        return anchor_code in path

# ---------------------------------------------------------------------
# Per-step state snapshot (tiny, deterministic keys)
# ---------------------------------------------------------------------

ActionStep = tuple[str, Action, str]  # (current_code, action, next_code)

class QueryState(BaseModel):
    """
    A single step snapshot. The runner keeps the RunContext and Navigator.
    """
    model_config = ConfigDict(frozen=False)

    prev: "QueryState | None" = None
    current: str  # normalized current code
    step: ActionStep
    finalized: bool = False  # True after report/exit is accepted
    ctx: RunContext

    @property
    def key(self) -> str:
        cur, action, nxt = self.step
        return f"{cur}|{action.value}|{nxt}"

    def explain(self) -> str:
        cur, action, nxt = self.step
        if action is Action.goto:
            return f"go from {cur} to {nxt}"
        if action is Action.memory:
            return f"defer {cur} for end-reporting; continue to {nxt}"
        if action is Action.report:
            return f"report {cur} as final"
        if action is Action.exit:
            return f"terminate at {cur}"
        if action is Action.fork:
            return f"fork a parallel branch at {nxt}"
        return f"{action.value} from {cur} to {nxt}"

# ---------------------------------------------------------------------
# Thin integration expectations for navigator / node
# ---------------------------------------------------------------------
# EXPECTED navigator API (from icd_tree.py):
#   - navigator.find_by_code(code) -> Node | None
#   - navigator.get_path_to_code(code) -> list[str] | None
# Node attributes used:
#   - node.code : str
#   - node.children : iterable of child Nodes
#   - node.notes : dict[str, list[tuple[str, str]]]  # [(code, desc)], keys: includes, excludes1, excludes2, codeFirst, useAdditionalCode, codeAlso, sevenChrNote, inclusionTerm

def resolve(navigator: Any, code: str) -> Any:
    """
    Normalize and resolve a code into a tree node via navigator.
    This is a pass-through to avoid duplicating indexes.
    """
    inner = to_inner_code(code)
    node = navigator.find_by_code(inner)
    if node is None:
        raise KeyError(f"Unknown code node: {inner}")
    return node

def node_children_codes(node: Any) -> list[str]:
    """List child codes from a Node."""
    return [child.code for child in node.children]

def note_codes(node: Any, key: str) -> list[str]:
    """
    Extract codes stored in node.notes[key], ignoring descriptions.
    Returns [] if key missing.
    """
    pairs: list[tuple[str, str]] = node.notes.get(key, [])
    return [c for (c, _d) in pairs if c]

def has_seven_requirement(node: Any) -> bool:
    """Detect a seven-character requirement on this node."""
    return "sevenChrNote" in node.notes

def parse_allowed_seven_from_note(node: Any) -> set[str] | None:
    """
    (Optional) Parse allowed 7th characters (e.g., {'A','D','S'}) from sevenChrNote text.
    This is a good place to plug in an LLM to interpret publisher prose.
    Returns a set or None if unconstrained / not parsable.
    """
    # LLM HOOK: inspect node.notes['sevenChrNote'] descriptions to infer allowed set.
    return None

def is_leaf(node: Any) -> bool:
    return len(node.children) == 0

def code_has_seven_char(code: str) -> bool:
    """Heuristic: 7-character when dots removed."""
    return len(code.replace(".", "")) >= 7

# ---------------------------------------------------------------------
# Candidate generation & guards
# ---------------------------------------------------------------------

def candidate_actions(state: QueryState, navigator: Any) -> list[tuple[str, Action]]:
    """
    Deterministic, note-guided next-step generator.
    Priority:
      1) codeFirst pending -> propose gotos to those prefixes (sorted)
      2) Node children as goto targets (stable order from tree)
      3) (Policy) caller may attempt report/exit if leaf and guards allow

    Side-effects (not here):
      - codeAlso: seeded to ctx.parallel when entering a node (via ingest_notes_into_context)
    """
    cur_node = resolve(navigator, state.current)

    # seed parallel suggestions from codeAlso separately (ingested on entry)

    # 1) Must satisfy codeFirst first, if pending
    if state.ctx.pending_prefix:
        return [(c, Action.goto) for c in sorted(state.ctx.pending_prefix)]

    # 2) Default next hops: children of current node
    children = node_children_codes(cur_node)
    return [(c, Action.goto) for c in children]

def guard_next(state: QueryState, target: str, action: Action, navigator: Any) -> GuardResult:
    """
    Enforce excludes1/excludes2, codeFirst ordering, and seven/suffix constraints
    at the appropriate times.
    """
    cur_node = resolve(navigator, state.current)
    target = to_inner_code(target)

    # excludes1 at current -> uproot if jump enters that family
    for fam in note_codes(cur_node, "excludes1"):
        if family_match(target, fam):
            return GuardResult(outcome=GuardOutcome.uproot, message=f"Jump to {target} violates excludes1 at {state.current} (family {fam}).")

    # excludes2 accumulated -> block if target enters blocked family
    for fam in state.ctx.blocked_families:
        if family_match(target, fam):
            return GuardResult(outcome=GuardOutcome.block, message=f"Jump to {target} blocked by excludes2 family {fam}.")

    # codeFirst: cannot leave current unless jumping to one of pending prefixes
    if action is Action.goto and state.ctx.pending_prefix and target not in state.ctx.pending_prefix:
        need = ", ".join(sorted(state.ctx.pending_prefix))
        return GuardResult(outcome=GuardOutcome.require_prefix, message=f"codeFirst pending: {need} before leaving {state.current}.")

    # Finalization checks (report/exit)
    if action in {Action.report, Action.exit}:
        # 7th required? Only consumable if anchor is in the leaf's lineage.
        if state.ctx.pending_seven:
            # must be a leaf to finalize
            if not is_leaf(cur_node):
                return GuardResult(outcome=GuardOutcome.require_seven, message="Seven-character deferred suffix pending; finalize at a leaf.")

            lineage_ok = any(
                state.ctx.lineage_handoff_ok(navigator, anchor, state.current)
                for anchor in state.ctx.pending_seven.keys()
            )
            if not lineage_ok:
                return GuardResult(outcome=GuardOutcome.require_seven, message="Seven-character requirement exists but not on this leaf's lineage.")

            if not code_has_seven_char(state.current):
                return GuardResult(outcome=GuardOutcome.require_seven, message="Seven-character extension required to finalize.")

        # useAdditionalCode suffix pending?
        if state.ctx.pending_suffix:
            need = ", ".join(sorted(state.ctx.pending_suffix))
            return GuardResult(outcome=GuardOutcome.require_suffix, message=f"useAdditionalCode pending: {need}.")

    return GuardResult(outcome=GuardOutcome.allow, message="ok")

# ---------------------------------------------------------------------
# Step application
# ---------------------------------------------------------------------

def ingest_notes_into_context(ctx: RunContext, node: Any) -> None:
    """
    On entering a node, fold its notes into the run context.
    """
    # excludes2 families
    ctx.add_excludes2(note_codes(node, "excludes2"))

    # codeFirst / useAdditionalCode
    ctx.add_prefixes(note_codes(node, "codeFirst"))
    ctx.add_suffixes(note_codes(node, "useAdditionalCode"))

    # seven-character requirement
    if has_seven_requirement(node):
        allowed = parse_allowed_seven_from_note(node)  # (LLM HOOK) parse allowed {'A','D','S'} if present
        ctx.add_seven_requirement(node.code, allowed)

    # codeAlso seeds for parallel discovery
    ctx.register_codealso(note_codes(node, "codeAlso"))

def apply_step(prev: QueryState, instruction: tuple[str, Action], navigator: Any) -> QueryState | GuardResult:
    """
    Attempt one action and return either a GuardResult or the next QueryState.

    LLM HOOKS:
      - A policy/LLM can decide *which* instruction to try next from candidate_actions().
      - A policy/LLM can interpret prose in sevenChrNote to set allowed sets.
      - A policy/LLM can rank children when there are many plausible clinical branches.
    """
    target, action = instruction
    target = to_inner_code(target)

    # Guards
    gr = guard_next(prev, target, action, navigator)
    if gr.outcome is not GuardOutcome.allow:
        return gr

    # Side-effects (framework-neutral)
    if action is Action.memory:
        if prev.current not in prev.ctx.deferred:
            prev.ctx.deferred.append(prev.current)
    if action is Action.fork:
        if target not in prev.ctx.parallel:
            prev.ctx.parallel.append(target)

    # Progress marking
    if action in {Action.goto, Action.memory, Action.fork}:
        prev.ctx.mark_progress(target)
    elif action in {Action.report, Action.exit}:
        prev.ctx.mark_progress(prev.current)

    # Jumps advance to the target node; finalization stays on current
    if action in {Action.goto, Action.memory, Action.fork}:
        target_node = resolve(navigator, target)
        # Ingest the next node's notes (blocks/prefix/suffix/7th/parallel seeds)
        ingest_notes_into_context(prev.ctx, target_node)
        next_state = QueryState(
            prev=prev,
            current=target_node.code,
            step=(prev.current, action, target),
            finalized=False,
            ctx=prev.ctx,
        )
        return next_state

    # Finalization step (report/exit) — accept and return finalized state
    return QueryState(
        prev=prev,
        current=prev.current,
        step=(prev.current, action, prev.current),
        finalized=True,
        ctx=prev.ctx,
    )

# ---------------------------------------------------------------------
# (Optional) Policy helpers for runners
# ---------------------------------------------------------------------

def can_attempt_finalize(state: QueryState, navigator: Any) -> bool:
    """
    Quick policy gate: only attempt report/exit on a true leaf.
    Guards enforce the rest (prefix/suffix/7th lineage).
    """
    node = resolve(navigator, state.current)
    return is_leaf(node)

def children_candidates(state: QueryState, navigator: Any) -> list[str]:
    """
    Plain child codes for UI/agents; useful if you want to present choices.
    (A policy/LLM can rank these.)
    """
    node = resolve(navigator, state.current)
    return node_children_codes(node)
