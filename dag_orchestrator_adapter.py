from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---- Import your tree + traversal layers -----------------------------------------
from icd_tree import ICDTreeNavigator
from icd_traversal import (
    RunContext,
    QueryState,
    Action,
    GuardResult,
    candidate_actions,
    apply_step,
    guard_next,
    resolve,
    ingest_notes_into_context,
)

# --- Decision policy (deterministic by default) -----------------------------------

class DecisionPolicy:
    """
    Orchestrator decision policy.
    Can be purely deterministic, heuristic, or delegate to an LLM in a subclass.
    MUST NOT invent new codes or actions; it may only reorder/prune legal candidates.
    """

    def rank_gotos(
        self,
        state: QueryState,
        goto_candidates: list[tuple[str, Action]],
        navigator: ICDTreeNavigator,
        top_k: int | None = None,
    ) -> list[tuple[str, Action]]:
        """
        Default: deterministic, leaf-biased heuristic.
        - Prefer candidates that are leaves
        - Then fewer children
        - Then lexical by code for stability
        """
        def score(cand: tuple[str, Action]) -> tuple[int, int, str]:
            code, _ = cand
            try:
                node = resolve(navigator, code)
                is_leaf = int(len(node.children) > 0)  # 0 if leaf, 1 otherwise
                child_count = len(node.children)
                return (is_leaf, child_count, code)
            except Exception:
                return (1, 9999, code)

        ranked = sorted(goto_candidates, key=score)
        return ranked[:top_k] if top_k else ranked

    def rank_forks(
        self,
        ctx: RunContext,
        navigator: ICDTreeNavigator,
        budget: int,
    ) -> list[str]:
        """
        Default: deterministic FIFO for parallel seeds.
        """
        return list(ctx.parallel)[:budget]

    def should_finalize(
        self,
        state: QueryState,
        navigator: ICDTreeNavigator,
    ) -> bool:
        """
        Advice only. Default: allow finalization when the runner attempts it.
        Guards in icd_traversal will still enforce legality.
        """
        return True


class LLMDecisionPolicy(DecisionPolicy):
    """
    Optional: override rank_* to call an LLM.
    Supports per-node overrides via a configurable provider.

    Contract:
    - Input candidates are already legal.
    - Output must be a reordering/subset of those candidates (no new ones).
    """

    def __init__(
        self,
        *,
        node_config_provider: callable | None = None,
    ) -> None:
        """
        node_config_provider(node_code: str) -> dict[str, Any]
        Lets you vary prompts/parameters per node (or return {}).
        """
        super().__init__()
        self.node_config_provider: callable | None = node_config_provider

    def get_node_call_config(
        self,
        node_code: str,
    ) -> dict[str, Any]:
        if self.node_config_provider:
            try:
                cfg = self.node_config_provider(node_code)
                return cfg if isinstance(cfg, dict) else {}
            except Exception:
                return {}
        return {}

    # Example sketch (pseudo-LLM): you would implement your client calls here.
    # def rank_gotos(self, state, goto_candidates, navigator, top_k=None):
    #     node_cfg = self.get_node_call_config(state.current)
    #     cards = build_candidate_cards(state, goto_candidates, navigator)  # your serializer
    #     llm_scores = call_llm(cards, node_cfg)  # returns mapping code->score
    #     ranked = sorted(goto_candidates, key=lambda c: -llm_scores.get(c[0], 0.0))
    #     return ranked[:top_k] if top_k else ranked


# ------------------------------------------------------------------------------
# Adapter outputs (neutral to any orchestrator)
# ------------------------------------------------------------------------------

@dataclass
class StepOutcome:
    """
    A single attempt result for the orchestrator.

    - status: "ok" when a new state is produced; otherwise a guard outcome string
              (e.g., "require_prefix", "require_suffix", "block", "uproot", ...)
    - snapshot: dict for logging/observability (DAG-runner friendly)
    - next_state: returned only on "ok"
    - side_effects: queues your runner might schedule (deferred, parallel, etc.)
    """
    status: str
    snapshot: dict[str, Any]
    next_state: QueryState | None
    side_effects: dict[str, Any]


# ------------------------------------------------------------------------------
# Orchestrator-agnostic adapter
# ------------------------------------------------------------------------------

class DAGTraversalAdapter:
    """
    Framework-neutral adapter that:

      • holds a navigator + run context
      • exposes deterministic next-step candidates
      • executes one step (apply_step) with guard enforcement
      • surfaces side-effects (memory/fork) for a global scheduler
      • lets the orchestrator append metadata to each committed step and node

    Metadata storage:
      - step_metadata[step_key] = [ {any}, ... ]
      - node_metadata[node_code] = [ {any}, ... ]
    e.g., { "reasoning": "...", "citations": ["doc://..."], "score": 0.87 }
    """

    def __init__(
        self,
        navigator: ICDTreeNavigator,
        start_code: str,
        *,
        ctx: RunContext | None = None,
        policy: DecisionPolicy | None = None,
    ) -> None:
        self.nav: ICDTreeNavigator = navigator
        self.ctx: RunContext = ctx or RunContext()
        self.policy: DecisionPolicy = policy or DecisionPolicy()

        # Resolve start and ingest its notes into context
        start_node = resolve(self.nav, start_code)
        ingest_notes_into_context(self.ctx, start_node)

        # Bootstrap a root state; step triple is descriptive here.
        self.state: QueryState = QueryState(
            prev=None,
            current=start_node.code,
            step=(start_node.code, Action.goto, start_node.code),
            finalized=False,
            ctx=self.ctx,
        )

        # Per-run metadata logs
        self.step_metadata: dict[str, list[dict[str, Any]]] = {}
        self.node_metadata: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Orchestrator-facing methods
    # ------------------------------------------------------------------

    def current_snapshot(self) -> dict[str, Any]:
        """Readable state for logging/telemetry."""
        return self._to_snapshot(self.state, action_hint=None, guard=None)

    def next_candidates(
        self,
        *,
        top_k: int | None = None,
    ) -> list[tuple[str, Action]]:
        """
        Enumerate deterministic next candidates, then let policy (deterministic or LLM) rank/prune.
        Returns a list of (target_code, Action) pairs for try_step().
        """
        base = candidate_actions(self.state, self.nav)
        return self.policy.rank_gotos(self.state, base, self.nav, top_k=top_k)

    def try_step(
        self,
        instruction: tuple[str, Action],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> StepOutcome:
        """
        Attempt one instruction (target_code, Action). Returns either a new state
        ("ok") or a guard-blocked outcome ("require_suffix", "block", etc.)
        with a snapshot explaining why.

        If status == "ok" and metadata is provided, it is appended to:
          - step_metadata[new_state.step_key]
          - node_metadata[new_state.current]
        """
        result = apply_step(self.state, instruction, self.nav)

        # Strict typing: check the guard result by isinstance
        if isinstance(result, GuardResult):
            snap = self._to_snapshot(self.state, action_hint=instruction, guard=result)
            return StepOutcome(
                status=result.outcome.value,
                snapshot=snap,
                next_state=None,
                side_effects=self._side_effects(),
            )

        # Allowed: advance state
        self.state = result
        # Optional metadata commit for this accepted step
        if metadata:
            self._record_metadata(self.state, metadata)

        snap = self._to_snapshot(self.state, action_hint=instruction, guard=None)
        return StepOutcome(
            status="ok",
            snapshot=snap,
            next_state=self.state,
            side_effects=self._side_effects(),
        )

    def attempt_finalize(
        self,
        action: Action = Action.report,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> StepOutcome:
        """
        Try to finalize at the current node (report or exit).
        Guard checks enforce pending prefix/suffix and lineage-safe 7th character.

        If status == "ok" and metadata is provided, it is appended to logs.
        """
        instr = (self.state.current, action)
        gr = guard_next(self.state, self.state.current, action, self.nav)
        if gr.outcome.value != "allow":
            snap = self._to_snapshot(self.state, action_hint=instr, guard=gr)
            return StepOutcome(
                status=gr.outcome.value,
                snapshot=snap,
                next_state=None,
                side_effects=self._side_effects(),
            )

        outcome = self.try_step(instr, metadata=metadata)
        return outcome

    def pick_parallel(self, budget: int) -> list[str]:
        """
        Select which parallel branches to launch now. This does not mutate ctx.parallel;
        let your orchestrator do that once it actually schedules the branches.
        """
        return self.policy.rank_forks(self.ctx, self.nav, budget)

    # ----- Metadata append APIs (for orchestrators/agents) ------------------------

    def append_metadata_for_current_step(self, metadata: dict[str, Any]) -> None:
        """
        Append metadata blobs (e.g., reasoning, citations) to:
          - the latest committed step
          - the current node
        """
        self._record_metadata(self.state, metadata)

    def append_metadata_for_node(self, node_code: str, metadata: dict[str, Any]) -> None:
        """
        Append metadata to an arbitrary node (by code), independent of step commits.
        Useful when an out-of-band agent annotates a node with evidence.
        """
        self._append(self.node_metadata, node_code, metadata)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_metadata(self, state: QueryState, metadata: dict[str, Any]) -> None:
        """Route metadata to per-step and per-node logs."""
        step_key = f"{state.step[0]}|{state.step[1].value}|{state.step[2]}"
        self._append(self.step_metadata, step_key, metadata)
        self._append(self.node_metadata, state.current, metadata)

    @staticmethod
    def _append(store: dict[str, list[dict[str, Any]]], key: str, payload: dict[str, Any]) -> None:
        if key not in store:
            store[key] = []
        store[key].append(payload)

    def _side_effects(self) -> dict[str, Any]:
        """Expose current global queues for the orchestrator."""
        return {
            "deferred": list(self.ctx.deferred),
            "parallel": list(self.ctx.parallel),
            "pending_prefix": sorted(self.ctx.pending_prefix),
            "pending_suffix": sorted(self.ctx.pending_suffix),
            "blocked_families": sorted(self.ctx.blocked_families),
            "pending_seven": {k: (sorted(v) if v else None) for k, v in self.ctx.pending_seven.items()},
        }

    def _to_snapshot(
        self,
        state: QueryState,
        *,
        action_hint: tuple[str, Action] | None,
        guard: GuardResult | None,
    ) -> dict[str, Any]:
        """
        Human-readable snapshot with enough context for debugging / telemetry.
        Includes any metadata already attached to the step and node.
        """
        cur, act, nxt = state.step
        if action_hint is not None:
            nxt = action_hint[0]
            act = action_hint[1]

        node = resolve(self.nav, state.current)
        meta = {
            "code": node.code,
            "name": getattr(node, "name", ""),
            "is_leaf": len(node.children) == 0,
            "children_count": len(node.children),
        }

        step_key = f"{cur}|{act.value}|{nxt}"
        snapshot: dict[str, Any] = {
            "key": step_key,
            "current": cur,
            "action": act.value,
            "next": nxt,
            "explain": self._explain(cur, act, nxt),
            "finalized": state.finalized,
            "node_meta": meta,
            "guards": {
                "blocked_families": sorted(self.ctx.blocked_families),
                "pending_prefix": sorted(self.ctx.pending_prefix),
                "pending_suffix": sorted(self.ctx.pending_suffix),
                "completed": sorted(self.ctx.completed),
                "pending_seven": {k: (sorted(v) if v else None) for k, v in self.ctx.pending_seven.items()},
            },
            "attached_metadata": {
                "step": list(self.step_metadata.get(step_key, [])),
                "node": list(self.node_metadata.get(state.current, [])),
            },
        }
        if guard is not None:
            snapshot["guard_outcome"] = guard.outcome.value
            snapshot["guard_message"] = guard.message
        return snapshot

    @staticmethod
    def _explain(cur: str, action: Action, nxt: str) -> str:
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
