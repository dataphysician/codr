from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Pattern, Protocol
import re

from burr.core import action, State, ApplicationBuilder, ApplicationContext, Result, expr
from burr.tracking.local import LocalTrackingClient

from icd_tree import create_simple_navigator, ICDTreeNavigator
from icd_traversal import (
    RunContext,
    QueryState,
    Action,
    candidate_actions,
    apply_step,
    guard_next,
    resolve,
    ingest_notes_into_context,
)
# Reuse the neutral agent interfaces so non-Burr runners can share agents:
from dag_orchestrator_adapter import CandidateAgent, RichCandidate

# =================================================================================
# Per-node Policy Router (exact / prefix / regex) + optimization hooks
# =================================================================================

class OptimizableAgent(Protocol):
    """Optional extension: allow per-node optimization (e.g., DSPy self-tuning)."""
    def optimize(self, node_code: str, dataset_ref: str | None = None) -> None: ...
    def archive_exemplar(self, example: dict[str, Any]) -> None: ...

class FeedbackSink(Protocol):
    """Store human feedback for later analysis / tuning."""
    def record(self, payload: dict[str, Any]) -> None: ...

class ExemplarSink(Protocol):
    """Archive training exemplars for future fine-tuning or DSPy traces."""
    def save(self, example: dict[str, Any]) -> None: ...

class NodePolicyRouter:
    """
    Route current node code -> CandidateAgent (LLM/DSPy/human-in-the-loop).
    Supports exact, prefix, and regex routes. Provides optimization hooks.
    """
    def __init__(
        self,
        default_agent: CandidateAgent | None = None,
        *,
        feedback_sink: FeedbackSink | None = None,
        exemplar_sink: ExemplarSink | None = None,
    ) -> None:
        self.default_agent = default_agent
        self.by_exact: dict[str, CandidateAgent] = {}
        self.by_prefix: list[tuple[str, CandidateAgent]] = []
        self.by_regex: list[tuple[Pattern[str], CandidateAgent]] = []
        self.feedback_sink = feedback_sink
        self.exemplar_sink = exemplar_sink

    # ---- registration ----
    def register_exact(self, code: str, agent: CandidateAgent) -> None:
        self.by_exact[code] = agent

    def register_prefix(self, prefix: str, agent: CandidateAgent) -> None:
        self.by_prefix.append((prefix, agent))

    def register_regex(self, pattern: str, agent: CandidateAgent) -> None:
        self.by_regex.append((re.compile(pattern), agent))

    # ---- routing ----
    def get_agent_for(self, node_code: str) -> CandidateAgent | None:
        if node_code in self.by_exact:
            return self.by_exact[node_code]
        for prefix, agent in self.by_prefix:
            if node_code.startswith(prefix):
                return agent
        for pat, agent in self.by_regex:
            if pat.search(node_code):
                return agent
        return self.default_agent

    # ---- optional hooks ----
    def optimize(self, node_code: str, dataset_ref: str | None = None) -> None:
        agent = self.get_agent_for(node_code)
        if agent and isinstance(agent, OptimizableAgent):  # type: ignore[arg-type]
            agent.optimize(node_code, dataset_ref)

    def archive(self, example: dict[str, Any]) -> None:
        if self.exemplar_sink:
            self.exemplar_sink.save(example)
        # If agent wants its own archive:
        node_code = example.get("node_code", "")
        agent = self.get_agent_for(node_code)
        if agent and isinstance(agent, OptimizableAgent):  # type: ignore[arg-type]
            agent.archive_exemplar(example)

    def feedback(self, payload: dict[str, Any]) -> None:
        if self.feedback_sink:
            self.feedback_sink.record(payload)

# =================================================================================
# Process-level singletons (navigator & policy router)
# =================================================================================

_NAV: ICDTreeNavigator | None = None
_POLICY: NodePolicyRouter | None = None

def set_policy_router(router: NodePolicyRouter) -> None:
    global _POLICY
    _POLICY = router

def _policy() -> NodePolicyRouter | None:
    return _POLICY

def _nav() -> ICDTreeNavigator:
    global _NAV
    if _NAV is None:
        _NAV = create_simple_navigator()
    return _NAV

# =================================================================================
# JSON-safe (de)serialization helpers for Burr state
# =================================================================================

def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):  # pydantic v2
        return obj.model_dump(mode="json")
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return obj
    raise TypeError(f"State value not JSON-serializable: {type(obj)}")

def _qstate_from(d: dict[str, Any]) -> QueryState:
    # pydantic v2 accepts dict with json-compatible containers
    return QueryState.model_validate(d)

def _rctx_from(d: dict[str, Any]) -> RunContext:
    return RunContext.model_validate(d)

# =================================================================================
# Burr actions: init, enumerate, select (per-node policy), apply, parallel, feedback
# =================================================================================

@action(reads=[], writes=["qstate", "timeline", "step_index", "rewind_map"])
def init_traversal(state: State, start_code: str) -> State:
    nav = _nav()
    node = resolve(nav, start_code)
    ctx = RunContext()
    ingest_notes_into_context(ctx, node)
    q = QueryState(
        prev=None,
        current=node.code,
        step=(node.code, Action.goto, node.code),
        finalized=False,
        ctx=ctx,
    )
    return (
        state
        .update(
            qstate=_to_jsonable(q),
            timeline=[],
            step_index=0,
            rewind_map={},
        )
    )

@action(reads=["qstate"], writes=["candidates"])
def enumerate_candidates(state: State, __context: ApplicationContext) -> State:
    nav = _nav()
    q = _qstate_from(state["qstate"])
    cands = candidate_actions(q, nav)
    packed = [(code, act.value) for (code, act) in cands]
    return state.update(candidates=packed)

@action(reads=["qstate", "candidates"], writes=["choice", "choice_meta", "choice_action"])
def select_candidate(state: State, __context: ApplicationContext, top_k: int | None = None, **agent_inputs: Any) -> State:
    """
    Per-node policy selection:
      - Route to an Agent via NodePolicyRouter
      - Agent returns ordered RichCandidates with metadata
      - We pick the first (or you can extend to multi-pick)
    If no agent, falls back to a simple deterministic heuristic.
    """
    nav = _nav()
    q = _qstate_from(state["qstate"])
    base: list[tuple[str, Action]] = [(c, Action(a)) for (c, a) in state.get("candidates", [])]
    router = _policy()

    selected: RichCandidate | None = None
    if router:
        agent = router.get_agent_for(q.current)
        if agent:
            rich = agent.select_gotos(q, base, nav, top_k=top_k)
            if rich:
                selected = rich[0]

    if selected is None:
        # Heuristic fallback: prefer leaves then fewest children then lexical
        ranked = sorted(
            base,
            key=lambda ca: (1 if len(resolve(nav, ca[0]).children) > 0 else 0, len(resolve(nav, ca[0]).children), ca[0]),
        )
        if ranked:
            code, act = ranked[0]
            selected = RichCandidate(target=code, action=act, metadata={"policy": "heuristic"})
        else:
            selected = RichCandidate(target=q.current, action=Action.report, metadata={"reason": "no candidates; report"})

    return state.update(choice=selected.target, choice_action=selected.action.value, choice_meta=selected.metadata or {})

@action(reads=["qstate", "choice", "choice_action", "timeline", "step_index", "rewind_map"],
        writes=["qstate", "timeline", "step_index", "rewind_map", "last_guard"])
def apply_selected_step(state: State, __context: ApplicationContext) -> State:
    """
    guard_next → apply_step; persist snapshots; map step_key → sequence_id for rewind.
    """
    nav = _nav()
    q = _qstate_from(state["qstate"])
    code = str(state["choice"])
    action = Action(state["choice_action"])

    gr = guard_next(q, code, action, nav)
    state = state.update(last_guard={"outcome": gr.outcome.value, "message": gr.message})

    if gr.outcome.value != "allow":
        snapshot = {
            "key": f"{q.step[0]}|{action.value}|{code}",
            "current": q.current,
            "action": action.value,
            "next": code,
            "explain": f"blocked: {gr.message}",
            "finalized": q.finalized,
            "guard": gr.outcome.value,
            "choice_meta": state.get("choice_meta", {}),
        }
        seq = __context.sequence_id
        new_tl = list(state["timeline"]) + [snapshot]
        new_map = dict(state["rewind_map"])
        new_map[snapshot["key"]] = seq
        return state.update(timeline=new_tl, rewind_map=new_map)

    new_q = apply_step(q, (code, action), nav)
    snapshot = {
        "key": f"{new_q.step[0]}|{new_q.step[1].value}|{new_q.step[2]}",
        "current": new_q.current,
        "action": new_q.step[1].value,
        "next": new_q.step[2],
        "finalized": new_q.finalized,
        "choice_meta": state.get("choice_meta", {}),
    }
    seq = __context.sequence_id
    new_tl = list(state["timeline"]) + [snapshot]
    new_map = dict(state["rewind_map"])
    new_map[snapshot["key"]] = seq

    return (
        state
        .update(qstate=_to_jsonable(new_q))
        .update(timeline=new_tl, rewind_map=new_map, step_index=int(state["step_index"]) + 1)
    )

@action(reads=["qstate"], writes=["parallel_queue"])
def schedule_parallel(state: State, __context: ApplicationContext) -> State:
    """
    Extract parallel seeds (from RunContext inside qstate) into Burr state queue.
    Burr parallelism (Map/Reduce) can pick these and run branches concurrently.
    """
    q = _qstate_from(state["qstate"])
    pq = sorted(list(q.ctx.parallel))
    return state.update(parallel_queue=pq)

@action(reads=["qstate"], writes=["final_report"])
def maybe_finalize(state: State) -> State:
    q = _qstate_from(state["qstate"])
    if q.finalized:
        return state.update(final_report={"code": q.current})
    return state

# ---------------- Human/LLM feedback + exemplar archiving + optimization --------

@action(reads=["qstate", "timeline"], writes=["timeline"])
def record_feedback(state: State, payload: dict[str, Any]) -> State:
    """
    Human (or LLM) feedback payload, appended to timeline,
    and forwarded to router.feedback() if available.
    """
    tl = list(state["timeline"])
    tl.append({"event": "feedback", "payload": payload})
    router = _policy()
    if router:
        router.feedback(payload | {"node_code": _qstate_from(state["qstate"]).current})
    return state.update(timeline=tl)

@action(reads=["qstate", "timeline"], writes=["timeline"])
def archive_exemplar(state: State, example: dict[str, Any]) -> State:
    """
    Archive a training example for tuning (prompt or model).
    The router can send it to a global sink and/or an agent-specific store.
    """
    q = _qstate_from(state["qstate"])
    ex = example | {
        "node_code": q.current,
        "qstate": _to_jsonable(q),
    }
    tl = list(state["timeline"])
    tl.append({"event": "archive_exemplar", "example": ex})
    router = _policy()
    if router:
        router.archive(ex)
    return state.update(timeline=tl)

@action(reads=["qstate", "timeline"], writes=["timeline"])
def optimize_node_policy(state: State, dataset_ref: str | None = None) -> State:
    """
    Trigger per-node optimization (e.g., DSPy self-optimization) via the router.
    """
    q = _qstate_from(state["qstate"])
    router = _policy()
    if router:
        router.optimize(q.current, dataset_ref)
    tl = list(state["timeline"])
    tl.append({"event": "optimize_policy", "node_code": q.current, "dataset_ref": dataset_ref})
    return state.update(timeline=tl)

# =================================================================================
# Build Burr app
# =================================================================================

def build_app(
    *,
    project: str = "icd-dag",
    tracking: bool = True,
) -> Any:
    tracker = LocalTrackingClient(project=project) if tracking else None
    builder = ApplicationBuilder()
    if tracker:
        builder = builder.with_tracker(tracker, use_otel_tracing=True)

    app = (
        builder
        .with_actions(
            init_traversal,
            enumerate_candidates,
            select_candidate,
            apply_selected_step,
            schedule_parallel,
            maybe_finalize,
            record_feedback,
            archive_exemplar,
            optimize_node_policy,
            terminal=Result("final_report"),
        )
        .with_transitions(
            ("init_traversal", "enumerate_candidates"),
            ("enumerate_candidates", "select_candidate"),
            ("select_candidate", "apply_selected_step"),
            ("apply_selected_step", "schedule_parallel"),
            ("schedule_parallel", "maybe_finalize"),
            ("maybe_finalize", "terminal", expr("final_report is not None")),
            ("maybe_finalize", "enumerate_candidates", expr("final_report is None")),
        )
        .with_entrypoint("init_traversal")
        .with_state()
        .build()
    )
    return app
