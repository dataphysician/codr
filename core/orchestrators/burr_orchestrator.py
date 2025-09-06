from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Pattern, Protocol
import re

# Import new core interfaces instead of legacy direct imports
from core import (
    NodeId, Action, QueryState, RunContext, DecisionContext,
    TreeIndex, TraversalEngine, CandidateAgent, RichCandidate
)

# Import Burr dependencies (only used for the legacy Burr app builder)
try:
    from burr.core import action, State, ApplicationBuilder, ApplicationContext, Result, expr
    from burr.tracking.client import LocalTrackingClient
    BURR_AVAILABLE = True
except ImportError:
    BURR_AVAILABLE = False

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

_TREE: TreeIndex | None = None
_ENGINE: TraversalEngine | None = None
_POLICY: NodePolicyRouter | None = None

def set_policy_router(router: NodePolicyRouter) -> None:
    global _POLICY
    _POLICY = router

def _policy() -> NodePolicyRouter | None:
    return _POLICY

def _tree() -> TreeIndex:
    global _TREE
    if _TREE is None:
        from core.domains.medical.trees.icd_tree import create_navigator
        _TREE = create_navigator()
    return _TREE

def _engine() -> TraversalEngine:
    global _ENGINE
    if _ENGINE is None:
        from core.domains.medical.traversals.icd_traversal_engine import create_icd_traversal_engine
        _ENGINE = create_icd_traversal_engine()
    return _ENGINE

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
    # Use the new engine to deserialize QueryState
    return _engine().load_state(d)

def _rctx_from(d: dict[str, Any]) -> RunContext:
    # Use the new engine to deserialize RunContext
    return _engine().load_context(d)

# =================================================================================
# Burr actions: init, enumerate, select (per-node policy), apply, parallel, feedback
# =================================================================================

@action(reads=[], writes=["qstate", "timeline", "step_index", "rewind_map", "chapter_selection_mode"])
def init_traversal(state: State, start_code: str = "ROOT", clinical_note: str = "") -> State:
    tree = _tree()
    engine = _engine()
    
    # Enable Chapter selection mode when starting from ROOT
    chapter_selection_mode = start_code == "ROOT"
    
    if chapter_selection_mode:
        # Start at ROOT for Chapter selection
        node_id = NodeId("ROOT")
        node = tree.get(node_id)
    else:
        # Normal operation with specific start code
        node_id = tree.normalize_id(start_code)
        node = tree.get(node_id)
    
    ctx = RunContext(data={"clinical_note": clinical_note})
    if node:
        ctx = engine.ingest(tree, node_id, ctx)
    q = QueryState(
        current=node_id,
        finalized=False,
        step=None,
        ctx=ctx,
    )
    return (
        state
        .update(
            qstate=_to_jsonable(q),
            timeline=[],
            step_index=0,
            rewind_map={},
            chapter_selection_mode=chapter_selection_mode,
        )
    )

@action(reads=["qstate", "chapter_selection_mode"], writes=["candidates", "chapter_choices"])
def enumerate_candidates(state: State, __context: ApplicationContext) -> State:
    tree = _tree()
    engine = _engine()
    q = _qstate_from(state["qstate"])
    
    # If in Chapter selection mode, provide Chapter candidates
    chapter_selection_mode = state.get("chapter_selection_mode", False)
    
    if chapter_selection_mode:
        # Get all 22 ICD-10 chapters for selection
        from core.domains.medical.trees.icd_tree import get_chapters_for_selection
        chapters = get_chapters_for_selection(tree)
        
        # Create Chapter selection candidates using goto action with chapter_ prefix
        chapter_candidates = []
        for chapter in chapters:
            chapter_candidates.append((f"chapter_{chapter['code']}", "goto"))
        
        return state.update(
            candidates=chapter_candidates,
            chapter_choices=chapters  # Store full Chapter info for agent context
        )
    else:
        # Normal tree traversal candidates
        cands = engine.candidate_actions(tree, q)
        packed = [(str(node_id), act.value) for (node_id, act) in cands]
        return state.update(candidates=packed, chapter_choices=[])

@action(reads=["qstate", "candidates", "chapter_selection_mode", "chapter_choices"], writes=["choice", "choice_meta", "choice_action"])
def select_candidate(state: State, __context: ApplicationContext, top_k: int | None = None, **agent_inputs: Any) -> State:
    """
    Per-node policy selection:
      - Route to an Agent via NodePolicyRouter
      - Agent returns ordered RichCandidates with metadata
      - We pick the first (or you can extend to multi-pick)
    If no agent, falls back to a simple deterministic heuristic.
    """
    tree = _tree()
    q = _qstate_from(state["qstate"])
    base: list[tuple[str, Action]] = [(c, Action(a)) for (c, a) in state.get("candidates", [])]
    router = _policy()
    
    # Handle Chapter selection mode
    chapter_selection_mode = state.get("chapter_selection_mode", False)
    chapter_choices = state.get("chapter_choices", [])

    selected: RichCandidate | None = None
    
    if chapter_selection_mode and chapter_choices:
        # Special handling for Chapter selection
        if router:
            agent = router.get_agent_for("ROOT")  # Use ROOT policy for Chapter selection
            if agent:
                # Provide Chapter context to the agent
                from core.domains.medical.trees.icd_tree import get_node_details
                
                # Create Chapter decision context
                chapter_context = {
                    "available_chapters": chapter_choices,
                    "mode": "chapter_selection",
                    "clinical_note": q.ctx.data.get("clinical_note", ""),
                    "instruction": "Select the most appropriate ICD-10 chapter based on the clinical scenario."
                }
                
                # Build decision context for Chapter selection
                decision_ctx = DecisionContext(
                    node=tree.get(NodeId("ROOT")),
                    ancestors=[],
                    children=[],  # Will be populated with chapters
                    path=[NodeId("ROOT")],
                    allowed_actions=[Action.GOTO],
                    pending_constraints=chapter_context,
                    external_context={"agent_inputs": agent_inputs}
                )
                
                # Use base candidates which are already properly formatted
                rich = agent.select(decision_ctx, base)
                if rich:
                    selected = rich[0]
    
    elif router:
        agent = router.get_agent_for(str(q.current))
        if agent:
            # Build decision context for new architecture
            tree = _tree()
            node = tree.get(q.current)
            if node:
                children = [tree.get(cid) for cid in tree.children(q.current)]
                children = [c for c in children if c is not None]
                ancestors = [tree.get(aid) for aid in tree.ancestors(q.current)]
                ancestors = [a for a in ancestors if a is not None]
                decision_ctx = DecisionContext(
                    node=node,
                    ancestors=ancestors,
                    children=children,
                    path=tree.path_to_root(q.current),
                    allowed_actions=[Action.GOTO, Action.REPORT, Action.EXIT],
                    pending_constraints=q.ctx.data,
                    external_context={"agent_inputs": agent_inputs}
                )
                rich = agent.select(decision_ctx, base)
                if rich:
                    # Check if agent suggests multiple high-quality parallel candidates
                    top_candidates = []
                    if len(rich) > 1:
                        # Check if multiple candidates have similar high ranking/quality
                        first_quality = rich[0].metadata.get("reasoning_quality", "MODERATE") if rich[0].metadata else "MODERATE"
                        for candidate in rich[:3]:  # Consider up to 3 top candidates
                            candidate_quality = candidate.metadata.get("reasoning_quality", "MODERATE") if candidate.metadata else "MODERATE"
                            # Include if high quality or similar to first candidate
                            if (candidate_quality in ["EXCELLENT", "GOOD"] or 
                                candidate_quality == first_quality):
                                top_candidates.append(candidate)
                    
                    # If multiple high-quality candidates, schedule them for parallel processing
                    if len(top_candidates) > 1:
                        # Select first candidate for immediate execution
                        selected = top_candidates[0]
                        # Schedule others as parallel seeds by updating context
                        parallel_seeds = [str(c.target) for c in top_candidates[1:]]
                        if parallel_seeds:
                            print(f"🤖 Agent identified {len(top_candidates)} high-quality candidates")
                            print(f"   Primary: {selected.target} (quality: {selected.metadata.get('reasoning_quality', 'N/A')})")
                            print(f"   Parallel: {parallel_seeds}")
                            
                            # Add to parallel queue for later scheduling
                            current_parallel = state.get("parallel_queue", [])
                            updated_parallel = list(set(current_parallel + parallel_seeds))
                            # Update state to include parallel candidates
                            state = state.update(parallel_queue=updated_parallel)
                    else:
                        selected = rich[0]

    if selected is None:
        # Heuristic fallback: prefer leaves then fewest children then lexical
        tree = _tree()
        def rank_key(ca):
            node_id = tree.normalize_id(ca[0])
            node = tree.get(node_id)
            if node:
                children = tree.children(node_id)
                return (1 if len(children) > 0 else 0, len(children), str(node_id))
            return (1, 9999, str(node_id))
        
        ranked = sorted(base, key=rank_key)
        if ranked:
            code, act = ranked[0]
            selected = RichCandidate(target=tree.normalize_id(code), action=act, metadata={"policy": "heuristic"})
        else:
            selected = RichCandidate(target=q.current, action=Action.REPORT, metadata={"reason": "no candidates; report"})

    return state.update(choice=str(selected.target), choice_action=selected.action.value, choice_meta=selected.metadata or {})

@action(reads=["qstate", "choice", "choice_action", "timeline", "step_index", "rewind_map", "chapter_selection_mode", "chapter_choices"],
        writes=["qstate", "timeline", "step_index", "rewind_map", "last_guard", "chapter_selection_mode"])
def apply_selected_step(state: State, __context: ApplicationContext) -> State:
    """
    Apply selected step using new architecture.
    """
    tree = _tree()
    engine = _engine()
    q = _qstate_from(state["qstate"])
    target_raw = str(state["choice"])
    action = Action(state["choice_action"])
    
    # Handle Chapter selection transition
    chapter_selection_mode = state.get("chapter_selection_mode", False)
    chapter_choices = state.get("chapter_choices", [])
    
    if chapter_selection_mode and target_raw.startswith("chapter_"):
        # Chapter was selected - extract chapter number and convert to actual Chapter navigation
        chapter_code = target_raw.replace("chapter_", "")
        selected_chapter = None
        
        for chapter in chapter_choices:
            if chapter["code"] == chapter_code:
                selected_chapter = chapter
                break
        
        if selected_chapter:
            # Navigate to the selected Chapter by using the chapter's actual code from the tree
            chapter_index = int(selected_chapter["code"]) - 1  # Convert to 0-based index
            
            # Find the actual chapter node using the navigator's internal structure
            if hasattr(tree, 'root') and tree.root and hasattr(tree.root, 'children'):
                chapter_nodes = [child for child in tree.root.children if getattr(child, 'element_type', None) == 'chapter']
                
                if 0 <= chapter_index < len(chapter_nodes):
                    chapter_node = chapter_nodes[chapter_index]
                    
                    # Navigate to the chapter itself (by chapter number), not to a specific code within it
                    # This preserves the proper parent-child pipeline
                    target_node_id = NodeId(str(chapter_index + 1))
                    
                    # Update query state to selected Chapter's first code
                    new_ctx = RunContext(data=q.ctx.data)
                    new_q = QueryState(
                        current=target_node_id,
                        finalized=False,
                        step=None,
                        ctx=new_ctx,
                    )
                    
                    # Exit Chapter selection mode and update state
                    return (
                        state
                        .update(
                            qstate=_to_jsonable(new_q),
                            timeline=state.get("timeline", []) + [{
                                "key": f"ROOT|chapter_select|{target_node_id}",
                                "current": "ROOT",
                                "action": "chapter_select", 
                                "next": str(target_node_id),
                                "explain": f"Selected Chapter: {selected_chapter['name']}",
                                "finalized": False,
                                "choice_meta": state.get("choice_meta", {}),
                            }],
                            step_index=state.get("step_index", 0) + 1,
                            chapter_selection_mode=False,  # Exit Chapter selection mode
                            last_guard={},
                        )
                    )
            
            # Fallback: exit Chapter selection mode even if navigation fails
            return (
                state
                .update(
                    chapter_selection_mode=False,
                    timeline=state.get("timeline", []) + [{
                        "key": f"ROOT|chapter_select_failed|{selected_chapter['code']}",
                        "current": "ROOT", 
                        "action": "chapter_select_failed",
                        "next": selected_chapter["code"],
                        "explain": f"Chapter selection failed: {selected_chapter['name']}",
                        "finalized": False,
                        "choice_meta": state.get("choice_meta", {}),
                    }],
                    step_index=state.get("step_index", 0) + 1,
                    last_guard={"outcome": "failed", "message": "Chapter navigation not available"},
                )
            )
    
    # Normal tree traversal (non-Chapter selection)
    target_id = tree.normalize_id(target_raw)

    # Try to apply the move
    move = (target_id, action)
    result = engine.apply(tree, q, move)
    
    # Check if it's a GuardResult (blocked)
    if hasattr(result, 'outcome'):
        # It's a GuardResult, move was blocked
        gr = result
        state = state.update(last_guard={"outcome": gr.outcome.value, "message": gr.message})
        
        snapshot = {
            "key": f"{q.current}|{action.value}|{target_id}",
            "current": str(q.current),
            "action": action.value,
            "next": str(target_id),
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

    # Move was successful, result is a new QueryState
    new_q = result
    # Update context with new node
    new_q = QueryState(
        current=new_q.current,
        finalized=new_q.finalized,
        step=new_q.step,
        ctx=engine.ingest(tree, new_q.current, new_q.ctx)
    )
    
    snapshot = {
        "key": f"{q.current}|{action.value}|{new_q.current}",
        "current": str(new_q.current),
        "action": action.value,
        "next": str(new_q.current),
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
        .update(timeline=new_tl, rewind_map=new_map, step_index=int(state["step_index"]) + 1, last_guard=None)
    )

@action(reads=["qstate"], writes=["parallel_queue", "parallel_results"])
def schedule_parallel(state: State, __context: ApplicationContext) -> State:
    """
    Execute parallel branches using async map-reduce pattern.
    True parallel execution for useAdditionalCode and codeAlso requirements.
    """
    import asyncio
    from .async_parallel_executor import AsyncParallelExecutor, ParallelBranch
    
    tree = _tree()
    engine = _engine()
    policy = _policy()
    q = _qstate_from(state["qstate"])
    
    # Get default agent from policy router (fallback to basic agent if needed)
    agent = None
    if policy and hasattr(policy, 'default_agent') and policy.default_agent:
        agent = policy.default_agent
    else:
        # Fallback: create basic deterministic agent for parallel execution
        from core.dag_agents.base_agents import DeterministicAgent
        agent = DeterministicAgent()
    
    # Create async executor with Burr context for tracking
    executor = AsyncParallelExecutor(tree, engine, agent, burr_context=__context)
    
    # Extract parallel branches from current state
    branches = executor.extract_parallel_branches(q)
    
    if branches:
        print(f"\n🚀 ASYNC MAP-REDUCE: Executing {len(branches)} parallel branches")
        
        # Execute async map-reduce
        try:
            # Run async execution in current thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            map_reduce_result = loop.run_until_complete(
                executor.execute_parallel_branches(q, branches)
            )
            loop.close()
            
            # Update state with parallel execution results
            updated_qstate = map_reduce_result.primary_result or q
            
            # Get tracking summary for Burr UI
            tracking_summary = executor.get_burr_tracking_summary()
            
            parallel_results = {
                "total_branches": len(branches),
                "successful_branches": map_reduce_result.completed_branches,
                "failed_branches": map_reduce_result.failed_branches,
                "execution_time_ms": map_reduce_result.total_execution_time_ms,
                "results": [
                    {
                        "target_code": r.branch.target_code,
                        "branch_type": r.branch.branch_type,
                        "success": r.success,
                        "final_code": r.final_code,
                        "execution_time_ms": r.execution_time_ms
                    }
                    for r in map_reduce_result.parallel_results
                ]
            }
            
            # Add tracking summary if available
            if tracking_summary:
                parallel_results["burr_tracking"] = tracking_summary
            
            return state.update(
                qstate=_to_jsonable(updated_qstate),
                parallel_queue=[],  # Clear queue after execution
                parallel_results=parallel_results
            )
            
        except Exception as e:
            print(f"❌ Async parallel execution failed: {e}")
            return state.update(
                parallel_queue=[],
                parallel_results={"error": str(e)}
            )
    else:
        # No parallel branches needed
        return state.update(
            parallel_queue=[],
            parallel_results={"message": "No parallel branches required"}
        )

@action(reads=["qstate"], writes=["final_report"])
def maybe_finalize(state: State) -> State:
    q = _qstate_from(state["qstate"])
    if q.finalized:
        return state.update(final_report={"code": str(q.current)})
    return state.update(final_report=None)

# ---------------- Human/LLM feedback + rewind + exemplar archiving + optimization --------

@action(reads=["qstate", "timeline", "rewind_map"], writes=["qstate", "timeline"])
def rewind_to_node(state: State, target_node: str, feedback: dict[str, Any] | None = None) -> State:
    """
    Rewind to a specific node in the traversal timeline with optional feedback.
    
    This function:
    1. Finds the target node in the timeline
    2. Reverts the query state to that point
    3. Adds feedback to the context for improved agent decision-making
    4. Preserves clinical context and adds code provenance information
    
    Args:
        target_node: The node ID to rewind to (e.g., "E10", "chapter_4")
        feedback: Optional feedback dict with correction information
    """
    timeline = state["timeline"]
    current_q = _qstate_from(state["qstate"])
    
    # Find the target node in timeline
    target_snapshot = None
    target_index = None
    
    for i, snapshot in enumerate(timeline):
        if snapshot.get("current") == target_node or snapshot.get("next") == target_node:
            target_snapshot = snapshot
            target_index = i
            break
    
    if target_snapshot is None:
        # If not found in timeline, try to create a fresh state at target node
        tree = _tree()
        engine = _engine()
        target_id = tree.normalize_id(target_node)
        target_node_obj = tree.get(target_id)
        
        if target_node_obj:
            # Create new query state at target node, preserving clinical context
            original_clinical_note = current_q.ctx.data.get("clinical_note", "")
            new_ctx = RunContext(data={"clinical_note": original_clinical_note})
            new_ctx = engine.ingest(tree, target_id, new_ctx)
            
            # Add feedback and provenance to context
            if feedback:
                new_ctx.data.update({
                    "feedback": feedback,
                    "rewind_feedback": feedback,
                    "previous_path": [str(s.get("current", "")) for s in timeline],
                    "rewind_from": str(current_q.current),
                    "rewind_reason": feedback.get("reason", "Manual rewind")
                })
            
            rewound_q = QueryState(
                current=target_id,
                finalized=False,
                step=None,
                ctx=new_ctx
            )
            
            # Add rewind event to timeline
            rewind_event = {
                "event": "rewind",
                "target_node": target_node,
                "rewind_from": str(current_q.current),
                "feedback": feedback or {},
                "timestamp": time.time() if hasattr(__builtins__, 'time') else 0
            }
            
            new_timeline = list(timeline) + [rewind_event]
            
            return state.update(
                qstate=_to_jsonable(rewound_q),
                timeline=new_timeline
            )
        else:
            # Invalid target node, record the attempt but don't change state
            error_event = {
                "event": "rewind_failed",
                "target_node": target_node,
                "error": f"Node {target_node} not found",
                "feedback": feedback or {}
            }
            new_timeline = list(timeline) + [error_event]
            return state.update(timeline=new_timeline)
    
    # Rewind to found snapshot
    tree = _tree()
    engine = _engine()
    target_id = tree.normalize_id(target_snapshot["current"])
    
    # Reconstruct query state at target point, preserving clinical context
    original_clinical_note = current_q.ctx.data.get("clinical_note", "")
    rewound_ctx = RunContext(data={"clinical_note": original_clinical_note})
    rewound_ctx = engine.ingest(tree, target_id, rewound_ctx)
    
    # Enhance context with feedback and provenance
    if feedback:
        rewound_ctx.data.update({
            "feedback": feedback,
            "rewind_feedback": feedback,
            "previous_path": [str(s.get("current", "")) for s in timeline[:target_index+1]],
            "rewind_from": str(current_q.current),
            "rewind_reason": feedback.get("reason", "Manual rewind with feedback")
        })
    
    rewound_q = QueryState(
        current=target_id,
        finalized=False,
        step=None,  # Clear step to allow new decision
        ctx=rewound_ctx
    )
    
    # Add rewind event to timeline
    rewind_event = {
        "event": "rewind",
        "target_node": target_node,
        "rewind_from": str(current_q.current),
        "timeline_position": target_index,
        "feedback": feedback or {},
        "preserved_clinical_context": bool(original_clinical_note)
    }
    
    # Keep timeline up to rewind point plus the rewind event
    new_timeline = timeline[:target_index+1] + [rewind_event]
    
    return state.update(
        qstate=_to_jsonable(rewound_q),
        timeline=new_timeline
    )

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
        router.feedback(payload | {"node_code": str(_qstate_from(state["qstate"]).current)})
    return state.update(timeline=tl)

@action(reads=["qstate", "timeline"], writes=["timeline"])
def archive_exemplar(state: State, example: dict[str, Any]) -> State:
    """
    Archive a training example for tuning (prompt or model).
    The router can send it to a global sink and/or an agent-specific store.
    """
    q = _qstate_from(state["qstate"])
    ex = example | {
        "node_code": str(q.current),
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
        router.optimize(str(q.current), dataset_ref)
    tl = list(state["timeline"])
    tl.append({"event": "optimize_policy", "node_code": str(q.current), "dataset_ref": dataset_ref})
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
        builder = builder.with_tracker(tracker, use_otel_tracing=False)

    app = (
        builder
        .with_actions(
            init_traversal,
            enumerate_candidates,
            select_candidate,
            apply_selected_step,
            schedule_parallel,
            maybe_finalize,
            rewind_to_node,
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


def start_burr_ui(port: int = 7241) -> None:
    """Start the Burr tracking UI server for visualization."""
    if not BURR_AVAILABLE:
        print("❌ Burr not available. Install with: pip install burr[tracking]")
        return
        
    try:
        # Try to import and start the server directly
        from burr.tracking.server.run import run_server
        import threading
        import time
        
        print(f"🚀 Starting Burr tracking UI on port {port}...")
        print(f"📊 Open browser to: http://localhost:{port}/")
        
        # Start the server in a separate thread
        server_thread = threading.Thread(
            target=lambda: run_server(port=port, host="127.0.0.1"),
            daemon=True
        )
        server_thread.start()
        
        # Give the server time to start
        time.sleep(2)
        
        print(f"✅ Burr UI server started on port {port}")
        print(f"💡 Run your workflow to see execution graphs and traces")
        
    except ImportError as e:
        print(f"⚠️  Burr server module not available: {e}")
        print(f"📊 Tracking data will still be collected locally")
        
        # Try subprocess approach as fallback
        try:
            import subprocess
            import sys
            
            subprocess.Popen([
                sys.executable, "-m", "burr.tracking.server.run", 
                "--host", "127.0.0.1", 
                "--port", str(port)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print(f"⚡ Attempted to start server via subprocess")
            
        except Exception:
            print(f"💡 Install full tracking support: pip install burr[tracking]")
            
    except Exception as e:
        print(f"❌ Failed to start Burr UI: {e}")
        print(f"📊 Tracking will continue without web UI")


def visualize_app_structure(app) -> None:
    """Display the Burr application structure and state transitions."""
    if not BURR_AVAILABLE or not hasattr(app, '_actions'):
        print("❌ Cannot visualize app structure")
        return
        
    print("🏗️  Burr Application Structure:")
    print("=" * 50)
    
    # Show actions
    actions = getattr(app, '_actions', {})
    print(f"📋 Actions ({len(actions)}):")
    for name in actions.keys():
        print(f"  • {name}")
    
    # Show transitions if available
    if hasattr(app, '_graph'):
        transitions = getattr(app._graph, '_transitions', [])
        print(f"\n🔀 Transitions ({len(transitions)}):")
        for transition in transitions[:10]:  # Limit to first 10
            from_action = getattr(transition, 'from_', 'unknown')
            to_action = getattr(transition, 'to', 'unknown') 
            print(f"  {from_action} → {to_action}")
        if len(transitions) > 10:
            print(f"  ... and {len(transitions) - 10} more")
    
    print(f"\n💡 Run workflow to generate execution traces")


# =================================================================================
# Example usage functions for Burr + DSPy integration
# =================================================================================

def create_burr_app(
    agent, 
    project: str = "coding-agent", 
    enable_tracking: bool = True,
    app_id: str | None = None
) -> Any:
    """
    Create a Burr application configured to use any agent via NodePolicyRouter.
    
    Args:
        agent: Any agent implementing CandidateAgent interface
        project: Tracking project name
        enable_tracking: Enable Burr tracking for visualization
        app_id: Optional application ID for tracking
        
    Returns:
        Configured Burr application ready to run
    """
    if not BURR_AVAILABLE:
        raise ImportError("Burr is not available. Install with: pip install burr[tracking]")
    
    # Set up the policy router to use the agent
    router = NodePolicyRouter(default_agent=agent)
    
    # Configure global state for Burr actions to use
    global _POLICY_ROUTER
    _POLICY_ROUTER = router
    
    # Build the Burr app with enhanced tracking
    app = build_app(project=project, tracking=enable_tracking)
    
    if enable_tracking:
        print(f"🔍 Burr tracking enabled for project: {project}")
        print(f"📊 View execution graphs at: http://localhost:7241/")
        print(f"🏗️  App ID: {getattr(app, 'uid', 'auto-generated')}")
    
    return app


def create_burr_app_with_dspy_agent(
    dspy_agent, 
    project: str = "icd-dag-dspy", 
    enable_tracking: bool = True,
    app_id: str | None = None
) -> Any:
    """
    Create a Burr application configured to use a DSPy agent.
    
    DEPRECATED: Use create_burr_app(agent, project) instead.
    This function will be removed in a future version.
    
    Args:
        dspy_agent: DSPy agent implementing CandidateAgent interface
        project: Tracking project name
        enable_tracking: Enable Burr tracking for visualization  
        app_id: Optional application ID for tracking
        
    Returns:
        Configured Burr application ready to run
    """
    import warnings
    warnings.warn(
        "create_burr_app_with_dspy_agent is deprecated. Use create_burr_app(agent, project) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_burr_app(dspy_agent, project, enable_tracking, app_id)
