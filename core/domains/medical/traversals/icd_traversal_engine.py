"""
ICD Traversal Engine Implementation

This module provides the TraversalEngine implementation for ICD-10-CM that bridges
ICD-specific rules and data structures (from icd_rules.py) to the core architecture.

Architecture:
- icd_rules.py: Contains ICD-specific business logic, data structures, and functions
- icd_traversal_engine.py: Implements TraversalEngine protocol, adapts ICD rules to core types
"""

from __future__ import annotations
from typing import Any

from core import (
    NodeId, Action, GuardOutcome, GuardResult, 
    QueryState, RunContext, TreeIndex, CORE_API_VERSION
)

# Import ICD-specific rules and data structures
from .icd_rules import (
    Action as ICDAction, QueryState as ICDQueryState, RunContext as ICDRunContext, 
    ActionStep, GuardResult as ICDGuardResult,
    candidate_actions, guard_next, apply_step, ingest_notes_into_context, resolve
)


def _convert_action_to_core(icd_action: ICDAction) -> Action:
    """Convert ICD Action to Core Action."""
    mapping = {
        ICDAction.goto: Action.GOTO,
        ICDAction.report: Action.REPORT,
        ICDAction.exit: Action.EXIT,
        ICDAction.memory: Action.MEMORY,
        ICDAction.fork: Action.FORK,
    }
    return mapping.get(icd_action, Action.GOTO)


def _convert_action_from_core(core_action: Action) -> ICDAction:
    """Convert Core Action to ICD Action."""
    mapping = {
        Action.GOTO: ICDAction.goto,
        Action.REPORT: ICDAction.report,
        Action.EXIT: ICDAction.exit,
        Action.MEMORY: ICDAction.memory,
        Action.FORK: ICDAction.fork,
    }
    return mapping.get(core_action, ICDAction.goto)


def _convert_guard_result_to_core(icd_result: ICDGuardResult) -> GuardResult:
    """Convert ICD GuardResult to Core GuardResult."""
    # Map ICD guard outcomes to core outcomes
    outcome_mapping = {
        "allow": GuardOutcome.ALLOW,
        "block": GuardOutcome.BLOCK,
        "uproot": GuardOutcome.UPROOT,
        "require_prefix": GuardOutcome.REQUIRE_PREFIX,
        "require_suffix": GuardOutcome.REQUIRE_SUFFIX,
        "require_seven": GuardOutcome.REQUIRE_SEVEN,
    }
    
    outcome = outcome_mapping.get(icd_result.outcome, GuardOutcome.BLOCK)
    
    return GuardResult(
        outcome=outcome,
        message=icd_result.message,
        details=getattr(icd_result, 'details', None)
    )


class ICDTraversalEngine:
    """
    ICD TraversalEngine implementation that implements the TraversalEngine protocol.
    
    This implementation wraps the existing ICD traversal functions and provides them
    through structural typing, implementing ICD-10 specific traversal rules while
    working with any TreeIndex. Uses composition over inheritance for flexibility.
    """
    
    implements_core_version = CORE_API_VERSION
    
    def _get_navigator(self, tree: TreeIndex):
        """Extract the underlying ICDTreeNavigator from the TreeIndex."""
        # After refactoring, ICDTreeNavigator implements TreeIndex directly
        if hasattr(tree, 'navigator'):
            return tree.navigator  # Legacy wrapper support
        elif hasattr(tree, 'find_by_code') and hasattr(tree, 'code_to_node'):
            return tree  # Direct ICDTreeNavigator
        else:
            raise ValueError("TreeIndex must be ICDTreeNavigator or have navigator attribute")
    
    def dump_state(self, q: QueryState) -> dict[str, Any]:
        """Serialize QueryState to JSON-compatible dict."""
        return {
            "current": str(q.current),
            "finalized": q.finalized,
            "step": [
                str(q.step[0]) if q.step else None,
                q.step[1].value if q.step else None,
                str(q.step[2]) if q.step else None,
            ] if q.step else None,
            "ctx": self.dump_context(q.ctx)
        }
    
    def load_state(self, d: dict[str, Any]) -> QueryState:
        """Deserialize QueryState from JSON-compatible dict."""
        step = None
        if d.get("step") and all(x is not None for x in d["step"]):
            step = (
                NodeId(d["step"][0]),
                Action(d["step"][1]), 
                NodeId(d["step"][2])
            )
        
        return QueryState(
            current=NodeId(d["current"]),
            finalized=d.get("finalized", False),
            step=step,
            ctx=self.load_context(d.get("ctx", {}))
        )
    
    def dump_context(self, c: RunContext) -> dict[str, Any]:
        """Serialize RunContext to JSON-compatible dict."""
        return c.data
    
    def load_context(self, d: dict[str, Any]) -> RunContext:
        """Deserialize RunContext from JSON-compatible dict."""
        return RunContext(data=d)
    
    def ingest(self, tree: TreeIndex, node_id: NodeId, ctx: RunContext) -> RunContext:
        """Fold node notes/constraints into context when entering a node."""
        navigator = self._get_navigator(tree)
        node = resolve(navigator, str(node_id))
        
        # Convert RunContext to ICDRunContext for existing function
        icd_ctx = ICDRunContext(**ctx.data)
        
        # Call existing ingest function
        ingest_notes_into_context(icd_ctx, node)
        
        # Convert back to RunContext
        return RunContext(data=icd_ctx.model_dump())
    
    def candidate_actions(self, tree: TreeIndex, q: QueryState) -> list[tuple[NodeId, Action]]:
        """Get legal next moves from current state."""
        navigator = self._get_navigator(tree)
        
        # Convert to ICD types
        icd_ctx = ICDRunContext(**q.ctx.data)
        step = None
        if q.step:
            step = ActionStep((
                str(q.step[0]),
                _convert_action_from_core(q.step[1]),
                str(q.step[2])
            ))
        
        icd_state = ICDQueryState(
            current=str(q.current),
            finalized=q.finalized,
            step=step or ActionStep((str(q.current), ICDAction.goto, str(q.current))),
            ctx=icd_ctx
        )
        
        # Get candidates from existing function
        candidates = candidate_actions(icd_state, navigator)
        
        # Convert back to core types
        core_candidates = []
        for node_id_str, icd_action in candidates:
            core_action = _convert_action_to_core(icd_action)
            core_candidates.append((NodeId(node_id_str), core_action))
        
        return core_candidates
    
    def guard_next(self, tree: TreeIndex, q: QueryState, move: tuple[NodeId, Action]) -> GuardResult:
        """Guard a proposed move against domain rules."""
        navigator = self._get_navigator(tree)
        
        # Convert to ICD types
        icd_ctx = ICDRunContext(**q.ctx.data)
        step = None
        if q.step:
            step = ActionStep((
                str(q.step[0]),
                _convert_action_from_core(q.step[1]),
                str(q.step[2])
            ))
        
        icd_state = ICDQueryState(
            current=str(q.current),
            finalized=q.finalized,
            step=step or ActionStep((str(q.current), ICDAction.goto, str(q.current))),
            ctx=icd_ctx
        )
        
        target_str, core_action = move
        icd_action = _convert_action_from_core(core_action)
        
        # Call existing guard function
        icd_result = guard_next(icd_state, str(target_str), icd_action, navigator)
        
        # Convert result
        return _convert_guard_result_to_core(icd_result)
    
    def apply(self, tree: TreeIndex, q: QueryState, move: tuple[NodeId, Action]) -> QueryState | GuardResult:
        """Apply move if allowed, returning new QueryState or GuardResult if blocked."""
        navigator = self._get_navigator(tree)
        
        # Convert to ICD types
        icd_ctx = ICDRunContext(**q.ctx.data)
        step = None
        if q.step:
            step = ActionStep((
                str(q.step[0]),
                _convert_action_from_core(q.step[1]),
                str(q.step[2])
            ))
        
        icd_state = ICDQueryState(
            current=str(q.current),
            finalized=q.finalized,
            step=step or ActionStep((str(q.current), ICDAction.goto, str(q.current))),
            ctx=icd_ctx
        )
        
        target_str, core_action = move
        icd_action = _convert_action_from_core(core_action)
        instruction = (str(target_str), icd_action)
        
        # Apply step using existing function
        result = apply_step(icd_state, instruction, navigator)
        
        if isinstance(result, ICDGuardResult):
            return _convert_guard_result_to_core(result)
        else:
            # Convert ICDQueryState back to QueryState
            new_step = None
            if result.step:
                new_step = (
                    NodeId(result.step[0]),
                    _convert_action_to_core(result.step[1]),
                    NodeId(result.step[2])
                )
            
            return QueryState(
                current=NodeId(result.current),
                finalized=result.finalized,
                step=new_step,
                ctx=RunContext(data=result.ctx.model_dump())
            )
    
    def can_attempt_finalize(self, tree: TreeIndex, q: QueryState) -> bool:
        """Check if current state allows attempting finalization."""
        # Simple heuristic: can finalize if we have a current node and it's not a chapter/block
        node_view = tree.get(q.current)
        if node_view is None:
            return False
        
        # Don't finalize on chapters or blocks
        if node_view.element_type in ["chapter", "block"]:
            return False
        
        return True
    
    def parallel_seeds(self, tree: TreeIndex, q: QueryState) -> list[NodeId]:
        """Get seeds for parallel branches (e.g., ICD codeAlso)."""
        # Extract codeAlso hints from context
        code_also_list = q.ctx.data.get("code_also", [])
        return [NodeId(code) for code in code_also_list]


# Factory function for creating the ICD traversal engine
def create_icd_traversal_engine() -> ICDTraversalEngine:
    """Create an ICDTraversalEngine."""
    return ICDTraversalEngine()