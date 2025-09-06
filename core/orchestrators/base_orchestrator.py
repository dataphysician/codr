"""
Base Orchestrator Implementation

This module provides the foundational orchestrator that demonstrates
core DAG traversal patterns. This is the base implementation that all
other orchestrators should extend or use as reference.
"""

from __future__ import annotations
import time
from typing import Any

from core import NodeId, Action, QueryState, RunContext, DecisionContext, TreeIndex, TraversalEngine, CandidateAgent


class BaseOrchestrator:
    """
    Base orchestrator demonstrating core DAG traversal operations.
    
    This orchestrator implements the fundamental traverse → decide → apply → finalize
    pattern that all other orchestrators should follow. Provides the essential
    primitives that any framework can use or extend.
    """
    
    def __init__(self, tree: TreeIndex, traversal: TraversalEngine, agent: CandidateAgent):
        self.tree = tree
        self.traversal = traversal
        self.agent = agent
        self.history: list[dict[str, Any]] = []
    
    def enumerate_candidates(self, state: QueryState) -> list[tuple[NodeId, Action]]:
        """Get legal next moves from current state."""
        return self.traversal.candidate_actions(self.tree, state)
    
    def decide(self, state: QueryState, candidates: list[tuple[NodeId, Action]]) -> list[tuple[NodeId, Action]]:
        """Use agent to select from candidates."""
        if not candidates:
            return []
        
        # Build decision context
        node = self.tree.get(state.current)
        if not node:
            return []
        
        # Get children for context
        children = [self.tree.get(cid) for cid in self.tree.children(state.current)]
        children = [c for c in children if c is not None]
        
        # Get ancestors for context
        ancestors = [self.tree.get(aid) for aid in self.tree.ancestors(state.current)]
        ancestors = [a for a in ancestors if a is not None]
        
        decision_ctx = DecisionContext(
            node=node,
            ancestors=ancestors,
            children=children,
            path=self.tree.path_to_root(state.current),
            allowed_actions=[Action.GOTO, Action.REPORT, Action.EXIT],
            pending_constraints=state.ctx.data,
            external_context={}
        )
        
        # Get agent selections
        rich_candidates = self.agent.select(decision_ctx, candidates)
        
        # Extract top selections (convert back to simple tuples)
        return [(rc.target, rc.action) for rc in rich_candidates]
    
    def try_step(self, state: QueryState, move: tuple[NodeId, Action]) -> QueryState | dict[str, Any]:
        """Attempt to apply a move, returning new state or error info."""
        result = self.traversal.apply(self.tree, state, move)
        
        if hasattr(result, 'current'):  # It's a QueryState
            # Record step in history
            step_info = {
                "from": str(state.current),
                "action": move[1].value,
                "to": str(result.current),
                "timestamp": time.time(),
                "finalized": result.finalized
            }
            self.history.append(step_info)
            
            # Update context with new node
            updated_ctx = self.traversal.ingest(self.tree, result.current, result.ctx)
            return QueryState(
                current=result.current,
                finalized=result.finalized,
                step=result.step,
                ctx=updated_ctx
            )
        else:  # It's a GuardResult
            return {
                "error": "move_blocked",
                "outcome": result.outcome.value,
                "message": result.message,
                "details": result.details
            }
    
    def attempt_finalize(self, state: QueryState) -> QueryState | dict[str, Any]:
        """Attempt to finalize current state."""
        if not self.traversal.can_attempt_finalize(self.tree, state):
            return {
                "error": "cannot_finalize",
                "message": "Current state does not allow finalization",
                "can_finalize": False
            }
        
        finalize_move = (state.current, Action.REPORT)
        return self.try_step(state, finalize_move)
    
    def get_parallel_seeds(self, state: QueryState) -> List[NodeId]:
        """Get parallel branch seeds."""
        return self.traversal.parallel_seeds(self.tree, state)
    
    def execute_workflow(
        self, 
        initial_node: NodeId, 
        initial_context: dict[str, Any] | None = None,
        max_steps: int = 10
    ) -> dict[str, Any]:
        """
        Execute a complete workflow from start to finish.
        
        Args:
            initial_node: Starting node ID
            initial_context: Initial context data
            max_steps: Maximum steps to prevent infinite loops
            
        Returns:
            Complete workflow results
        """
        # Initialize state
        from core import RunContext
        
        initial_ctx = RunContext(data=initial_context or {})
        state = QueryState(
            current=initial_node,
            finalized=False,
            step=None,
            ctx=initial_ctx
        )
        
        # Update context with initial node
        state = QueryState(
            current=state.current,
            finalized=state.finalized,
            step=state.step,
            ctx=self.traversal.ingest(self.tree, state.current, state.ctx)
        )
        
        results = {
            "initial_node": str(initial_node),
            "steps": [],
            "final_state": None,
            "success": False,
            "error": None
        }
        
        step_count = 0
        
        while step_count < max_steps and not state.finalized:
            step_count += 1
            
            # Get candidates
            candidates = self.enumerate_candidates(state)
            if not candidates:
                break
            
            # Let agent decide
            decisions = self.decide(state, candidates)
            if not decisions:
                break
            
            # Try the first decision
            move = decisions[0]
            result = self.try_step(state, move)
            
            if isinstance(result, dict):  # Error occurred
                results["error"] = result
                break
            else:  # Successful step
                state = result
                results["steps"].append({
                    "step": step_count,
                    "move": f"{move[0]} ({move[1].value})",
                    "success": True
                })
        
        results["final_state"] = {
            "current": str(state.current),
            "finalized": state.finalized,
            "step_count": step_count
        }
        results["success"] = state.finalized or step_count < max_steps
        
        return results