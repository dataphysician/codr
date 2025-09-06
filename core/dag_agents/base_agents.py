"""
Base agent classes and common implementations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass

# Import from core interfaces
from core import (
    NodeId, Action, DecisionContext, RichCandidate,
    CandidateAgent, CandidateScorer
)


class DeterministicAgent(CandidateAgent):
    """
    Simple deterministic agent using heuristics.
    
    Prefers leaf nodes, then nodes with fewer children, then lexical order.
    Good for testing and as a fallback agent.
    """
    
    def select(
        self,
        decision: DecisionContext,
        candidates: list[tuple[NodeId, Action]]
    ) -> list[RichCandidate]:
        """Select candidates using deterministic heuristics."""
        
        def score_candidate(candidate: tuple[NodeId, Action]) -> tuple[int, int, str]:
            node_id, action = candidate
            
            # Find the target node in children
            target_node = None
            for child in decision.children:
                if child.id == node_id:
                    target_node = child
                    break
            
            if target_node is None:
                # Fallback scoring for non-child targets
                return (1, 9999, str(node_id))
            
            # Score: (is_not_leaf, estimated_child_count, lexical_order)
            # Lower scores are better (will be reversed for ranking)
            is_leaf = len(decision.children) == 0
            is_not_leaf = 0 if is_leaf else 1
            estimated_children = len([c for c in decision.children if c.id.startswith(str(node_id))])
            
            return (is_not_leaf, estimated_children, str(node_id))
        
        # Sort candidates by score (best first)
        sorted_candidates = sorted(candidates, key=score_candidate)
        
        # Convert to RichCandidate with metadata
        rich_candidates = []
        for i, (node_id, action) in enumerate(sorted_candidates):
            score_tuple = score_candidate((node_id, action))
            metadata = {
                "selection_method": "deterministic_heuristic",
                "rank": i + 1,
                "score_tuple": score_tuple,
                "reasoning": f"Rank {i+1}: deterministic heuristic scoring"
            }
            
            rich_candidates.append(RichCandidate(
                target=node_id,
                action=action,
                metadata=metadata
            ))
        
        return rich_candidates


class RandomAgent(CandidateAgent):
    """Agent that randomly selects from candidates (for testing)."""
    
    def __init__(self, seed: int | None = None):
        import random
        self.random = random.Random(seed)
    
    def select(
        self,
        decision: DecisionContext,
        candidates: list[tuple[NodeId, Action]]
    ) -> list[RichCandidate]:
        """Randomly shuffle candidates."""
        
        shuffled = candidates.copy()
        self.random.shuffle(shuffled)
        
        rich_candidates = []
        for i, (node_id, action) in enumerate(shuffled):
            metadata = {
                "selection_method": "random",
                "rank": i + 1,
                "reasoning": f"Random selection, rank {i+1}"
            }
            
            rich_candidates.append(RichCandidate(
                target=node_id,
                action=action, 
                metadata=metadata
            ))
        
        return rich_candidates


class SimpleScorer(CandidateScorer):
    """Simple scoring agent using the same heuristics as DeterministicAgent."""
    
    def score(
        self,
        decision: DecisionContext,
        candidates: list[tuple[NodeId, Action]]
    ) -> dict[tuple[NodeId, Action], float]:
        """Score candidates using deterministic heuristics."""
        scores = {}
        
        for candidate in candidates:
            node_id, action = candidate
            
            # Find the target node in children
            target_node = None
            for child in decision.children:
                if child.id == node_id:
                    target_node = child
                    break
            
            if target_node is None:
                scores[candidate] = 0.1  # Low score for unknown targets
                continue
            
            # Higher scores for leaf nodes, lower child counts
            is_leaf = len([c for c in decision.children if str(c.id).startswith(str(node_id))]) == 0
            estimated_children = len([c for c in decision.children if str(c.id).startswith(str(node_id))])
            
            base_score = 1.0 if is_leaf else 0.5
            child_penalty = estimated_children * 0.1
            final_score = max(0.1, base_score - child_penalty)
            
            scores[candidate] = final_score
        
        return scores