"""
Test Suite for Agent Implementations

Tests different agent implementations and selection strategies:
- DeterministicAgent with heuristic-based selection
- RandomAgent for testing and comparison
- Custom agent implementations
- Agent comparison and benchmarking
- DecisionContext usage patterns
"""

import pytest
import time
import random
from typing import List

from core import (
    NodeId, Action, DecisionContext, RichCandidate, 
    TreeIndex, CandidateAgent, CandidateScorer, NodeView
)
from core.domains.medical.trees.icd_tree import create_navigator
from core.dag_agents.base_agents import DeterministicAgent, RandomAgent, SimpleScorer


class TestAgent(CandidateAgent):
    """Test agent that prefers deeper nodes."""
    
    def select(
        self,
        decision: DecisionContext,
        candidates: list[tuple[NodeId, Action]]
    ) -> list[RichCandidate]:
        """Select candidates preferring deeper tree levels."""
        
        def get_depth_score(candidate: tuple[NodeId, Action]) -> int:
            node_id, action = candidate
            code_str = str(node_id)
            dot_count = code_str.count('.')
            return dot_count * 10 + len(code_str)
        
        # Sort by depth score (descending)
        sorted_candidates = sorted(candidates, key=get_depth_score, reverse=True)
        
        # Return top candidates as RichCandidates
        return [
            RichCandidate(
                target=node_id,
                action=action,
                metadata={"depth_score": get_depth_score((node_id, action))}
            )
            for node_id, action in sorted_candidates[:3]
        ]


@pytest.fixture
def tree():
    """Create tree instance for tests."""
    return create_navigator()


@pytest.fixture
def decision_context(tree):
    """Create a sample decision context for testing."""
    node_id = NodeId("E10")
    node = tree.get(node_id)
    children = [tree.get(cid) for cid in tree.children(node_id)]
    children = [c for c in children if c is not None]
    ancestors = [tree.get(aid) for aid in tree.ancestors(node_id)]
    ancestors = [a for a in ancestors if a is not None]
    
    return DecisionContext(
        node=node,
        ancestors=ancestors,
        children=children,
        path=tree.path_to_root(node_id),
        allowed_actions=[Action.GOTO, Action.REPORT],
        pending_constraints={},
        external_context={"test": "context"}
    )


class TestDeterministicAgent:
    """Test the deterministic agent implementation."""
    
    def test_agent_creation(self):
        """Test that deterministic agent can be created."""
        agent = DeterministicAgent()
        assert isinstance(agent, CandidateAgent)
    
    def test_agent_selection(self, decision_context):
        """Test that agent can select candidates."""
        agent = DeterministicAgent()
        candidates = [
            (NodeId("E10.1"), Action.GOTO),
            (NodeId("E10.2"), Action.GOTO),
            (NodeId("E10.3"), Action.GOTO)
        ]
        
        selections = agent.select(decision_context, candidates)
        
        assert isinstance(selections, list)
        assert len(selections) <= len(candidates)
        
        for selection in selections:
            assert isinstance(selection, RichCandidate)
            assert isinstance(selection.target, NodeId)
            assert isinstance(selection.action, Action)
    
    def test_empty_candidates(self, decision_context):
        """Test agent behavior with no candidates."""
        agent = DeterministicAgent()
        selections = agent.select(decision_context, [])
        
        assert isinstance(selections, list)
        assert len(selections) == 0


class TestRandomAgent:
    """Test the random agent implementation."""
    
    def test_agent_creation(self):
        """Test that random agent can be created."""
        agent = RandomAgent()
        assert isinstance(agent, CandidateAgent)
    
    def test_agent_randomness(self, decision_context):
        """Test that random agent produces varied results."""
        agent = RandomAgent()
        candidates = [
            (NodeId("E10.1"), Action.GOTO),
            (NodeId("E10.2"), Action.GOTO),
            (NodeId("E10.3"), Action.GOTO),
            (NodeId("E10.4"), Action.GOTO)
        ]
        
        # Run multiple times to check for randomness
        results = []
        for _ in range(10):
            selections = agent.select(decision_context, candidates)
            if selections:
                results.append(str(selections[0].target))
        
        # Should have some variation (not all the same)
        unique_results = set(results)
        assert len(unique_results) > 1 or len(results) == 0


class TestCustomAgent:
    """Test custom agent implementations."""
    
    def test_custom_agent(self, decision_context):
        """Test custom agent that prefers deeper nodes."""
        agent = TestAgent()
        candidates = [
            (NodeId("E10"), Action.GOTO),      # Less specific
            (NodeId("E10.1"), Action.GOTO),    # More specific
            (NodeId("E10.21"), Action.GOTO)    # Most specific
        ]
        
        selections = agent.select(decision_context, candidates)
        
        assert isinstance(selections, list)
        assert len(selections) > 0
        
        # Should prefer more specific codes
        first_selection = selections[0]
        assert isinstance(first_selection, RichCandidate)
        
        # Check that metadata is included
        if first_selection.metadata:
            assert "depth_score" in first_selection.metadata


class TestAgentPerformance:
    """Test agent performance characteristics."""
    
    def test_selection_performance(self, decision_context):
        """Test that agent selection is reasonably fast."""
        agent = DeterministicAgent()
        candidates = [
            (NodeId(f"E10.{i}"), Action.GOTO)
            for i in range(100)
        ]
        
        start_time = time.time()
        for _ in range(10):
            agent.select(decision_context, candidates)
        selection_time = time.time() - start_time
        
        # Should complete 10 selections in under 1 second
        assert selection_time < 1.0
    
    def test_large_candidate_handling(self, decision_context):
        """Test agent handling of large candidate lists."""
        agent = DeterministicAgent()
        candidates = [
            (NodeId(f"E10.{i:03d}"), Action.GOTO)
            for i in range(1000)
        ]
        
        selections = agent.select(decision_context, candidates)
        
        assert isinstance(selections, list)
        # Should handle large lists without error
        assert len(selections) <= len(candidates)


class TestSimpleScorer:
    """Test the simple scorer implementation."""
    
    def test_scorer_creation(self):
        """Test that simple scorer can be created."""
        scorer = SimpleScorer()
        assert isinstance(scorer, CandidateScorer)
    
    def test_scoring_functionality(self, decision_context):
        """Test basic scoring functionality."""
        scorer = SimpleScorer()
        candidates = [
            (NodeId("E10.1"), Action.GOTO),
            (NodeId("E10.2"), Action.GOTO),
            (NodeId("E10.3"), Action.GOTO)
        ]
        
        scores = scorer.score(decision_context, candidates)
        
        assert isinstance(scores, dict)
        assert len(scores) == len(candidates)
        
        for candidate, score in scores.items():
            assert candidate in candidates
            assert isinstance(score, (int, float))


class TestDecisionContext:
    """Test DecisionContext usage patterns."""
    
    def test_context_structure(self, decision_context):
        """Test that decision context has required structure."""
        assert hasattr(decision_context, 'node')
        assert hasattr(decision_context, 'ancestors')
        assert hasattr(decision_context, 'children')
        assert hasattr(decision_context, 'path')
        assert hasattr(decision_context, 'allowed_actions')
        assert hasattr(decision_context, 'pending_constraints')
        assert hasattr(decision_context, 'external_context')
        
        # Verify types
        assert isinstance(decision_context.node, NodeView)
        assert isinstance(decision_context.ancestors, list)
        assert isinstance(decision_context.children, list)
        assert isinstance(decision_context.path, list)
        assert isinstance(decision_context.allowed_actions, list)
        assert isinstance(decision_context.pending_constraints, dict)
        assert isinstance(decision_context.external_context, dict)
    
    def test_context_content(self, decision_context):
        """Test that decision context contains meaningful data."""
        # Node should have basic properties
        assert decision_context.node.id
        assert decision_context.node.name
        assert decision_context.node.element_type
        
        # Should have allowed actions
        assert len(decision_context.allowed_actions) > 0
        for action in decision_context.allowed_actions:
            assert isinstance(action, Action)


class TestAgentComparison:
    """Test comparing different agents."""
    
    def test_agent_consistency(self, decision_context):
        """Test that deterministic agent is consistent."""
        agent = DeterministicAgent()
        candidates = [
            (NodeId("E10.1"), Action.GOTO),
            (NodeId("E10.2"), Action.GOTO),
            (NodeId("E10.3"), Action.GOTO)
        ]
        
        # Multiple runs should produce same results
        selections1 = agent.select(decision_context, candidates)
        selections2 = agent.select(decision_context, candidates)
        
        # Convert to comparable format
        if selections1 and selections2:
            targets1 = [s.target for s in selections1]
            targets2 = [s.target for s in selections2]
            assert targets1 == targets2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])