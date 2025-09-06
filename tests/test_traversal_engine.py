"""
Test Suite for TraversalEngine Interface

Tests the TraversalEngine interface capabilities:
- State management and serialization
- Domain-specific rules and guards
- Context ingestion from tree nodes
- Candidate action generation
- Move validation and application
- Parallel branch seeding
"""

import pytest
import json

from core import NodeId, Action, QueryState, RunContext, TreeIndex, TraversalEngine
from core.domains.medical.trees.icd_tree import create_navigator
from core.domains.medical.traversals.icd_traversal_engine import create_icd_traversal_engine


@pytest.fixture
def tree():
    """Create tree instance for tests."""
    return create_navigator()


@pytest.fixture
def traversal():
    """Create traversal engine instance for tests."""
    return create_icd_traversal_engine()


class TestStateManagement:
    """Test state serialization and management."""
    
    def test_state_serialization(self, traversal: TraversalEngine):
        """Test state serialization and deserialization."""
        # Create initial state
        initial_state = QueryState(
            current=NodeId("E10"),
            finalized=False,
            ctx=RunContext(data={"test_key": "test_value"})
        )
        
        # Test serialization
        serialized = traversal.dump_state(initial_state)
        assert isinstance(serialized, dict)
        assert "current" in serialized
        assert serialized["current"] == "E10"
        assert "finalized" in serialized
        assert serialized["finalized"] == False
        
        # Test deserialization
        deserialized = traversal.load_state(serialized)
        assert isinstance(deserialized, QueryState)
        assert deserialized.current == initial_state.current
        assert deserialized.finalized == initial_state.finalized
        assert deserialized.ctx.data == initial_state.ctx.data
    
    def test_context_serialization(self, traversal: TraversalEngine):
        """Test context serialization and deserialization."""
        # Create test context
        context = RunContext(data={
            "pending_codes": ["E10.1", "E10.2"],
            "excluded_families": ["E11"],
            "step_count": 5
        })
        
        # Test serialization
        serialized = traversal.dump_context(context)
        assert isinstance(serialized, dict)
        
        # Test deserialization  
        deserialized = traversal.load_context(serialized)
        assert isinstance(deserialized, RunContext)
        assert deserialized.data == context.data


class TestContextIngestion:
    """Test context ingestion from tree nodes."""
    
    def test_context_ingestion(self, tree: TreeIndex, traversal: TraversalEngine):
        """Test that context is updated when entering nodes."""
        # Start with empty context
        empty_ctx = RunContext(data={})
        
        # Ingest from a node
        node_id = NodeId("E10")
        updated_ctx = traversal.ingest(tree, node_id, empty_ctx)
        
        assert isinstance(updated_ctx, RunContext)
        assert isinstance(updated_ctx.data, dict)
    
    def test_context_accumulation(self, tree: TreeIndex, traversal: TraversalEngine):
        """Test that context accumulates properly across nodes."""
        # Start with some initial context
        ctx = RunContext(data={"initial": "value"})
        
        # Ingest from multiple nodes
        nodes = [NodeId("4"), NodeId("E08-E13"), NodeId("E10")]
        
        for node_id in nodes:
            ctx = traversal.ingest(tree, node_id, ctx)
            assert isinstance(ctx, RunContext)
            assert "initial" in ctx.data  # Should preserve initial data


class TestCandidateGeneration:
    """Test candidate action generation."""
    
    def test_basic_candidate_generation(self, tree: TreeIndex, traversal: TraversalEngine):
        """Test basic candidate generation."""
        # Create state at a node with children
        state = QueryState(
            current=NodeId("E10"),
            finalized=False,
            ctx=RunContext()
        )
        
        # Update context
        state.ctx = traversal.ingest(tree, state.current, state.ctx)
        
        # Get candidates
        candidates = traversal.candidate_actions(tree, state)
        
        assert isinstance(candidates, list)
        for node_id, action in candidates:
            assert isinstance(node_id, NodeId)
            assert isinstance(action, Action)
    
    def test_leaf_node_candidates(self, tree: TreeIndex, traversal: TraversalEngine):
        """Test candidate generation for leaf nodes."""
        # Find a leaf node
        leaf_candidates = []
        test_nodes = ["E10.21", "E10.22", "E10.29"]
        
        for node_code in test_nodes:
            if tree.is_leaf(NodeId(node_code)):
                leaf_candidates.append(NodeId(node_code))
        
        if leaf_candidates:
            state = QueryState(
                current=leaf_candidates[0],
                finalized=False,
                ctx=RunContext()
            )
            
            # Update context
            state.ctx = traversal.ingest(tree, state.current, state.ctx)
            
            # Get candidates
            candidates = traversal.candidate_actions(tree, state)
            
            assert isinstance(candidates, list)
            # Leaf nodes may have no candidates or only report/exit actions


class TestGuardsAndValidation:
    """Test move validation and guards."""
    
    def test_valid_moves(self, tree: TreeIndex, traversal: TraversalEngine):
        """Test validation of valid moves."""
        # Create state
        state = QueryState(
            current=NodeId("E10"),
            finalized=False,
            ctx=RunContext()
        )
        
        # Update context
        state.ctx = traversal.ingest(tree, state.current, state.ctx)
        
        # Get candidates and test first valid move
        candidates = traversal.candidate_actions(tree, state)
        
        if candidates:
            move = candidates[0]
            guard_result = traversal.guard_next(tree, state, move)
            
            # Should have an outcome
            assert hasattr(guard_result, 'outcome')
            assert hasattr(guard_result, 'message')
    
    def test_move_application(self, tree: TreeIndex, traversal: TraversalEngine):
        """Test move application."""
        # Create state
        state = QueryState(
            current=NodeId("E10"),
            finalized=False,
            ctx=RunContext()
        )
        
        # Update context
        state.ctx = traversal.ingest(tree, state.current, state.ctx)
        
        # Get candidates and try to apply first move
        candidates = traversal.candidate_actions(tree, state)
        
        if candidates:
            move = candidates[0]
            result = traversal.apply(tree, state, move)
            
            # Result should be either a new QueryState or a GuardResult
            assert (hasattr(result, 'current') or hasattr(result, 'outcome'))


class TestFinalization:
    """Test finalization capabilities."""
    
    def test_finalization_check(self, tree: TreeIndex, traversal: TraversalEngine):
        """Test finalization capability check."""
        # Test different nodes
        test_nodes = ["E10", "E10.21"]
        
        for node_code in test_nodes:
            state = QueryState(
                current=NodeId(node_code),
                finalized=False,
                ctx=RunContext()
            )
            
            # Update context
            state.ctx = traversal.ingest(tree, state.current, state.ctx)
            
            can_finalize = traversal.can_attempt_finalize(tree, state)
            assert isinstance(can_finalize, bool)


class TestParallelBranching:
    """Test parallel branch seeding."""
    
    def test_parallel_seeds(self, tree: TreeIndex, traversal: TraversalEngine):
        """Test parallel branch seed generation."""
        # Create state
        state = QueryState(
            current=NodeId("E10"),
            finalized=False,
            ctx=RunContext()
        )
        
        # Update context
        state.ctx = traversal.ingest(tree, state.current, state.ctx)
        
        # Get parallel seeds
        seeds = traversal.parallel_seeds(tree, state)
        
        assert isinstance(seeds, list)
        for seed in seeds:
            assert isinstance(seed, NodeId)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_node_handling(self, tree: TreeIndex, traversal: TraversalEngine):
        """Test handling of invalid nodes."""
        # Create state with nonexistent node
        state = QueryState(
            current=NodeId("INVALID"),
            finalized=False,
            ctx=RunContext()
        )
        
        # This should handle gracefully (may raise exception or return empty results)
        try:
            candidates = traversal.candidate_actions(tree, state)
            assert isinstance(candidates, list)
        except Exception:
            pass  # Exception is acceptable for invalid nodes
    
    def test_empty_context_handling(self, tree: TreeIndex, traversal: TraversalEngine):
        """Test handling of empty contexts."""
        empty_state = QueryState(
            current=NodeId("E10"),
            finalized=False,
            ctx=RunContext(data={})
        )
        
        # Should handle empty context gracefully
        candidates = traversal.candidate_actions(tree, empty_state)
        assert isinstance(candidates, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])