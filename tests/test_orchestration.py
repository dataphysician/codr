"""
Test Suite for Orchestrator Implementations

Tests orchestrator functionality:
- BaseOrchestrator workflow execution
- Candidate enumeration and decision making
- Move application and state management
- Error handling and edge cases
- Complete workflow execution
"""

import pytest
import time

from core import (
    NodeId, Action, QueryState, RunContext, DecisionContext,
    TreeIndex, TraversalEngine, CandidateAgent, RichCandidate
)
from core.domains.medical.trees.icd_tree import create_navigator
from core.domains.medical.traversals.icd_traversal_engine import create_icd_traversal_engine
from core.orchestrators import BaseOrchestrator
from core.dag_agents.base_agents import DeterministicAgent


@pytest.fixture
def tree():
    """Create tree instance for tests."""
    return create_navigator()


@pytest.fixture
def traversal():
    """Create traversal engine instance for tests."""
    return create_icd_traversal_engine()


@pytest.fixture
def agent():
    """Create agent instance for tests."""
    return DeterministicAgent()


@pytest.fixture
def orchestrator(tree, traversal, agent):
    """Create orchestrator instance for tests."""
    return BaseOrchestrator(tree, traversal, agent)


class TestOrchestratorBasics:
    """Test basic orchestrator functionality."""
    
    def test_orchestrator_creation(self, tree, traversal, agent):
        """Test that orchestrator can be created."""
        orchestrator = BaseOrchestrator(tree, traversal, agent)
        
        assert orchestrator.tree is tree
        assert orchestrator.traversal is traversal
        assert orchestrator.agent is agent
        assert isinstance(orchestrator.history, list)
        assert len(orchestrator.history) == 0
    
    def test_candidate_enumeration(self, orchestrator):
        """Test candidate enumeration."""
        state = QueryState(
            current=NodeId("E10"),
            finalized=False,
            ctx=RunContext()
        )
        
        # Update context
        state.ctx = orchestrator.traversal.ingest(orchestrator.tree, state.current, state.ctx)
        
        candidates = orchestrator.enumerate_candidates(state)
        
        assert isinstance(candidates, list)
        for node_id, action in candidates:
            assert isinstance(node_id, NodeId)
            assert isinstance(action, Action)
    
    def test_decision_making(self, orchestrator):
        """Test agent decision making."""
        state = QueryState(
            current=NodeId("E10"),
            finalized=False,
            ctx=RunContext()
        )
        
        # Update context
        state.ctx = orchestrator.traversal.ingest(orchestrator.tree, state.current, state.ctx)
        
        candidates = orchestrator.enumerate_candidates(state)
        
        if candidates:
            decisions = orchestrator.decide(state, candidates)
            
            assert isinstance(decisions, list)
            assert len(decisions) <= len(candidates)
            
            for node_id, action in decisions:
                assert isinstance(node_id, NodeId)
                assert isinstance(action, Action)


class TestStepExecution:
    """Test step execution and state management."""
    
    def test_valid_step(self, orchestrator):
        """Test execution of valid steps."""
        state = QueryState(
            current=NodeId("E10"),
            finalized=False,
            ctx=RunContext()
        )
        
        # Update context
        state.ctx = orchestrator.traversal.ingest(orchestrator.tree, state.current, state.ctx)
        
        candidates = orchestrator.enumerate_candidates(state)
        
        if candidates:
            move = candidates[0]
            result = orchestrator.try_step(state, move)
            
            # Result should be either a new QueryState or error dict
            assert isinstance(result, (QueryState, dict))
            
            if isinstance(result, QueryState):
                assert isinstance(result.current, NodeId)
                assert isinstance(result.finalized, bool)
                # History should be updated
                assert len(orchestrator.history) > 0
            else:
                # Error case
                assert "error" in result
    
    def test_invalid_step(self, orchestrator):
        """Test handling of invalid steps."""
        state = QueryState(
            current=NodeId("E10"),
            finalized=False,
            ctx=RunContext()
        )
        
        # Try an invalid move
        invalid_move = (NodeId("NONEXISTENT"), Action.GOTO)
        result = orchestrator.try_step(state, invalid_move)
        
        # Should return error information
        assert isinstance(result, dict)
        assert "error" in result


class TestFinalization:
    """Test finalization capabilities."""
    
    def test_finalization_check(self, orchestrator):
        """Test finalization capability checking."""
        # Test with different node types
        test_nodes = ["E10", "E10.21"]
        
        for node_code in test_nodes:
            state = QueryState(
                current=NodeId(node_code),
                finalized=False,
                ctx=RunContext()
            )
            
            # Update context
            state.ctx = orchestrator.traversal.ingest(orchestrator.tree, state.current, state.ctx)
            
            result = orchestrator.attempt_finalize(state)
            
            # Should return either a new QueryState or error dict
            assert isinstance(result, (QueryState, dict))
    
    def test_parallel_seeds(self, orchestrator):
        """Test parallel seed generation."""
        state = QueryState(
            current=NodeId("E10"),
            finalized=False,
            ctx=RunContext()
        )
        
        # Update context
        state.ctx = orchestrator.traversal.ingest(orchestrator.tree, state.current, state.ctx)
        
        seeds = orchestrator.get_parallel_seeds(state)
        
        assert isinstance(seeds, list)
        for seed in seeds:
            assert isinstance(seed, NodeId)


class TestWorkflowExecution:
    """Test complete workflow execution."""
    
    def test_simple_workflow(self, orchestrator):
        """Test execution of a simple workflow."""
        result = orchestrator.execute_workflow(
            initial_node=NodeId("E10"),
            initial_context={"test": "workflow"},
            max_steps=3
        )
        
        assert isinstance(result, dict)
        assert "initial_node" in result
        assert "steps" in result
        assert "final_state" in result
        assert "success" in result
        
        # Check structure
        assert isinstance(result["steps"], list)
        assert isinstance(result["final_state"], dict)
        assert isinstance(result["success"], bool)
    
    def test_workflow_with_context(self, orchestrator):
        """Test workflow execution with initial context."""
        initial_context = {
            "clinical_note": "Patient has diabetes",
            "preference": "specific_codes"
        }
        
        result = orchestrator.execute_workflow(
            initial_node=NodeId("E10"),
            initial_context=initial_context,
            max_steps=5
        )
        
        assert isinstance(result, dict)
        assert result["initial_node"] == "E10"
        assert "steps" in result
    
    def test_workflow_termination(self, orchestrator):
        """Test that workflow terminates properly."""
        # Use small max_steps to ensure termination
        result = orchestrator.execute_workflow(
            initial_node=NodeId("E10"),
            max_steps=1
        )
        
        assert isinstance(result, dict)
        assert "success" in result
        assert len(result["steps"]) <= 1


class TestHistoryTracking:
    """Test history tracking functionality."""
    
    def test_history_recording(self, orchestrator):
        """Test that history is recorded properly."""
        initial_history_length = len(orchestrator.history)
        
        state = QueryState(
            current=NodeId("E10"),
            finalized=False,
            ctx=RunContext()
        )
        
        # Update context
        state.ctx = orchestrator.traversal.ingest(orchestrator.tree, state.current, state.ctx)
        
        candidates = orchestrator.enumerate_candidates(state)
        
        if candidates:
            move = candidates[0]
            result = orchestrator.try_step(state, move)
            
            if isinstance(result, QueryState):
                # History should have grown
                assert len(orchestrator.history) > initial_history_length
                
                # Latest entry should have expected structure
                latest = orchestrator.history[-1]
                assert "from" in latest
                assert "action" in latest
                assert "to" in latest
                assert "timestamp" in latest
                assert "finalized" in latest


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_candidates(self, orchestrator):
        """Test handling when no candidates are available."""
        # Create state that might have no candidates
        state = QueryState(
            current=NodeId("E10.999"),  # Might not exist
            finalized=False,
            ctx=RunContext()
        )
        
        candidates = orchestrator.enumerate_candidates(state)
        decisions = orchestrator.decide(state, candidates)
        
        # Should handle gracefully
        assert isinstance(candidates, list)
        assert isinstance(decisions, list)
    
    def test_invalid_initial_node(self, orchestrator):
        """Test workflow with invalid initial node."""
        result = orchestrator.execute_workflow(
            initial_node=NodeId("INVALID"),
            max_steps=1
        )
        
        # Should complete (possibly with error) but not crash
        assert isinstance(result, dict)
        assert "success" in result


class TestPerformance:
    """Test performance characteristics."""
    
    def test_step_performance(self, orchestrator):
        """Test that single steps execute quickly."""
        state = QueryState(
            current=NodeId("E10"),
            finalized=False,
            ctx=RunContext()
        )
        
        # Update context
        state.ctx = orchestrator.traversal.ingest(orchestrator.tree, state.current, state.ctx)
        
        start_time = time.time()
        
        candidates = orchestrator.enumerate_candidates(state)
        if candidates:
            decisions = orchestrator.decide(state, candidates)
            if decisions:
                orchestrator.try_step(state, decisions[0])
        
        step_time = time.time() - start_time
        
        # Single step should complete quickly
        assert step_time < 1.0
    
    def test_workflow_performance(self, orchestrator):
        """Test that short workflows execute reasonably fast."""
        start_time = time.time()
        
        result = orchestrator.execute_workflow(
            initial_node=NodeId("E10"),
            max_steps=3
        )
        
        workflow_time = time.time() - start_time
        
        # Short workflow should complete reasonably fast
        assert workflow_time < 5.0
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])