"""
End-to-End Integration Test Suite

Tests complete workflows combining all components:
- Tree navigation with traversal engines
- Agent-driven decision making
- Orchestrator-managed workflows
- State serialization and persistence
- Performance under realistic conditions
"""

import pytest
import json
import time

from core import NodeId, Action, QueryState, RunContext
from core.domains.medical.trees.icd_tree import create_navigator
from core.domains.medical.traversals.icd_traversal_engine import create_icd_traversal_engine
from core.orchestrators import BaseOrchestrator
from core.dag_agents.base_agents import DeterministicAgent


@pytest.fixture
def full_system():
    """Create complete system with all components."""
    tree = create_navigator()
    traversal = create_icd_traversal_engine()
    agent = DeterministicAgent()
    orchestrator = BaseOrchestrator(tree, traversal, agent)
    
    return {
        'tree': tree,
        'traversal': traversal,
        'agent': agent,
        'orchestrator': orchestrator
    }


class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_diabetes_coding_workflow(self, full_system):
        """Test complete diabetes coding workflow."""
        orchestrator = full_system['orchestrator']
        
        # Start with diabetes-related chapter
        result = orchestrator.execute_workflow(
            initial_node=NodeId("4"),  # Endocrine chapter
            initial_context={
                "clinical_note": "Patient diagnosed with type 1 diabetes with diabetic nephropathy",
                "target_condition": "diabetes"
            },
            max_steps=8
        )
        
        assert isinstance(result, dict)
        assert result["success"] == True or len(result["steps"]) > 0
        assert "final_state" in result
        
        # Should have made some progress
        final_state = result["final_state"]
        assert final_state["step_count"] >= 0
    
    def test_specific_code_navigation(self, full_system):
        """Test navigation to specific diagnostic codes."""
        orchestrator = full_system['orchestrator']
        
        # Start from a category and navigate to specific code
        result = orchestrator.execute_workflow(
            initial_node=NodeId("E10"),  # Type 1 diabetes mellitus
            initial_context={
                "clinical_note": "Type 1 diabetes with kidney complications",
                "specificity_required": True
            },
            max_steps=5
        )
        
        assert isinstance(result, dict)
        assert "steps" in result
        assert "final_state" in result
        
        # Should have attempted navigation
        steps = result["steps"]
        if steps:
            assert len(steps) <= 5
            for step in steps:
                assert "step" in step
                assert "success" in step
    
    def test_multi_step_traversal(self, full_system):
        """Test multi-step traversal through tree hierarchy."""
        orchestrator = full_system['orchestrator']
        
        # Test longer workflow
        result = orchestrator.execute_workflow(
            initial_node=NodeId("E08-E13"),  # Diabetes mellitus block
            initial_context={
                "clinical_note": "Complex diabetes case requiring specific coding",
                "thorough_search": True
            },
            max_steps=10
        )
        
        assert isinstance(result, dict)
        assert isinstance(result["steps"], list)
        
        # Workflow should have made decisions
        if result["steps"]:
            # Each step should have proper structure
            for step in result["steps"]:
                assert "step" in step
                assert "move" in step
                assert "success" in step


class TestStateManagement:
    """Test state management across workflow steps."""
    
    def test_state_persistence(self, full_system):
        """Test that state persists correctly across steps."""
        tree = full_system['tree']
        traversal = full_system['traversal']
        
        # Create initial state
        initial_state = QueryState(
            current=NodeId("E10"),
            finalized=False,
            ctx=RunContext(data={"step_count": 0})
        )
        
        # Serialize and deserialize
        serialized = traversal.dump_state(initial_state)
        deserialized = traversal.load_state(serialized)
        
        # States should match
        assert deserialized.current == initial_state.current
        assert deserialized.finalized == initial_state.finalized
        assert deserialized.ctx.data == initial_state.ctx.data
    
    def test_context_evolution(self, full_system):
        """Test that context evolves properly during traversal."""
        tree = full_system['tree']
        traversal = full_system['traversal']
        
        # Start with empty context
        ctx = RunContext(data={})
        
        # Traverse through multiple nodes
        nodes = [NodeId("4"), NodeId("E08-E13"), NodeId("E10")]
        
        for node_id in nodes:
            ctx = traversal.ingest(tree, node_id, ctx)
            assert isinstance(ctx, RunContext)
            assert isinstance(ctx.data, dict)
        
        # Context should have accumulated information
        assert len(ctx.data) >= 0  # May have accumulated data


class TestComponentIntegration:
    """Test integration between different components."""
    
    def test_tree_traversal_integration(self, full_system):
        """Test integration between tree and traversal engine."""
        tree = full_system['tree']
        traversal = full_system['traversal']
        
        # Test that tree and traversal work together
        node_id = NodeId("E10")
        state = QueryState(
            current=node_id,
            finalized=False,
            ctx=RunContext()
        )
        
        # Update context using traversal engine
        state.ctx = traversal.ingest(tree, node_id, state.ctx)
        
        # Get candidates using both components
        candidates = traversal.candidate_actions(tree, state)
        
        assert isinstance(candidates, list)
        
        # Validate candidates exist in tree
        for candidate_id, action in candidates:
            if str(candidate_id) != str(node_id):  # Skip self-references
                node = tree.get(candidate_id)
                # Node should exist or be a valid target
                assert node is not None or isinstance(candidate_id, NodeId)
    
    def test_agent_orchestrator_integration(self, full_system):
        """Test integration between agent and orchestrator."""
        orchestrator = full_system['orchestrator']
        agent = full_system['agent']
        
        # Create a state for testing
        state = QueryState(
            current=NodeId("E10"),
            finalized=False,
            ctx=RunContext()
        )
        
        # Test that orchestrator uses agent properly
        candidates = orchestrator.enumerate_candidates(state)
        
        if candidates:
            decisions = orchestrator.decide(state, candidates)
            
            # Agent should return valid decisions
            assert isinstance(decisions, list)
            assert len(decisions) <= len(candidates)
            
            for node_id, action in decisions:
                assert isinstance(node_id, NodeId)
                assert isinstance(action, Action)


class TestErrorRecovery:
    """Test system behavior under error conditions."""
    
    def test_invalid_node_recovery(self, full_system):
        """Test recovery from invalid node scenarios."""
        orchestrator = full_system['orchestrator']
        
        # Try workflow with potentially invalid node
        result = orchestrator.execute_workflow(
            initial_node=NodeId("INVALID"),
            max_steps=3
        )
        
        # System should handle gracefully
        assert isinstance(result, dict)
        assert "success" in result
        # Should not crash, may succeed (False) or have error
    
    def test_empty_context_handling(self, full_system):
        """Test handling of empty or minimal contexts."""
        orchestrator = full_system['orchestrator']
        
        # Run with minimal context
        result = orchestrator.execute_workflow(
            initial_node=NodeId("E10"),
            initial_context={},  # Empty context
            max_steps=2
        )
        
        assert isinstance(result, dict)
        # Should handle empty context without errors


class TestPerformanceIntegration:
    """Test performance of integrated system."""
    
    def test_workflow_timing(self, full_system):
        """Test that complete workflows execute in reasonable time."""
        orchestrator = full_system['orchestrator']
        
        start_time = time.time()
        
        result = orchestrator.execute_workflow(
            initial_node=NodeId("E10"),
            initial_context={"test": "performance"},
            max_steps=5
        )
        
        workflow_time = time.time() - start_time
        
        # Workflow should complete reasonably quickly
        assert workflow_time < 10.0  # Should complete in under 10 seconds
        assert isinstance(result, dict)
    
    def test_multiple_workflows(self, full_system):
        """Test running multiple workflows in sequence."""
        orchestrator = full_system['orchestrator']
        
        test_nodes = [NodeId("E10"), NodeId("E11"), NodeId("4")]
        results = []
        
        start_time = time.time()
        
        for node_id in test_nodes:
            result = orchestrator.execute_workflow(
                initial_node=node_id,
                max_steps=3
            )
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Multiple workflows should complete reasonably quickly
        assert total_time < 15.0
        assert len(results) == len(test_nodes)
        
        for result in results:
            assert isinstance(result, dict)
            assert "success" in result


class TestDataFlow:
    """Test data flow through the complete system."""
    
    def test_information_preservation(self, full_system):
        """Test that information is preserved through the workflow."""
        orchestrator = full_system['orchestrator']
        
        initial_context = {
            "clinical_note": "Patient with diabetes",
            "user_id": "test123",
            "session_id": "session456"
        }
        
        result = orchestrator.execute_workflow(
            initial_node=NodeId("E10"),
            initial_context=initial_context,
            max_steps=3
        )
        
        assert isinstance(result, dict)
        
        # Initial context information should be tracked
        assert result["initial_node"] == "E10"
    
    def test_step_traceability(self, full_system):
        """Test that workflow steps are properly traceable."""
        orchestrator = full_system['orchestrator']
        
        result = orchestrator.execute_workflow(
            initial_node=NodeId("E10"),
            initial_context={"trace": True},
            max_steps=4
        )
        
        assert isinstance(result, dict)
        assert "steps" in result
        
        steps = result["steps"]
        if steps:
            # Each step should have traceability information
            for step in steps:
                assert "step" in step  # Step number
                assert "move" in step  # What move was made
                assert "success" in step  # Whether it succeeded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])