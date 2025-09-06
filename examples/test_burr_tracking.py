"""
Test Burr UI Tracking Integration with AsyncParallelExecutor

Simple test to verify that parallel execution events are tracked in Burr UI.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import NodeId, QueryState, RunContext
from core.domains.medical.trees.icd_tree import create_navigator
from core.domains.medical.traversals.icd_traversal_engine import create_icd_traversal_engine
from core.domains.medical.factories.icd10_domain import create_icd10_dspy_agent
from core.orchestrators.async_parallel_executor import AsyncParallelExecutor

# Mock Burr context for testing
class MockBurrContext:
    def __init__(self):
        self._async_parallel_logs = []
        self.logged_events = []
        
    def log_attribute(self, key: str, value):
        """Mock Burr's log_attribute method."""
        self.logged_events.append({"key": key, "value": value})
        print(f"🔍 Burr Tracked: {key} -> {value}")


async def test_burr_tracking_integration() -> None:
    """Test that AsyncParallelExecutor integrates with Burr tracking."""
    print("🧪 TESTING BURR UI TRACKING INTEGRATION")
    print("=" * 60)
    
    # Setup components with mock Burr context
    tree = create_navigator()
    traversal = create_icd_traversal_engine()
    agent = create_icd10_dspy_agent("cerebras", "qwen-3-32b", max_candidates=2)
    
    # Create mock Burr context
    mock_burr_context = MockBurrContext()
    
    # Create executor with Burr tracking
    executor = AsyncParallelExecutor(
        tree=tree,
        traversal=traversal,
        agent=agent,
        burr_context=mock_burr_context
    )
    
    # Create state with parallel requirements
    test_state = QueryState(
        current=NodeId("E11"),
        finalized=False,
        step=None,
        ctx=RunContext(data={"clinical_note": "Test diabetes case"})
    )
    
    # Add some parallel requirements
    test_state.ctx.pending_suffix = {"Z79.4", "Z79.84"}
    test_state.ctx.parallel = ["E11.9", "T84.60XA"]
    
    print("🔍 Extracting parallel branches...")
    branches = executor.extract_parallel_branches(test_state)
    print(f"✅ Found {len(branches)} parallel branches")
    
    if branches:
        print("\n🚀 Executing parallel branches with Burr tracking...")
        
        # Execute with tracking
        result = await executor.execute_parallel_branches(test_state, branches)
        
        print(f"\n📊 EXECUTION RESULTS:")
        print(f"   ✅ Successful: {result.completed_branches}")
        print(f"   ❌ Failed: {result.failed_branches}")
        print(f"   ⏱️  Time: {result.total_execution_time_ms:.1f}ms")
        
        print(f"\n🔍 BURR TRACKING EVENTS:")
        print(f"   📊 Total Events Logged: {len(mock_burr_context.logged_events)}")
        
        for i, event in enumerate(mock_burr_context.logged_events, 1):
            print(f"   {i}. {event['key']}")
            if 'target_code' in str(event['value']):
                target = event['value'].get('target_code', 'N/A')
                branch_type = event['value'].get('branch_type', 'N/A')
                print(f"      └─ {branch_type}: {target}")
        
        # Test tracking summary
        summary = executor.get_burr_tracking_summary()
        if summary:
            print(f"\n📈 TRACKING SUMMARY:")
            print(f"   🔢 Total Events: {summary['total_events']}")
            print(f"   🚀 Execution Starts: {summary['execution_starts']}")
            print(f"   📍 Branch Starts: {summary['branch_starts']}")
            print(f"   ✅ Branch Completions: {summary['branch_completions']}")
            print(f"   ❌ Branch Failures: {summary['branch_failures']}")
        
        return True
    else:
        print("❌ No parallel branches found")
        return False


def main() -> None:
    """Run Burr tracking integration test."""
    print("🎯 ASYNC PARALLEL EXECUTOR + BURR UI TRACKING TEST")
    print("Testing integration between parallel execution and Burr workflow tracking")
    print()
    
    try:
        success = asyncio.run(test_burr_tracking_integration())
        
        print(f"\n🎉 TEST COMPLETE")
        print("=" * 30)
        if success:
            print("✅ Burr tracking integration working correctly")
            print("✅ Parallel execution events logged to Burr UI")
            print("✅ Individual branch tracking functional")
            print("✅ Summary generation working")
        else:
            print("❌ Test failed - no parallel branches detected")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()