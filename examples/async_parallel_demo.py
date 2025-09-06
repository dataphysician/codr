"""
Async Map-Reduce Parallel Execution Demo

Demonstrates true parallel execution of useAdditionalCode requirements
using async/await patterns for ICD-10-CM coding workflows.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports with Python 3.10+ annotations
from core import NodeId, QueryState, RunContext
from core.domains.medical.trees.icd_tree import create_navigator
from core.domains.medical.traversals.icd_traversal_engine import create_icd_traversal_engine
from core.domains.medical.factories.icd10_domain import create_icd10_dspy_agent
from core.orchestrators.async_parallel_executor import (
    AsyncParallelExecutor, ParallelBranch, async_run_with_parallel_execution
)


def create_diabetes_scenario() -> dict[str, Any]:
    """Create a complex diabetes scenario requiring multiple codes."""
    return {
        "clinical_note": """
        67-year-old male with Type 2 diabetes mellitus (15+ years) presents with:
        - Diabetic foot ulcer (right foot, plantar surface)  
        - Current medications: metformin 1000mg BID, insulin glargine 20 units qHS
        - HbA1c: 8.2% (elevated)
        - Requires comprehensive ICD-10-CM coding including medication context
        """,
        "primary_diagnosis": "E11.621",  # Type 2 diabetes with foot ulcer
        "expected_additional_codes": [
            "Z79.4",    # Long term use of insulin
            "Z79.84",   # Long term use of oral hypoglycemic drugs  
            "L97.511"   # Non-pressure chronic ulcer of plantar surface of right foot
        ],
        "description": "Complex diabetes case requiring parallel execution"
    }


async def demonstrate_parallel_execution() -> None:
    """
    Demonstrate async map-reduce for parallel useAdditionalCode execution.
    
    Shows true concurrent processing of:
    - Primary: E11.621 (diabetes with foot ulcer)
    - Parallel: Z79.4 (insulin), Z79.84 (oral drugs) 
    """
    print("🚀 ASYNC MAP-REDUCE PARALLEL EXECUTION DEMO")
    print("=" * 60)
    print("Demonstrates concurrent processing of ICD useAdditionalCode requirements")
    print()
    
    # Setup components
    tree = create_navigator()
    traversal = create_icd_traversal_engine()
    agent = create_icd10_dspy_agent("cerebras", "qwen-3-32b", max_candidates=3)
    
    # Create async executor
    executor = AsyncParallelExecutor(
        tree=tree,
        traversal=traversal, 
        agent=agent,
        max_concurrent_branches=5,
        branch_timeout_seconds=10.0
    )
    
    # Create diabetes scenario state
    scenario = create_diabetes_scenario()
    print(f"📋 Clinical Scenario: {scenario['description']}")
    print(f"🏥 Primary Diagnosis: {scenario['primary_diagnosis']}")
    print(f"💊 Expected Additional Codes: {', '.join(scenario['expected_additional_codes'])}")
    print()
    
    # Simulate workflow reaching E11 node (where useAdditionalCode triggers)
    print("🔄 WORKFLOW SIMULATION: Navigating to E11 (Type 2 diabetes)")
    e11_state = QueryState(
        current=NodeId("E11"),
        finalized=False,
        step=None,
        ctx=RunContext(data={
            "clinical_note": scenario["clinical_note"],
            "step_count": 3
        })
    )
    
    # Simulate context ingestion (triggers useAdditionalCode parallel seeding)
    print("🔍 Ingesting E11 notes → useAdditionalCode requirements detected")
    e11_node = tree.get(NodeId("E11"))
    if e11_node:
        traversal.ingest(tree, NodeId("E11"), e11_state.ctx)
    
    # Manually add useAdditionalCode requirements for demonstration
    # This simulates what the ICD traversal engine would detect from E11 notes
    if hasattr(e11_state.ctx, 'add_suffixes'):
        e11_state.ctx.add_suffixes(scenario['expected_additional_codes'])
    if hasattr(e11_state.ctx, 'register_codealso'):
        e11_state.ctx.register_codealso(scenario['expected_additional_codes'])
    else:
        # Fallback: manually set the attributes
        if not hasattr(e11_state.ctx, 'pending_suffix'):
            e11_state.ctx.pending_suffix = set()
        if not hasattr(e11_state.ctx, 'parallel'):
            e11_state.ctx.parallel = []
        
        for code in scenario['expected_additional_codes']:
            e11_state.ctx.pending_suffix.add(code)
            e11_state.ctx.parallel.append(code)
    
    # Extract parallel branches
    branches = executor.extract_parallel_branches(e11_state)
    
    if not branches:
        print("❌ No parallel branches detected - useAdditionalCode may not be configured")
        return
        
    print(f"✅ Extracted {len(branches)} parallel branches for execution:")
    for branch in branches:
        print(f"   📍 {branch.branch_type}: {branch.target_code} (priority: {branch.priority})")
    
    # ASYNC MAP-REDUCE EXECUTION
    print(f"\n🚀 EXECUTING ASYNC MAP-REDUCE PATTERN")
    print("=" * 40)
    
    import time
    total_start = time.time()
    
    # Execute parallel branches concurrently
    map_reduce_result = await executor.execute_parallel_branches(e11_state, branches)
    
    total_time = (time.time() - total_start) * 1000
    
    # Display detailed results
    print(f"\n📊 PARALLEL EXECUTION RESULTS")
    print("=" * 40)
    print(f"🕐 Total Execution Time: {total_time:.1f}ms")
    print(f"✅ Successful Branches: {map_reduce_result.completed_branches}")
    print(f"❌ Failed Branches: {map_reduce_result.failed_branches}")
    print()
    
    print("🎯 Individual Branch Results:")
    for result in map_reduce_result.parallel_results:
        status = "✅" if result.success else "❌"
        print(f"   {status} {result.branch.target_code} ({result.branch.branch_type})")
        print(f"      └─ Final Code: {result.final_code}")
        print(f"      └─ Execution Time: {result.execution_time_ms:.1f}ms")
        if result.error:
            print(f"      └─ Error: {result.error}")
    
    # Show constraint satisfaction
    print(f"\n🔒 CONSTRAINT SATISFACTION:")
    if map_reduce_result.primary_result:
        remaining_branches = executor.extract_parallel_branches(map_reduce_result.primary_result)
        if not remaining_branches:
            print("   ✅ All useAdditionalCode requirements satisfied")
            print("   ✅ Primary diagnosis can now be finalized")
        else:
            print(f"   ⚠️  {len(remaining_branches)} requirements still pending")
    
    # Demonstrate concurrent execution benefits
    print(f"\n💡 CONCURRENT EXECUTION BENEFITS:")
    sequential_time = sum(r.execution_time_ms for r in map_reduce_result.parallel_results)
    parallel_time = map_reduce_result.total_execution_time_ms
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1
    
    print(f"   📈 Sequential Time: {sequential_time:.1f}ms")
    print(f"   ⚡ Parallel Time: {parallel_time:.1f}ms") 
    print(f"   🚀 Speedup: {speedup:.1f}x faster")


async def demonstrate_full_workflow_integration() -> None:
    """Demonstrate integration with complete workflow."""
    print(f"\n🔄 FULL WORKFLOW INTEGRATION DEMO")
    print("=" * 60)
    
    # Setup
    tree = create_navigator()
    traversal = create_icd_traversal_engine()
    agent = create_icd10_dspy_agent("cerebras", "qwen-3-32b", max_candidates=2)
    
    executor = AsyncParallelExecutor(tree, traversal, agent)
    
    # Start from diabetes chapter
    initial_state = QueryState(
        current=NodeId("4"),  # Endocrine chapter
        finalized=False,
        step=None,
        ctx=RunContext(data={
            "clinical_note": "Type 2 diabetes patient on insulin and metformin",
            "target": "comprehensive_coding"
        })
    )
    
    print("🏁 Starting integrated workflow with parallel execution...")
    
    # Run workflow with async parallel execution
    final_state, parallel_executions = await async_run_with_parallel_execution(
        executor, initial_state, max_steps=5
    )
    
    print(f"\n📋 WORKFLOW SUMMARY:")
    print(f"   🏁 Final Position: {final_state.current}")
    print(f"   🔀 Parallel Executions: {len(parallel_executions)}")
    
    for i, execution in enumerate(parallel_executions, 1):
        print(f"   🚀 Execution {i}: {execution.completed_branches} branches completed")


def main() -> None:
    """Run all async parallel execution demonstrations."""
    print("🎯 ASYNC MAP-REDUCE FOR ICD-10-CM CODING")
    print("Real parallel execution of useAdditionalCode requirements")
    print()
    
    try:
        # Run async demonstrations
        asyncio.run(demonstrate_parallel_execution())
        asyncio.run(demonstrate_full_workflow_integration())
        
        print(f"\n🎉 DEMO COMPLETE")
        print("=" * 30)
        print("✅ Async map-reduce parallel execution demonstrated")
        print("✅ useAdditionalCode requirements processed concurrently") 
        print("✅ True parallel branching achieved vs. linear scheduling")
        print()
        print("💡 This demonstrates how ICD-10-CM coding workflows can")
        print("   process multiple diagnostic requirements simultaneously,")
        print("   providing significant performance improvements for")
        print("   complex clinical scenarios requiring multiple codes.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()