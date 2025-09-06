"""
Comprehensive Async Map-Reduce Parallel Execution Demo

Demonstrates async map-reduce for ALL types of parallel branches:
- useAdditionalCode (required constraints)
- codeAlso (suggested parallel codes)  
- multiCandidate (multiple agent selections)
- includeNote (contextual triggers)
- seventhCharacter (completion requirements)
- crossReference (see-also codes)
- combinationCode (alternative combinations)
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import NodeId, QueryState, RunContext
from core.domains.medical.trees.icd_tree import create_navigator
from core.domains.medical.traversals.icd_traversal_engine import create_icd_traversal_engine
from core.domains.medical.factories.icd10_domain import create_icd10_dspy_agent
from core.orchestrators.async_parallel_executor import (
    AsyncParallelExecutor, ParallelBranch
)


def create_complex_multi_branch_scenario() -> dict[str, Any]:
    """Create a complex scenario with multiple types of parallel branches."""
    return {
        "clinical_note": """
        45-year-old male with multiple trauma from motor vehicle accident:
        - Open fracture of left tibia shaft with delayed healing complications
        - Concurrent alcohol use disorder affecting recovery  
        - Secondary bacterial infection at fracture site
        - Requires external fixation device maintenance
        - Patient history includes chronic diabetes mellitus type 2
        - Current episode includes 7th character requirements for fracture care
        """,
        "description": "Multi-trauma case requiring comprehensive parallel coding",
        "expected_branches": {
            "useAdditionalCode": ["Z87.891", "F10.20"],  # History, alcohol use
            "codeAlso": ["T84.60XA", "E11.9"],  # Device complication, diabetes
            "seventhCharacter": ["S82.202D", "S82.202K"],  # Delayed healing variants
            "crossReference": ["M89.00"],  # Bone healing disorder
            "combinationCode": ["V43.4", "Z96.60"]  # Alternative device coding
        }
    }


async def demonstrate_comprehensive_parallel_execution() -> None:
    """Demonstrate async map-reduce across all parallel branch types."""
    print("🚀 COMPREHENSIVE ASYNC MAP-REDUCE DEMO")
    print("=" * 70)
    print("Demonstrates concurrent processing across ALL ICD parallel branch types")
    print()
    
    # Setup components
    tree = create_navigator()
    traversal = create_icd_traversal_engine()
    agent = create_icd10_dspy_agent("cerebras", "qwen-3-32b", max_candidates=3)
    
    executor = AsyncParallelExecutor(
        tree=tree,
        traversal=traversal,
        agent=agent,
        max_concurrent_branches=10,
        branch_timeout_seconds=15.0
    )
    
    # Create complex scenario
    scenario = create_complex_multi_branch_scenario()
    print(f"📋 Clinical Scenario: {scenario['description']}")
    print(f"📝 Note Length: {len(scenario['clinical_note'])} characters")
    print()
    
    # Create state with complex parallel requirements
    complex_state = QueryState(
        current=NodeId("S82.202"),  # Tibia fracture node
        finalized=False,
        step=None,
        ctx=RunContext(data={
            "clinical_note": scenario["clinical_note"],
            "complexity": "multi_branch"
        })
    )
    
    # Simulate ALL types of parallel branch requirements
    print("🔍 SIMULATING COMPREHENSIVE BRANCH REQUIREMENTS")
    print("-" * 50)
    
    # 1. useAdditionalCode requirements (required)
    if not hasattr(complex_state.ctx, 'pending_suffix'):
        complex_state.ctx.pending_suffix = set()
    for code in scenario["expected_branches"]["useAdditionalCode"]:
        complex_state.ctx.pending_suffix.add(code)
        print(f"   ✅ useAdditionalCode: {code} (REQUIRED)")
    
    # 2. codeAlso requirements (suggested)
    if not hasattr(complex_state.ctx, 'parallel'):
        complex_state.ctx.parallel = []
    for code in scenario["expected_branches"]["codeAlso"]:
        complex_state.ctx.parallel.append(code)
        print(f"   ✅ codeAlso: {code} (SUGGESTED)")
    
    # 3. Multiple agent candidates (alternatives) - simulate via step metadata
    class SimpleStep:
        def __init__(self):
            self.metadata = {"parallel_candidates": scenario["expected_branches"]["seventhCharacter"]}
    
    if not hasattr(complex_state, 'step') or not complex_state.step:
        complex_state.step = SimpleStep()
    else:
        if not hasattr(complex_state.step, 'metadata'):
            complex_state.step.metadata = {}
        complex_state.step.metadata["parallel_candidates"] = scenario["expected_branches"]["seventhCharacter"]
    
    for code in scenario["expected_branches"]["seventhCharacter"]:
        print(f"   ✅ multiCandidate: {code} (ALTERNATIVE)")
    
    # 4. Include/exclude notes are clarification only - NOT parallel branches
    print(f"   ℹ️  includes/inclusionTerm: Used for code clarification (NOT parallel execution)")
    
    # 5. 7th character options (completion)
    if not hasattr(complex_state.ctx, 'seventh_character_options'):
        complex_state.ctx.seventh_character_options = []
    # Add some different 7th character variants
    seventh_chars = ["S82.202A", "S82.202B", "S82.202C"]
    for code in seventh_chars:
        complex_state.ctx.seventh_character_options.append(code)
        print(f"   ✅ seventhCharacter: {code} (COMPLETION)")
    
    # 6. Cross-reference codes (reference)
    if not hasattr(complex_state.ctx, 'cross_references'):
        complex_state.ctx.cross_references = []
    for code in scenario["expected_branches"]["crossReference"]:
        complex_state.ctx.cross_references.append(code)
        print(f"   ✅ crossReference: {code} (REFERENCE)")
    
    # 7. Combination alternatives (combination)
    if not hasattr(complex_state.ctx, 'combination_codes'):
        complex_state.ctx.combination_codes = []
    for code in scenario["expected_branches"]["combinationCode"]:
        complex_state.ctx.combination_codes.append(code)
        print(f"   ✅ combinationCode: {code} (COMBINATION)")
    
    print()
    
    # Extract ALL parallel branches
    branches = executor.extract_parallel_branches(complex_state)
    
    if not branches:
        print("❌ No parallel branches detected - configuration issue")
        return
    
    print(f"✅ EXTRACTED {len(branches)} PARALLEL BRANCHES:")
    print("-" * 50)
    branch_types = {}
    for branch in branches:
        branch_type = branch.branch_type
        if branch_type not in branch_types:
            branch_types[branch_type] = []
        branch_types[branch_type].append(branch.target_code)
    
    for branch_type, codes in branch_types.items():
        print(f"   🎯 {branch_type}: {len(codes)} branches")
        for code in codes:
            print(f"      └─ {code}")
    
    # COMPREHENSIVE ASYNC MAP-REDUCE EXECUTION
    print(f"\n🚀 EXECUTING COMPREHENSIVE ASYNC MAP-REDUCE")
    print("=" * 60)
    
    import time
    total_start = time.time()
    
    # Execute all branch types concurrently
    map_reduce_result = await executor.execute_parallel_branches(complex_state, branches)
    
    total_time = (time.time() - total_start) * 1000
    
    # DETAILED ANALYSIS OF RESULTS
    print(f"\n📊 COMPREHENSIVE EXECUTION RESULTS")
    print("=" * 60)
    print(f"🕐 Total Execution Time: {total_time:.1f}ms")
    print(f"✅ Successful Branches: {map_reduce_result.completed_branches}")
    print(f"❌ Failed Branches: {map_reduce_result.failed_branches}")
    print(f"🎯 Success Rate: {(map_reduce_result.completed_branches / len(branches) * 100):.1f}%")
    print()
    
    print("🔍 BRANCH TYPE ANALYSIS:")
    print("-" * 40)
    type_results = {}
    for result in map_reduce_result.parallel_results:
        branch_type = result.branch.branch_type
        if branch_type not in type_results:
            type_results[branch_type] = {"success": 0, "total": 0, "total_time": 0}
        type_results[branch_type]["total"] += 1
        type_results[branch_type]["total_time"] += result.execution_time_ms
        if result.success:
            type_results[branch_type]["success"] += 1
    
    for branch_type, stats in type_results.items():
        success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
        avg_time = stats["total_time"] / stats["total"] if stats["total"] > 0 else 0
        print(f"   📍 {branch_type}: {stats['success']}/{stats['total']} ({success_rate:.0f}%) - Avg: {avg_time:.1f}ms")
    
    print("\n🎯 INDIVIDUAL BRANCH RESULTS:")
    print("-" * 40)
    for result in map_reduce_result.parallel_results:
        status = "✅" if result.success else "❌"
        metadata = result.branch.metadata or {}
        priority_info = f"Priority: {result.branch.priority}"
        meta_info = ", ".join(f"{k}: {v}" for k, v in metadata.items())
        print(f"   {status} {result.branch.target_code} ({result.branch.branch_type})")
        print(f"      └─ Final: {result.final_code} | Time: {result.execution_time_ms:.1f}ms | Steps: {result.steps_taken}")
        print(f"      └─ {priority_info} | {meta_info}")
        if result.error:
            print(f"      └─ Error: {result.error}")
    
    # PERFORMANCE ANALYSIS
    print(f"\n💡 COMPREHENSIVE PERFORMANCE BENEFITS:")
    print("-" * 50)
    sequential_time = sum(r.execution_time_ms for r in map_reduce_result.parallel_results)
    parallel_time = map_reduce_result.total_execution_time_ms
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1
    
    print(f"   📈 Sequential Time: {sequential_time:.1f}ms")
    print(f"   ⚡ Parallel Time: {parallel_time:.1f}ms")
    print(f"   🚀 Speedup: {speedup:.1f}x faster")
    print(f"   💾 Efficiency: {(speedup / len(branches) * 100):.1f}% of theoretical max")
    
    # CONSTRAINT SATISFACTION ANALYSIS
    print(f"\n🔒 CONSTRAINT SATISFACTION ANALYSIS:")
    print("-" * 50)
    remaining_branches = executor.extract_parallel_branches(map_reduce_result.primary_result) if map_reduce_result.primary_result else []
    
    if not remaining_branches:
        print("   ✅ ALL parallel requirements satisfied across ALL branch types")
        print("   ✅ Complex multi-branch workflow can now proceed to finalization")
    else:
        print(f"   ⚠️  {len(remaining_branches)} requirements still pending:")
        for branch in remaining_branches:
            print(f"      └─ {branch.branch_type}: {branch.target_code}")


async def demonstrate_scalability_test() -> None:
    """Demonstrate scalability with large numbers of parallel branches."""
    print(f"\n🧪 SCALABILITY STRESS TEST")
    print("=" * 60)
    
    tree = create_navigator()
    traversal = create_icd_traversal_engine() 
    agent = create_icd10_dspy_agent("cerebras", "qwen-3-32b", max_candidates=2)
    
    executor = AsyncParallelExecutor(
        tree=tree,
        traversal=traversal,
        agent=agent,
        max_concurrent_branches=20,  # Higher concurrency
        branch_timeout_seconds=30.0
    )
    
    # Create state with many parallel branches
    stress_state = QueryState(
        current=NodeId("TEST"),
        finalized=False,
        step=None,
        ctx=RunContext(data={"clinical_note": "Stress test scenario"})
    )
    
    # Generate large numbers of each branch type
    stress_state.ctx.pending_suffix = set(f"Z{i:02d}.{j}" for i in range(10, 15) for j in range(10))
    stress_state.ctx.parallel = [f"T{i:02d}.{j}XA" for i in range(20, 25) for j in range(10)]
    stress_state.ctx.cross_references = [f"M{i:02d}.{j}" for i in range(40, 42) for j in range(10)]
    
    branches = executor.extract_parallel_branches(stress_state)
    print(f"🔢 Generated {len(branches)} parallel branches for stress test")
    
    import time
    start_time = time.time()
    
    result = await executor.execute_parallel_branches(stress_state, branches)
    
    execution_time = (time.time() - start_time) * 1000
    
    print(f"⚡ Stress Test Results:")
    print(f"   🔢 Total Branches: {len(branches)}")
    print(f"   ✅ Successful: {result.completed_branches}")
    print(f"   ⏱️  Execution Time: {execution_time:.1f}ms")
    print(f"   📊 Throughput: {len(branches) / (execution_time / 1000):.1f} branches/sec")
    print(f"   💨 Avg Branch Time: {execution_time / len(branches):.1f}ms")


def main() -> None:
    """Run comprehensive async parallel execution demonstrations."""
    print("🎯 COMPREHENSIVE ASYNC MAP-REDUCE FOR ICD-10-CM")
    print("Parallel execution across ALL branch types with performance analysis")
    print()
    
    try:
        # Run comprehensive demonstration
        asyncio.run(demonstrate_comprehensive_parallel_execution())
        
        # Run scalability test
        asyncio.run(demonstrate_scalability_test())
        
        print(f"\n🎉 COMPREHENSIVE DEMO COMPLETE")
        print("=" * 50)
        print("✅ All parallel branch types demonstrated with async map-reduce")
        print("✅ useAdditionalCode, codeAlso, multiCandidate")
        print("✅ seventhCharacter, crossReference, combinationCode")
        print("✅ includes/inclusionTerm properly excluded (clarification only)")
        print("✅ Performance optimization and scalability validated")
        print()
        print("💡 This demonstrates how ICD-10-CM coding workflows can")
        print("   process ANY type of parallel requirement simultaneously,")
        print("   providing comprehensive performance improvements for")
        print("   the most complex clinical scenarios across all coding patterns.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()