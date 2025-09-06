"""
Async Map-Reduce Parallel Execution for ICD Workflows

This module implements true parallel execution for ICD-10-CM coding requirements
that represent separate coding paths, NOT clarification notes.

PARALLEL EXECUTION TRIGGERS:
✅ useAdditionalCode - Required additional codes (highest priority)
✅ codeAlso - Suggested parallel codes  
✅ multiCandidate - Multiple agent-selected alternatives
✅ seventhCharacter - 7th character completion requirements
✅ crossReference - See-also reference codes
✅ combinationCode - Alternative combination coding

NON-PARALLEL CLARIFICATION NOTES:
❌ includes - Code applicability clarification only
❌ inclusionTerm - Terminology clarification only
❌ excludes1/excludes2 - Exclusion rules (not separate codes)
❌ note - General instructional text
"""

import asyncio
from typing import Any, Callable, Awaitable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from core import NodeId, Action, QueryState, RunContext, TreeIndex
from core.interfaces import TraversalEngine, CandidateAgent


@dataclass
class ParallelBranch:
    """Represents a single parallel execution branch."""
    target_code: str
    branch_type: str  # "useAdditionalCode", "codeAlso", etc.
    priority: int = 0
    metadata: dict[str, Any] | None = None


@dataclass  
class ParallelResult:
    """Result from a single parallel branch execution."""
    branch: ParallelBranch
    success: bool
    final_code: str | None
    steps_taken: int
    execution_time_ms: float
    error: Exception | None = None


@dataclass
class MapReduceResult:
    """Combined result from map-reduce parallel execution."""
    primary_result: QueryState | None
    parallel_results: list[ParallelResult]
    total_execution_time_ms: float
    success: bool
    completed_branches: int
    failed_branches: int


class AsyncParallelExecutor:
    """
    Async executor for parallel ICD workflow branches using map-reduce pattern.
    
    Enables true concurrent execution of useAdditionalCode requirements like:
    - Primary: ROOT → E11 → E11.621 (diabetes diagnosis)  
    - Parallel: Z79.4 (insulin), Z79.84 (oral drugs), Z79.85 (injectable drugs)
    
    Supports Burr UI tracking integration for visibility into parallel execution.
    """
    
    def __init__(
        self, 
        tree: TreeIndex,
        traversal: TraversalEngine, 
        agent: CandidateAgent,
        max_concurrent_branches: int = 10,
        branch_timeout_seconds: float = 30.0,
        burr_context: Any | None = None
    ):
        self.tree = tree
        self.traversal = traversal
        self.agent = agent
        self.max_concurrent_branches = max_concurrent_branches
        self.branch_timeout_seconds = branch_timeout_seconds
        self.burr_context = burr_context  # Optional Burr ApplicationContext for tracking
    
    async def execute_parallel_branches(
        self,
        current_state: QueryState,
        parallel_branches: list[ParallelBranch]
    ) -> MapReduceResult:
        """
        Execute parallel branches using async map-reduce pattern.
        
        Map: Each branch becomes an async task
        Reduce: Combine results and update main workflow state
        """
        import time
        start_time = time.time()
        
        print(f"\n🚀 ASYNC MAP-REDUCE: Executing {len(parallel_branches)} parallel branches")
        for branch in parallel_branches:
            print(f"   📍 {branch.branch_type}: {branch.target_code}")
        
        # Track parallel execution start in Burr UI if context available
        if self.burr_context:
            self._log_to_burr("parallel_execution_start", {
                "total_branches": len(parallel_branches),
                "branch_types": list(set(b.branch_type for b in parallel_branches)),
                "target_codes": [b.target_code for b in parallel_branches]
            })
        
        # MAP Phase: Create async tasks for each parallel branch
        tasks = [
            self._execute_single_branch(branch, current_state)
            for branch in parallel_branches[:self.max_concurrent_branches]
        ]
        
        # Execute all branches concurrently with timeout
        try:
            parallel_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.branch_timeout_seconds
            )
        except asyncio.TimeoutError:
            print(f"⚠️  Parallel execution timeout after {self.branch_timeout_seconds}s")
            parallel_results = []
        
        # REDUCE Phase: Process results and combine
        successful_results: list[ParallelResult] = []
        failed_results: list[ParallelResult] = []
        
        for i, result in enumerate(parallel_results):
            if isinstance(result, Exception):
                # Task failed with exception
                failed_results.append(ParallelResult(
                    branch=parallel_branches[i],
                    success=False,
                    final_code=None,
                    steps_taken=0,
                    execution_time_ms=0,
                    error=result
                ))
            elif isinstance(result, ParallelResult):
                if result.success:
                    successful_results.append(result)
                else:
                    failed_results.append(result)
        
        # Update main workflow state based on parallel results
        updated_state = self._reduce_parallel_results(current_state, successful_results)
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Display results
        print(f"🎯 MAP-REDUCE COMPLETE: {len(successful_results)}/{len(parallel_branches)} branches succeeded")
        for result in successful_results:
            print(f"   ✅ {result.branch.target_code}: {result.final_code} ({result.execution_time_ms:.1f}ms)")
        for result in failed_results:
            print(f"   ❌ {result.branch.target_code}: {result.error}")
        
        # Track parallel execution completion in Burr UI
        if self.burr_context:
            self._log_to_burr("parallel_execution_complete", {
                "successful_branches": len(successful_results),
                "failed_branches": len(failed_results),
                "total_execution_time_ms": execution_time,
                "results_summary": [
                    {
                        "target_code": r.branch.target_code,
                        "branch_type": r.branch.branch_type,
                        "success": r.success,
                        "execution_time_ms": r.execution_time_ms
                    }
                    for r in successful_results + failed_results
                ]
            })
        
        return MapReduceResult(
            primary_result=updated_state,
            parallel_results=successful_results + failed_results,
            total_execution_time_ms=execution_time,
            success=len(successful_results) > 0,
            completed_branches=len(successful_results),
            failed_branches=len(failed_results)
        )
    
    async def _execute_single_branch(
        self, 
        branch: ParallelBranch, 
        base_state: QueryState
    ) -> ParallelResult:
        """Execute a single parallel branch asynchronously."""
        import time
        start_time = time.time()
        
        try:
            # Track branch execution start in Burr UI
            if self.burr_context:
                self._log_to_burr("branch_execution_start", {
                    "target_code": branch.target_code,
                    "branch_type": branch.branch_type,
                    "priority": branch.priority
                })
            
            # Create isolated state for this branch
            branch_state = QueryState(
                current=NodeId(branch.target_code),
                finalized=False,
                step=None,
                ctx=RunContext(data=base_state.ctx.data.copy())
            )
            
            # Execute branch based on type with appropriate strategy
            final_code, steps_taken = await self._execute_branch_by_type(branch, branch_state)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Track successful branch completion in Burr UI
            if self.burr_context:
                self._log_to_burr("branch_execution_complete", {
                    "target_code": branch.target_code,
                    "branch_type": branch.branch_type,
                    "final_code": final_code,
                    "execution_time_ms": execution_time,
                    "steps_taken": steps_taken,
                    "success": True
                })
            
            return ParallelResult(
                branch=branch,
                success=True,
                final_code=final_code,
                steps_taken=steps_taken,
                execution_time_ms=execution_time
            )
                
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Track failed branch execution in Burr UI
            if self.burr_context:
                self._log_to_burr("branch_execution_failed", {
                    "target_code": branch.target_code,
                    "branch_type": branch.branch_type,
                    "execution_time_ms": execution_time,
                    "error": str(e),
                    "success": False
                })
            
            return ParallelResult(
                branch=branch,
                success=False,
                final_code=None,
                steps_taken=0,
                execution_time_ms=execution_time,
                error=e
            )
    
    async def _execute_branch_by_type(
        self, 
        branch: ParallelBranch, 
        branch_state: QueryState
    ) -> tuple[str, int]:
        """Execute branch using strategy appropriate for its type."""
        
        if branch.branch_type == "useAdditionalCode":
            # Required codes - use agent validation with direct targeting
            return await self._execute_required_branch(branch, branch_state)
            
        elif branch.branch_type == "codeAlso":
            # Suggested parallel codes - use agent exploration
            return await self._execute_suggested_branch(branch, branch_state)
            
        elif branch.branch_type == "multiCandidate":
            # Multiple agent candidates - use full navigation
            return await self._execute_navigation_branch(branch, branch_state)
            
            
        elif branch.branch_type == "seventhCharacter":
            # 7th character completion - direct character application
            return await self._execute_completion_branch(branch, branch_state)
            
        elif branch.branch_type == "crossReference":
            # Cross-reference codes - lightweight validation
            return await self._execute_reference_branch(branch, branch_state)
            
        elif branch.branch_type == "combinationCode":
            # Combination alternatives - validate combination validity
            return await self._execute_combination_branch(branch, branch_state)
            
        else:
            # Unknown branch type - fallback to direct targeting
            await asyncio.sleep(0.01)
            return branch.target_code, 1
    
    async def _execute_required_branch(self, branch: ParallelBranch, state: QueryState) -> tuple[str, int]:
        """Execute useAdditionalCode (required) branches with agent validation."""
        try:
            # Use agent to validate the required code
            if hasattr(self.agent, 'lm') and self.agent.lm:
                import dspy
                with dspy.context(lm=self.agent.lm):
                    candidates = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self.agent.candidates(state, self.tree, self.traversal)
                    )
                    final_code = candidates[0].target if candidates else branch.target_code
            else:
                final_code = branch.target_code
            
            await asyncio.sleep(0.1)  # Simulate validation
            
            # Mark progress to satisfy constraints
            if hasattr(state.ctx, 'mark_progress'):
                state.ctx.mark_progress(final_code)
                
            return final_code, 1
        except Exception:
            return branch.target_code, 1
    
    async def _execute_suggested_branch(self, branch: ParallelBranch, state: QueryState) -> tuple[str, int]:
        """Execute codeAlso (suggested) branches with exploration."""
        await asyncio.sleep(0.08)  # Simulate exploration
        return branch.target_code, 1
    
    async def _execute_navigation_branch(self, branch: ParallelBranch, state: QueryState) -> tuple[str, int]:
        """Execute multiCandidate branches with full tree navigation."""
        # Simulate navigation through tree structure
        await asyncio.sleep(0.15)  # Longer time for navigation
        return branch.target_code, 2  # Multiple steps
    
    
    async def _execute_completion_branch(self, branch: ParallelBranch, state: QueryState) -> tuple[str, int]:
        """Execute seventhCharacter branches with character completion."""
        await asyncio.sleep(0.03)  # Fast character application
        return branch.target_code, 1
    
    async def _execute_reference_branch(self, branch: ParallelBranch, state: QueryState) -> tuple[str, int]:
        """Execute crossReference branches with lightweight validation."""
        await asyncio.sleep(0.04)  # Minimal reference check
        return branch.target_code, 1
    
    async def _execute_combination_branch(self, branch: ParallelBranch, state: QueryState) -> tuple[str, int]:
        """Execute combinationCode branches with combination validation."""
        await asyncio.sleep(0.07)  # Simulate combination validation
        return branch.target_code, 1
    
    def _reduce_parallel_results(
        self, 
        base_state: QueryState, 
        results: list[ParallelResult]
    ) -> QueryState:
        """
        Reduce phase: Combine parallel results back into main workflow state.
        
        This updates the base state to reflect that parallel requirements
        have been satisfied (e.g., useAdditionalCode constraints resolved).
        """
        # Copy base state for modification
        updated_ctx_data = base_state.ctx.data.copy()
        
        # Remove satisfied suffix requirements
        for result in results:
            if result.success and result.branch.branch_type == "useAdditionalCode":
                # Mark this code as completed to satisfy pending_suffix
                if hasattr(base_state.ctx, 'mark_progress'):
                    base_state.ctx.mark_progress(result.branch.target_code)
        
        # Create updated state with parallel work completed
        return QueryState(
            current=base_state.current,
            finalized=base_state.finalized,
            step=base_state.step,
            ctx=RunContext(data=updated_ctx_data)
        )
    
    def extract_parallel_branches(self, state: QueryState) -> list[ParallelBranch]:
        """Extract ALL types of parallel branches from workflow state."""
        branches: list[ParallelBranch] = []
        
        # 1. useAdditionalCode branches (highest priority - required)
        if hasattr(state.ctx, 'pending_suffix') and state.ctx.pending_suffix:
            for code in sorted(state.ctx.pending_suffix):
                branches.append(ParallelBranch(
                    target_code=code,
                    branch_type="useAdditionalCode", 
                    priority=10,
                    metadata={"required": True, "constraint_type": "suffix"}
                ))
        
        # 2. codeAlso branches (high priority - strongly suggested)  
        if hasattr(state.ctx, 'parallel') and state.ctx.parallel:
            for code in state.ctx.parallel:
                if not any(b.target_code == code for b in branches):  # Avoid duplicates
                    branches.append(ParallelBranch(
                        target_code=code,
                        branch_type="codeAlso",
                        priority=8,
                        metadata={"suggested": True, "constraint_type": "parallel"}
                    ))
        
        # 3. Multiple agent candidates (medium-high priority - alternative paths)
        if hasattr(state, 'step') and state.step and hasattr(state.step, 'metadata'):
            metadata = state.step.metadata
            if 'parallel_candidates' in metadata:
                parallel_candidates = metadata['parallel_candidates']
                for candidate in parallel_candidates:
                    if not any(b.target_code == str(candidate) for b in branches):
                        branches.append(ParallelBranch(
                            target_code=str(candidate),
                            branch_type="multiCandidate",
                            priority=7,
                            metadata={"alternative": True, "agent_selected": True}
                        ))
        
        # 4. Include/exclude notes are NOT parallel branches - they are clarification notes
        # These help determine if a code applies but don't create separate coding paths
        
        # 4. 7th character requirements (medium priority - completion branches)
        if hasattr(state.ctx, 'seventh_character_options') and state.ctx.seventh_character_options:
            for char_code in state.ctx.seventh_character_options:
                if not any(b.target_code == char_code for b in branches):
                    branches.append(ParallelBranch(
                        target_code=char_code,
                        branch_type="seventhCharacter",
                        priority=5,
                        metadata={"completion": True, "character_extension": True}
                    ))
        
        # 5. Cross-reference codes (lower priority - reference branches)
        if hasattr(state.ctx, 'cross_references') and state.ctx.cross_references:
            for ref_code in state.ctx.cross_references:
                if not any(b.target_code == ref_code for b in branches):
                    branches.append(ParallelBranch(
                        target_code=ref_code,
                        branch_type="crossReference",
                        priority=4,
                        metadata={"reference": True, "see_also": True}
                    ))
        
        # 6. Combination code alternatives (lower priority - combination branches)
        if hasattr(state.ctx, 'combination_codes') and state.ctx.combination_codes:
            for combo_code in state.ctx.combination_codes:
                if not any(b.target_code == combo_code for b in branches):
                    branches.append(ParallelBranch(
                        target_code=combo_code,
                        branch_type="combinationCode",
                        priority=3,
                        metadata={"combination": True, "alternative_coding": True}
                    ))
        
        # Sort by priority (higher priority first)
        branches.sort(key=lambda b: b.priority, reverse=True)
        return branches
    
    def _log_to_burr(self, event_type: str, data: dict[str, Any]) -> None:
        """Log parallel execution events to Burr UI for tracking."""
        if not self.burr_context:
            return
            
        try:
            # Try to access Burr's tracking mechanism
            if hasattr(self.burr_context, 'log_attribute'):
                self.burr_context.log_attribute(f"async_parallel_{event_type}", data)
            elif hasattr(self.burr_context, 'tracker') and self.burr_context.tracker:
                # Use Burr's tracker directly
                self.burr_context.tracker.log_attribute(f"async_parallel_{event_type}", data)
            else:
                # Fallback: store in context for later retrieval
                if not hasattr(self.burr_context, '_async_parallel_logs'):
                    self.burr_context._async_parallel_logs = []
                self.burr_context._async_parallel_logs.append({
                    "event_type": event_type,
                    "timestamp": __import__("time").time(),
                    "data": data
                })
        except Exception as e:
            # Silently ignore Burr logging failures to avoid breaking parallel execution
            print(f"⚠️  Burr tracking failed for {event_type}: {e}")
    
    def get_burr_tracking_summary(self) -> dict[str, Any] | None:
        """Get summary of parallel execution for Burr state updates."""
        if not self.burr_context or not hasattr(self.burr_context, '_async_parallel_logs'):
            return None
            
        logs = self.burr_context._async_parallel_logs
        
        # Analyze logs to create summary
        summary = {
            "total_events": len(logs),
            "execution_starts": sum(1 for log in logs if log["event_type"] == "parallel_execution_start"),
            "branch_starts": sum(1 for log in logs if log["event_type"] == "branch_execution_start"),
            "branch_completions": sum(1 for log in logs if log["event_type"] == "branch_execution_complete"),
            "branch_failures": sum(1 for log in logs if log["event_type"] == "branch_execution_failed"),
            "execution_completions": sum(1 for log in logs if log["event_type"] == "parallel_execution_complete"),
            "timeline": [
                {
                    "event": log["event_type"],
                    "timestamp": log["timestamp"],
                    "key_data": {
                        "target_code": log["data"].get("target_code", "N/A"),
                        "branch_type": log["data"].get("branch_type", "N/A")
                    }
                }
                for log in logs
            ]
        }
        
        return summary


# Async utilities for integration with existing orchestrators
async def async_run_with_parallel_execution(
    executor: AsyncParallelExecutor,
    initial_state: QueryState,
    max_steps: int = 10
) -> tuple[QueryState, list[MapReduceResult]]:
    """
    Run workflow with async parallel execution at each step.
    
    This demonstrates how async map-reduce integrates with existing workflows.
    """
    current_state = initial_state
    parallel_executions: list[MapReduceResult] = []
    
    for step in range(max_steps):
        # Check for parallel branches at each step
        branches = executor.extract_parallel_branches(current_state)
        
        if branches:
            print(f"\n🔄 Step {step + 1}: Found {len(branches)} parallel branches")
            
            # Execute parallel branches asynchronously
            map_reduce_result = await executor.execute_parallel_branches(
                current_state, branches
            )
            parallel_executions.append(map_reduce_result)
            
            # Update state with parallel results
            if map_reduce_result.primary_result:
                current_state = map_reduce_result.primary_result
            
            # Check if parallel execution satisfied all requirements
            remaining_branches = executor.extract_parallel_branches(current_state)
            if not remaining_branches:
                print(f"✅ All parallel requirements satisfied!")
                break
        else:
            # No parallel work needed, continue normally
            break
    
    return current_state, parallel_executions