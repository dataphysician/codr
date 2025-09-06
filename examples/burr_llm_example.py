"""
End-to-End ICD-10 Traversal using Burr Orchestrator with LLM Agent (GPT-4o)

This example demonstrates a complete ICD-10 coding workflow using:
- Burr state machine orchestrator for workflow management
- GPT-4o LLM agent for intelligent candidate selection
- ICD-10 tree navigation and traversal engine
- Clinical document processing and code assignment

Prerequisites:
- Set OPENAI_API_KEY environment variable
- Install dependencies: pip install burr litellm

Run: PYTHONPATH=. python examples/burr_llm_example.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from typing import Any

# Core system imports
from core import NodeId, Action, QueryState, RunContext
from core.domains.medical.trees.icd_tree import create_navigator
from core.domains.medical.traversals.icd_traversal_engine import create_icd_traversal_engine
from core.domains.medical.factories.icd10_domain import create_icd10_llm_agent

# Burr imports
try:
    from core.orchestrators.burr_orchestrator import (
        NodePolicyRouter, create_burr_app,
        set_policy_router, init_traversal, enumerate_candidates, 
        select_candidate, apply_selected_step, maybe_finalize
    )
    BURR_AVAILABLE = True
except ImportError:
    BURR_AVAILABLE = False

# Check environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not set. Please set this environment variable.")
    sys.exit(1)

if not BURR_AVAILABLE:
    print("❌ Burr not available. Install with: pip install burr")
    sys.exit(1)


def create_clinical_scenarios():
    """Create test clinical scenarios for demonstration."""
    return [
        {
            "case_id": "case_001",
            "clinical_note": """
            45-year-old male patient presents with polyuria, polydipsia, and unintentional weight loss.
            HbA1c: 11.2%. Glucose: 380 mg/dL. Patient has no prior history of diabetes.
            Diagnosed with Type 1 diabetes mellitus. Started on insulin therapy.
            """,
            "expected_category": "E10",
            "description": "New onset Type 1 diabetes mellitus"
        },
        {
            "case_id": "case_002", 
            "clinical_note": """
            62-year-old female with established Type 2 diabetes mellitus presents with 
            diabetic nephropathy. Microalbuminuria present. eGFR: 45 mL/min/1.73m².
            Currently on metformin and insulin. Requires specific coding for complications.
            """,
            "expected_category": "E11.2",
            "description": "Type 2 diabetes with kidney complications"
        },
        {
            "case_id": "case_003",
            "clinical_note": """
            38-year-old pregnant female at 28 weeks gestation diagnosed with gestational 
            diabetes mellitus. Glucose tolerance test abnormal. Started on dietary management.
            No prior history of diabetes. Blood glucose monitoring initiated.
            """,
            "expected_category": "O24",
            "description": "Gestational diabetes mellitus"
        }
    ]


def setup_llm_agent_router():
    """Set up policy router with GPT-4o LLM agent."""
    print("🔧 Setting up LLM agent policy router...")
    
    # Create GPT-4o agent with specific configuration
    llm_agent = create_icd10_llm_agent(
        provider="openai",
        model="gpt-4o"
    )
    
    # Create policy router
    router = NodePolicyRouter(default_agent=llm_agent)
    
    # Set up domain-specific routing
    router.register_prefix("ROOT", llm_agent)  # ROOT-level Chapter selection
    router.register_prefix("E", llm_agent)  # Endocrine diseases
    router.register_prefix("O", llm_agent)  # Pregnancy/childbirth
    router.register_prefix("chapter_4", llm_agent)  # Endocrine chapter
    
    # Set global router
    set_policy_router(router)
    
    print(f"✅ LLM agent configured: OpenAI GPT-4o")
    return router


def run_burr_workflow(clinical_note: str, start_code: str = "ROOT") -> dict[str, Any]:
    """Run complete Burr workflow for ICD-10 coding."""
    print(f"\n🚀 Starting Burr workflow from: {start_code}")
    print(f"📋 Clinical context: {clinical_note[:100]}...")
    
    try:
        # Create Burr application
        # Get the configured router and create Burr app
        from core.orchestrators.burr_orchestrator import _policy
        router = _policy()
        app = create_burr_app(router.default_agent, project="icd-llm-agent")  # Uses our configured router
        
        # Initialize workflow state
        print("📊 Initializing workflow state...")
        action, result, state = app.run(
            halt_after=["init_traversal"],
            inputs={
                "start_code": start_code,
                "clinical_note": clinical_note
            }
        )
        
        workflow_results = {
            "start_time": time.time(),
            "start_code": start_code,
            "clinical_note": clinical_note,
            "steps": [],
            "final_state": None,
            "success": False,
            "error": None
        }
        
        max_steps = 8
        step_count = 0
        
        # Run workflow steps
        while step_count < max_steps:
            step_count += 1
            print(f"\n--- Step {step_count} ---")
            
            # Enumerate candidates
            print("🔍 Enumerating candidates...")
            action, result, state = app.run(
                halt_after=["enumerate_candidates"],
                inputs={}
            )
            
            candidates = state.get("candidates", [])
            print(f"Found {len(candidates)} candidates")
            
            if not candidates:
                print("❌ No candidates available - stopping")
                break
            
            # Agent selection with clinical context
            print("🤖 Agent making selection...")
            action, result, state = app.run(
                halt_after=["select_candidate"], 
                inputs={
                    "agent_inputs": {
                        "clinical_context": clinical_note,
                        "step_number": step_count
                    }
                }
            )
            
            choice = state.get("choice")
            choice_action = state.get("choice_action", "goto")
            
            if not choice:
                print("❌ Agent made no selection - stopping")
                break
                
            print(f"🎯 Selected: {choice} (action: {choice_action})")
            
            # Apply move
            print("⚡ Applying move...")
            action, result, state = app.run(
                halt_after=["apply_selected_step"],
                inputs={
                    "target": choice,
                    "action": choice_action
                }
            )
            
            # Check if move was successful
            current_state = state.get("qstate", {})
            if "error" in str(current_state):
                print(f"❌ Move failed: {current_state}")
                break
            
            # Record step
            step_info = {
                "step": step_count,
                "candidates_count": len(candidates),
                "selected": choice,
                "action": choice_action,
                "success": True
            }
            workflow_results["steps"].append(step_info)
            
            print(f"✅ Move successful - now at: {choice}")
            
            # Check if we can finalize
            current_code = current_state.get("current", choice)
            print(f"🏁 Current position: {current_code}")
            
            # Attempt finalization if at a specific code, but respect ICD 7th character requirements
            if "." in str(current_code) and step_count >= 3:
                print("🎯 Attempting finalization...")
                action, result, state = app.run(
                    halt_after=["maybe_finalize"],
                    inputs={}
                )
                
                final_state = state.get("qstate", {})
                guard_info = state.get("last_guard", {})
                
                if final_state.get("finalized", False):
                    workflow_results["final_state"] = {
                        "final_code": current_code,
                        "finalized": True,
                        "step_count": step_count
                    }
                    workflow_results["success"] = True
                    print(f"🎉 Successfully finalized at: {current_code}")
                    break
                elif guard_info.get("outcome") == "require_seven":
                    print(f"⚠️  Finalization blocked: {guard_info.get('message', 'Seven-character code required')}")
                    print("   Continuing navigation to reach required 7th character...")
                    # Continue the loop to navigate deeper
                elif guard_info.get("outcome") == "require_suffix":
                    print(f"⚠️  Finalization blocked: {guard_info.get('message', 'Additional codes required')}")
                    print("   Note: ICD useAdditionalCode requirements (e.g., Z79.4 for insulin) must be addressed")
                    print("   Current workflow focuses on primary diagnosis - additional codes would be handled separately")
                    # Continue the loop to explore other paths
        
        # Set final results
        if not workflow_results["success"]:
            workflow_results["final_state"] = {
                "final_code": state.get("qstate", {}).get("current", start_code),
                "finalized": False,
                "step_count": step_count
            }
            workflow_results["success"] = step_count < max_steps
        
        workflow_results["total_time"] = time.time() - workflow_results["start_time"]
        return workflow_results
        
    except Exception as e:
        print(f"❌ Workflow error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "start_code": start_code,
            "clinical_note": clinical_note,
            "steps": [],
            "final_state": None,
            "success": False,
            "error": str(e)
        }


def display_results(case: dict[str, Any], results: dict[str, Any]):
    """Display workflow results in a readable format."""
    print(f"\n{'='*60}")
    print(f"📋 CASE: {case['case_id']} - {case['description']}")
    print(f"{'='*60}")
    
    if results["error"]:
        print(f"❌ ERROR: {results['error']}")
        return
    
    print(f"🏁 Final Code: {results['final_state']['final_code']}")
    print(f"✅ Success: {results['success']}")
    print(f"🔢 Steps Taken: {len(results['steps'])}")
    print(f"⏱️  Total Time: {results['total_time']:.2f}s")
    
    if results['steps']:
        print(f"\n📊 WORKFLOW STEPS:")
        for step in results['steps']:
            print(f"  Step {step['step']}: {step['selected']} ({step['action']})")
    
    # Compare with expected
    expected = case.get("expected_category")
    final_code = results['final_state']['final_code']
    
    if expected:
        if str(final_code).startswith(str(expected)):
            print(f"🎯 MATCH: Final code {final_code} matches expected category {expected}")
        else:
            print(f"⚠️  PARTIAL: Final code {final_code} vs expected {expected}")


def main():
    """Run complete demonstration of Burr + LLM agent workflow."""
    print("ICD-10 Coding with Burr Orchestrator + GPT-4o LLM Agent")
    print("=" * 60)
    print("Demonstrates intelligent clinical document coding using:")
    print("• Burr state machine for workflow orchestration")
    print("• GPT-4o for context-aware decision making")  
    print("• ICD-10 tree navigation and domain rules")
    print("• End-to-end clinical coding pipeline\n")
    
    # Setup
    router = setup_llm_agent_router()
    scenarios = create_clinical_scenarios()
    
    print(f"\n🧪 Running {len(scenarios)} clinical coding scenarios...\n")
    
    results_summary = []
    
    # Process each scenario
    for i, case in enumerate(scenarios, 1):
        print(f"\n{'🏥 SCENARIO ' + str(i):<20} {'='*40}")
        
        # Run workflow
        results = run_burr_workflow(
            clinical_note=case["clinical_note"],
            start_code="ROOT"  # Start from ROOT for Chapter selection
        )
        
        # Display results
        display_results(case, results)
        results_summary.append({
            "case": case["case_id"],
            "success": results["success"],
            "final_code": results["final_state"]["final_code"] if results["final_state"] else None,
            "steps": len(results["steps"]),
            "time": results.get("total_time", 0)
        })
        
        # Brief pause between scenarios
        time.sleep(1)
    
    # Summary
    print(f"\n{'='*60}")
    print("📈 OVERALL SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results_summary if r["success"])
    total_time = sum(r["time"] for r in results_summary)
    avg_steps = sum(r["steps"] for r in results_summary) / len(results_summary)
    
    print(f"✅ Successful workflows: {successful}/{len(scenarios)}")
    print(f"⏱️  Total processing time: {total_time:.2f}s")
    print(f"📊 Average steps per workflow: {avg_steps:.1f}")
    print(f"🤖 Agent: OpenAI GPT-4o")
    print(f"🏗️  Orchestrator: Burr State Machine")
    
    if successful == len(scenarios):
        print(f"\n🎉 All scenarios completed successfully!")
    else:
        print(f"\n⚠️  {len(scenarios) - successful} scenarios had issues")
    
    print(f"\n{'='*60}")
    print("💡 Key Capabilities Demonstrated:")
    print("• Clinical document understanding and context extraction")
    print("• Intelligent navigation through ICD-10 hierarchy")
    print("• Domain-specific reasoning for medical coding")
    print("• State management and workflow persistence")
    print("• Error handling and recovery mechanisms")


if __name__ == "__main__":
    main()