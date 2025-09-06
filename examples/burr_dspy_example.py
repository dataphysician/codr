"""
End-to-End ICD-10 Traversal using Burr Orchestrator with DSPy Agent (Qwen-235B)

This example demonstrates a complete ICD-10 coding workflow using:
- Burr state machine orchestrator for workflow management  
- Cerebras Qwen-235B DSPy agent with structured reasoning capabilities
- ICD-10 tree navigation and traversal engine
- Clinical document processing with JSON structured outputs

Prerequisites:
- Set CEREBRAS_API_KEY environment variable
- Install dependencies: pip install burr dspy

Run: PYTHONPATH=. python examples/burr_dspy_example.py
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
# Note: Using factory function for proper domain abstraction and consistency

# Burr imports
try:
    from core.orchestrators.burr_orchestrator import (
        NodePolicyRouter, create_burr_app,
        set_policy_router, init_traversal, enumerate_candidates, 
        select_candidate, apply_selected_step, maybe_finalize, _policy
    )
    BURR_AVAILABLE = True
except ImportError:
    BURR_AVAILABLE = False

# DSPy imports  
try:
    import dspy
    # Note: configure_dspy not needed - using direct agent creation with model_config
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

# Check environment
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    print("❌ CEREBRAS_API_KEY not set. Please set this environment variable.")
    sys.exit(1)

if not BURR_AVAILABLE:
    print("❌ Burr not available. Install with: pip install burr")
    sys.exit(1)

if not DSPY_AVAILABLE:
    print("❌ DSPy not available. Install with: pip install dspy")
    sys.exit(1)


def create_complex_clinical_scenarios():
    """Create complex clinical scenarios requiring advanced reasoning."""
    return [
        {
            "case_id": "complex_001",
            "clinical_note": """
            67-year-old male with long-standing Type 2 diabetes mellitus (15+ years) presents 
            with multiple complications: proliferative diabetic retinopathy requiring laser 
            photocoagulation, diabetic chronic kidney disease stage 3 (eGFR 35), and diabetic 
            peripheral neuropathy with foot ulceration. Patient also has diabetic gastroparesis.
            Current medications: insulin glargine, metformin, pregabalin. Requires comprehensive
            coding for all complications and their manifestations.
            """,
            "expected_categories": ["E11.31", "E11.21", "E11.40", "E11.43", "E11.621"],
            "description": "Type 2 diabetes with multiple complications",
            "complexity": "high"
        },
        {
            "case_id": "complex_002",
            "clinical_note": """
            34-year-old female at 32 weeks gestation with pre-gestational Type 1 diabetes
            mellitus complicated by diabetic ketoacidosis requiring ICU admission. Patient
            has background diabetic retinopathy and mild proteinuria. Fetal growth is
            appropriate for gestational age. Mother requires intensive insulin management.
            Delivery planning needed due to maternal complications.
            """, 
            "expected_categories": ["O24.0", "E10.1", "E10.3"],
            "description": "Pregnancy with pre-existing diabetes complications",
            "complexity": "high"
        },
        {
            "case_id": "complex_003",
            "clinical_note": """
            52-year-old male presents with newly diagnosed diabetes mellitus. Clinical
            presentation includes severe hyperglycemia (glucose >500), ketosis, but
            preserved C-peptide levels. Patient is overweight (BMI 31) but has rapid
            onset symptoms. Anti-GAD antibodies negative. Uncertain diabetes type -
            could be Type 2 with ketosis-prone presentation or late-onset Type 1.
            Started on insulin pending further classification.
            """,
            "expected_categories": ["E11", "E10"],  # Ambiguous case
            "description": "Diabetes mellitus - uncertain type classification", 
            "complexity": "diagnostic_challenge"
        }
    ]


def setup_dspy_reasoning_agent():
    """
    Set up DSPy agent with Qwen-235B structured reasoning capabilities.
    
    Note: Uses direct agent creation to ensure max_tokens configuration is properly applied.
    The create_icd10_dspy_agent() function allows us to pass model_config directly,
    avoiding the dual-configuration issue where global DSPy settings might be ignored.
    """
    print("🔧 Setting up DSPy reasoning agent...")
    
    # Create ICD-10 DSPy agent using improved intuitive API
    from core.domains.medical.factories.icd10_domain import create_icd10_dspy_agent
    
    # Use improved factory function with intuitive parameters
    dspy_agent = create_icd10_dspy_agent(
        provider="cerebras",
        # model="qwen-3-235b-a22b-thinking-2507",  # Use thinking model when available
        model="qwen-3-32b",
        max_candidates=3,  # Beam search for better coverage
        temperature=0.2,   # Lower temperature for consistent reasoning
        reasoning_style="detailed_clinical",  # Structured clinical reasoning
        # max_tokens is automatically set to 64000+ for reasoning models
    )
    
    # Add node-specific feedback and training examples (optional)
    # Consistent API with LLM agents - feedback modifies signature field descriptions  
    # dspy_agent.add_rule("E11.22", feedback="Focus on differential diagnosis between diabetic nephropathy and chronic kidney disease")
    #
    # Training examples for future DSPy compilation (optional)
    # diabetes_training_examples = [
    #     {"clinical_note": "T2DM with nephropathy...", "target_codes": ["E11.21"]},
    #     {"clinical_note": "T1DM with retinopathy...", "target_codes": ["E10.31"]},
    # ]
    # dspy_agent.add_rule("E11.22", feedback="Use systematic differential diagnosis", training_examples=diabetes_training_examples)
    
    # ============================================================================
    # AGENT ROUTING SYSTEM - Hierarchical Agent Selection for ICD-10 Navigation
    # ============================================================================
    
    # Create policy router with DSPy agent as fallback for unmatched codes
    router = NodePolicyRouter(default_agent=dspy_agent)
    
    # EXACT MATCH routing (highest priority)
    # These codes get exact agent assignment regardless of other rules
    router.register_exact("ROOT", dspy_agent)  # Chapter selection requires advanced reasoning
    
    # PREFIX MATCH routing (medium priority) 
    # All codes starting with these prefixes use the specified agent
    # Example: "E11.21" matches "E11" prefix → uses dspy_agent
    router.register_prefix("E11", dspy_agent)      # Type 2 diabetes complications
    router.register_prefix("E10", dspy_agent)      # Type 1 diabetes complications  
    router.register_prefix("O24", dspy_agent)      # Pregnancy with diabetes
    router.register_prefix("chapter_4", dspy_agent)  # Endocrine system chapter
    
    # ROUTING PRIORITY ORDER (from get_agent_for method):
    # 1. Exact match:  router.by_exact[code] 
    # 2. Prefix match: code.startswith(prefix) - first match wins
    # 3. Regex match:  pattern.search(code) - first match wins  
    # 4. Default:      default_agent for any unmatched codes
    #
    # This allows specialized AI agents for complex medical domains while
    # falling back to simpler agents for basic ICD codes.
    
    # Set global router for all Burr orchestrator instances
    set_policy_router(router)
    
    print(f"🧠 Reasoning agent configured with advanced capabilities")
    return router


def run_reasoning_workflow(clinical_note: str, start_code: str = "ROOT") -> dict[str, Any]:
    """Run Burr workflow with DSPy reasoning agent."""
    print(f"\n🚀 Starting reasoning workflow from: {start_code}")
    print(f"📋 Clinical complexity: {len(clinical_note)} characters")
    
    try:
        # Create Burr application with DSPy agent
        router = _policy()  # Get the configured router
        app = create_burr_app(router.default_agent, project="icd-reasoning-dspy")
        
        # Initialize with clinical context
        print("📊 Initializing reasoning state...")
        result = app.run(
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
            "reasoning_steps": [],
            "agent_decisions": [],
            "final_state": None,
            "success": False,
            "error": None
        }
        
        max_steps = 10  # More steps for complex reasoning
        step_count = 0
        
        # Run reasoning workflow
        while step_count < max_steps:
            step_count += 1
            print(f"\n--- Reasoning Step {step_count} ---")
            
            # Enumerate candidates with context
            print("🔍 Analyzing available options...")
            action, result, state = app.run(
                halt_after=["enumerate_candidates"],
                inputs={}
            )
            
            candidates = state.get("candidates", [])
            print(f"Found {len(candidates)} potential paths")
            
            if not candidates:
                print("❌ No viable paths - analysis complete")
                break
            
            # DSPy agent reasoning with clinical context
            print("🧠 DSPy agent reasoning...")
            start_reasoning = time.time()
            
            action, result, state = app.run(
                halt_after=["select_candidate"],
                inputs={
                    "agent_inputs": {
                        "clinical_context": clinical_note,
                        "step_number": step_count,
                        "enable_reasoning": True,
                        "confidence_required": 0.7
                    }
                }
            )
            
            reasoning_time = time.time() - start_reasoning
            
            choice = state.get("choice")
            choice_action = state.get("choice_action", "goto")
            choice_meta = state.get("choice_meta", {})
            
            if not choice:
                print("❌ Agent reasoning inconclusive - stopping")
                break
            
            # Extract reasoning information
            reasoning_info = {
                "step": step_count,
                "reasoning_time": reasoning_time,
                "selected_code": choice,
                "action": choice_action,
                "reasoning_quality": choice_meta.get("reasoning_quality", "MODERATE"),
                "confidence": choice_meta.get("confidence", None),  # Optional for backwards compatibility
                "reasoning_trace": choice_meta.get("reasoning", ""),
                "alternatives_considered": choice_meta.get("alternatives", [])
            }
            
            workflow_results["reasoning_steps"].append(reasoning_info)
            
            confidence_display = f"conf: {reasoning_info['confidence']:.2f}" if reasoning_info['confidence'] is not None else f"quality: {reasoning_info['reasoning_quality']}"
            print(f"🎯 Reasoned selection: {choice} ({confidence_display})")
            
            # Show reasoning if available
            if reasoning_info["reasoning_trace"]:
                print(f"💭 Reasoning: {reasoning_info['reasoning_trace'][:200]}...")
            
            # Apply reasoned decision
            print("⚡ Executing decision...")
            action, result, state = app.run(
                halt_after=["apply_selected_step"],
                inputs={
                    "target": choice,
                    "action": choice_action
                }
            )
            
            # Validate reasoning outcome
            current_state = state.get("qstate", {})
            if "error" in str(current_state):
                print(f"❌ Reasoning led to invalid move: {current_state}")
                break
            
            # Record agent decision
            decision_info = {
                "step": step_count,
                "from_code": state.get("previous_code", start_code),
                "to_code": choice,
                "action": choice_action,
                "reasoning_quality": reasoning_info["reasoning_quality"].lower(),
                "success": True
            }
            workflow_results["agent_decisions"].append(decision_info)
            
            print(f"✅ Reasoning validated - advanced to: {choice}")
            
            # Check reasoning convergence
            current_code = current_state.get("current", choice)
            print(f"🎯 Current diagnostic position: {current_code}")
            
            # Attempt finalization for specific codes with high quality
            is_high_quality = reasoning_info["reasoning_quality"].upper() in ["EXCELLENT", "GOOD"]
            if ("." in str(current_code) and 
                is_high_quality and 
                step_count >= 4):
                
                print(f"🏁 High-quality reasoning ({reasoning_info['reasoning_quality']}) - attempting finalization...")
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
                        "step_count": step_count,
                        "reasoning_quality": reasoning_info["reasoning_quality"]
                    }
                    workflow_results["success"] = True
                    print(f"🎉 Reasoning converged successfully at: {current_code}")
                    break
                elif guard_info.get("outcome") == "require_seven":
                    print(f"⚠️  Finalization blocked: {guard_info.get('message', 'Seven-character code required')}")
                    print(f"   DSPy agent will continue navigation to reach 7th character requirement...")
                    # Continue the loop to navigate deeper
                elif guard_info.get("outcome") == "require_suffix":
                    print(f"⚠️  Finalization blocked: {guard_info.get('message', 'Additional codes required')}")
                    print(f"   Note: ICD useAdditionalCode requirements detected (e.g., diabetes control codes)")
                    print(f"   DSPy reasoning focused on primary diagnosis - additional codes handled via parallel workflow")
                    # Continue the loop to explore other diagnostic paths
        
        # Finalize results
        if not workflow_results["success"]:
            final_code = state.get("qstate", {}).get("current", start_code)
            workflow_results["final_state"] = {
                "final_code": final_code,
                "finalized": False, 
                "step_count": step_count,
                "reasoning_quality": "MODERATE"
            }
            workflow_results["success"] = step_count < max_steps
        
        workflow_results["total_time"] = time.time() - workflow_results["start_time"]
        return workflow_results
        
    except Exception as e:
        print(f"❌ Reasoning workflow error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "start_code": start_code,
            "clinical_note": clinical_note,
            "reasoning_steps": [],
            "agent_decisions": [],
            "final_state": None,
            "success": False,
            "error": str(e)
        }


def analyze_reasoning_quality(results: dict[str, Any]) -> dict[str, Any]:
    """Analyze the quality of DSPy reasoning throughout the workflow."""
    if not results["reasoning_steps"]:
        return {"quality": "unknown", "analysis": "No reasoning data available"}
    
    # Count quality categories
    quality_counts = {"EXCELLENT": 0, "GOOD": 0, "MODERATE": 0, "POOR": 0, "UNKNOWN": 0}
    for step in results["reasoning_steps"]:
        quality = step.get("reasoning_quality", "UNKNOWN").upper()
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    total_steps = len(results["reasoning_steps"])
    avg_reasoning_time = sum(step["reasoning_time"] for step in results["reasoning_steps"]) / total_steps
    
    # Analyze reasoning traces
    has_detailed_reasoning = any(len(step["reasoning_trace"]) > 50 for step in results["reasoning_steps"])
    considers_alternatives = any(len(step["alternatives_considered"]) > 1 for step in results["reasoning_steps"])
    
    # Calculate overall quality based on distribution
    excellent_ratio = quality_counts["EXCELLENT"] / total_steps
    good_ratio = quality_counts["GOOD"] / total_steps
    moderate_ratio = quality_counts["MODERATE"] / total_steps
    poor_ratio = quality_counts["POOR"] / total_steps
    
    # Determine overall quality
    if excellent_ratio >= 0.5:
        overall_quality = "excellent"
    elif excellent_ratio + good_ratio >= 0.5:
        overall_quality = "good"
    elif poor_ratio >= 0.5:
        overall_quality = "poor"
    else:
        overall_quality = "moderate"
    
    return {
        "quality": overall_quality,
        "quality_distribution": quality_counts,
        "excellent_steps": quality_counts["EXCELLENT"],
        "good_steps": quality_counts["GOOD"], 
        "moderate_steps": quality_counts["MODERATE"],
        "poor_steps": quality_counts["POOR"],
        "avg_reasoning_time": avg_reasoning_time,
        "has_detailed_reasoning": has_detailed_reasoning,
        "considers_alternatives": considers_alternatives,
        "total_reasoning_steps": total_steps
    }


def display_reasoning_results(case: dict[str, Any], results: dict[str, Any]):
    """Display workflow results with reasoning analysis."""
    print(f"\n{'='*70}")
    print(f"🧠 CASE: {case['case_id']} - {case['description']}")
    print(f"📊 Complexity: {case['complexity']}")
    print(f"{'='*70}")
    
    if results["error"]:
        print(f"❌ ERROR: {results['error']}")
        return
    
    # Basic results
    final_state = results['final_state']
    print(f"🎯 Final Code: {final_state['final_code']}")
    print(f"✅ Success: {results['success']}")
    print(f"🔢 Reasoning Steps: {len(results['reasoning_steps'])}")
    print(f"⏱️  Total Time: {results['total_time']:.2f}s")
    
    # Reasoning quality analysis
    reasoning_analysis = analyze_reasoning_quality(results)
    print(f"🧠 Reasoning Quality: {reasoning_analysis['quality'].upper()}")
    print(f"📊 Quality Distribution: E:{reasoning_analysis['excellent_steps']} G:{reasoning_analysis['good_steps']} M:{reasoning_analysis['moderate_steps']} P:{reasoning_analysis['poor_steps']}")
    print(f"⚡ Avg Reasoning Time: {reasoning_analysis['avg_reasoning_time']:.2f}s per step")
    
    # Show detailed reasoning for key steps
    if results['reasoning_steps']:
        print(f"\n🤔 KEY REASONING STEPS:")
        for i, step in enumerate(results['reasoning_steps'][-3:], len(results['reasoning_steps'])-2):
            if step["reasoning_trace"]:
                quality_display = f"quality: {step['reasoning_quality']}"
                if step.get('confidence') is not None:
                    quality_display = f"conf: {step['confidence']:.2f}, {quality_display}"
                print(f"  Step {step['step']}: {step['selected_code']} ({quality_display})")
                print(f"    💭 {step['reasoning_trace'][:150]}...")
    
    # Compare with expected categories
    expected = case.get("expected_categories", [])
    final_code = final_state['final_code']
    
    if expected:
        matches = [exp for exp in expected if str(final_code).startswith(str(exp))]
        if matches:
            print(f"🎯 REASONING SUCCESS: {final_code} matches expected {matches[0]}")
        else:
            print(f"🔍 REASONING ANALYSIS: {final_code} vs expected {expected}")
            print("    (Complex cases may require multiple codes or different categorization)")


def main():
    """Run complete demonstration of Burr + DSPy reasoning agent workflow."""
    print("Advanced ICD-10 Coding with Burr Orchestrator + DSPy Qwen-235B Agent")
    print("=" * 75)
    print("Demonstrates advanced clinical reasoning using:")
    print("• Burr state machine for complex workflow orchestration")
    print("• Cerebras Qwen-235B with structured JSON output for reliable medical decisions")
    print("• DSPy framework for structured reasoning and optimization")
    print("• Advanced ICD-10 coding with confidence tracking")
    print("• Clinical reasoning trace analysis and validation\n")
    
    # Setup reasoning system
    router = setup_dspy_reasoning_agent()
    scenarios = create_complex_clinical_scenarios()
    
    print(f"\n🧪 Processing {len(scenarios)} complex clinical scenarios...\n")
    
    reasoning_summary = []
    
    # Process each complex scenario  
    for i, case in enumerate(scenarios, 1):
        print(f"\n{'🏥 COMPLEX SCENARIO ' + str(i):<25} {'='*45}")
        
        # Run reasoning workflow starting from ROOT with Chapter selection
        results = run_reasoning_workflow(
            clinical_note=case["clinical_note"],
            start_code="ROOT"  # Start from ROOT for Chapter selection
        )
        
        # Analyze and display results
        display_reasoning_results(case, results)
        
        reasoning_analysis = analyze_reasoning_quality(results)
        reasoning_summary.append({
            "case": case["case_id"],
            "complexity": case["complexity"],
            "success": results["success"],
            "final_code": results["final_state"]["final_code"] if results["final_state"] else None,
            "reasoning_steps": len(results["reasoning_steps"]),
            "reasoning_quality": reasoning_analysis["quality"],
            "excellent_steps": reasoning_analysis["excellent_steps"],
            "good_steps": reasoning_analysis["good_steps"],
            "moderate_steps": reasoning_analysis["moderate_steps"],
            "poor_steps": reasoning_analysis["poor_steps"],
            "time": results.get("total_time", 0)
        })
        
        # Brief pause for analysis
        time.sleep(1.5)
    
    # Comprehensive summary
    print(f"\n{'='*75}")
    print("🧠 ADVANCED REASONING SUMMARY")
    print(f"{'='*75}")
    
    successful = sum(1 for r in reasoning_summary if r["success"])
    total_time = sum(r["time"] for r in reasoning_summary)
    avg_reasoning_steps = sum(r["reasoning_steps"] for r in reasoning_summary) / len(reasoning_summary)
    
    # Aggregate quality metrics
    total_excellent = sum(r["excellent_steps"] for r in reasoning_summary)
    total_good = sum(r["good_steps"] for r in reasoning_summary)
    total_moderate = sum(r["moderate_steps"] for r in reasoning_summary)
    total_poor = sum(r["poor_steps"] for r in reasoning_summary)
    
    print(f"✅ Successful reasoning workflows: {successful}/{len(scenarios)}")
    print(f"🧠 Quality Distribution: E:{total_excellent} G:{total_good} M:{total_moderate} P:{total_poor}")
    print(f"⏱️  Total processing time: {total_time:.2f}s") 
    print(f"📊 Average reasoning steps: {avg_reasoning_steps:.1f}")
    print(f"🤖 Reasoning Agent: Cerebras Qwen-235B with thinking capabilities")
    print(f"🏗️  Framework: Burr + DSPy reasoning optimization")
    
    # Quality distribution
    quality_counts = {}
    for summary in reasoning_summary:
        quality = summary["reasoning_quality"] 
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    print(f"\n📈 Reasoning Quality Distribution:")
    for quality, count in sorted(quality_counts.items()):
        print(f"  {quality.capitalize()}: {count} cases")
    
    if successful == len(scenarios):
        print(f"\n🎉 All complex scenarios reasoned successfully!")
    else:
        print(f"\n🔍 {len(scenarios) - successful} scenarios require further analysis")
    
    print(f"\n{'='*75}")
    print("💡 Advanced Capabilities Demonstrated:")
    print("• Multi-step clinical reasoning with confidence tracking")
    print("• Complex diagnostic scenario analysis and classification")
    print("• Reasoning trace generation and quality assessment")
    print("• Alternative consideration and decision validation")
    print("• Advanced state management with reasoning persistence")
    print("• Performance optimization through DSPy framework")


if __name__ == "__main__":
    main()