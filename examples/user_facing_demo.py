#!/usr/bin/env python3
"""
CODR User-Facing Clinical Note Processing Demo
==============================================

Demonstrates CODR as a user-facing "low-code" solution where clinical notes
act like high-level syntax that gets processed into ICD-10-CM codes through
natural DAG execution.

When processing stops, it's because agents need more specificity, not because
of technical compiler errors.

Run: PYTHONPATH=. python examples/user_facing_demo.py
"""

import sys
from pathlib import Path
from typing import Any, NamedTuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.domains.medical.trees.icd_tree import create_navigator
from core.domains.medical.traversals.icd_traversal_engine import create_icd_traversal_engine
from core.dag_agents import DeterministicAgent
from core import NodeId

class ProcessingResult(NamedTuple):
    """Result of clinical note processing through the DAG."""
    completed: bool
    final_code: str | None
    is_final: bool
    processing_path: list[str]
    user_feedback: str
    next_steps: list[str]

class ClinicalNoteProcessor:
    """
    CODR Clinical Note Processor.
    
    Processes clinical notes through ICD-10-CM coding workflow using natural
    DAG execution. Stops when agents can't proceed, providing user-friendly feedback.
    """
    
    def __init__(self):
        self.tree = create_navigator()
        self.traversal = create_icd_traversal_engine()
        self.agent = DeterministicAgent()
        
    def process_note(self, clinical_note: str, max_steps: int = 5) -> ProcessingResult:
        """
        Process clinical note through ICD-10-CM workflow.
        
        Natural DAG execution that stops when agents need more information.
        """
        print(f"🏥 PROCESSING: {clinical_note[:60]}...")
        print(f"   Agent: Clinical coding assistant")
        print()
        
        processing_path = ["ROOT"]
        current_position = "ROOT"
        
        for step in range(max_steps):
            # Format current position for display
            display_position = current_position
            if current_position.isdigit() and 1 <= int(current_position) <= 22:
                display_position = f"Chapter {current_position}"
            print(f"   Step {step + 1}: Agent working on '{display_position}'")
            
            # Get what the agent can see
            from core import QueryState, RunContext
            state = QueryState(current=NodeId(current_position), ctx=RunContext())
            available_options = self.traversal.candidate_actions(self.tree, state)
            
            if not available_options:
                # Natural completion - agent reached final code
                return ProcessingResult(
                    completed=True,
                    final_code=current_position,
                    is_final=True,
                    processing_path=processing_path,
                    user_feedback=f"✅ Processing complete! Your clinical note was successfully processed to code: {current_position}",
                    next_steps=[]
                )
            
            print(f"      Agent sees {len(available_options)} possible options")
            
            # Agent tries to make a decision
            agent_choice = self._agent_decision(clinical_note, current_position, available_options)
            
            if not agent_choice:
                # Agent got stuck - needs more info from user
                suggestions = self._get_user_suggestions(current_position)
                return ProcessingResult(
                    completed=False,
                    final_code=current_position,
                    is_final=False,
                    processing_path=processing_path,
                    user_feedback=f"⏸️  Agent paused at: {current_position}\n\nYour clinical note needs more specific details for the agent to continue selecting the best code.",
                    next_steps=suggestions
                )
            
            next_position = str(agent_choice[0])
            processing_path.append(next_position)
            
            # Show agent's choice to user
            target_node = self.tree.get(NodeId(next_position))
            if target_node:
                # Format chapter numbers for display
                display_code = next_position
                if next_position.isdigit() and 1 <= int(next_position) <= 22:
                    display_code = f"Chapter {next_position}"
                print(f"      Agent chose: {display_code} - {target_node.name}")
            
            current_position = next_position
        
        # Reached step limit
        is_final = self.tree.is_leaf(NodeId(current_position))
        
        if is_final:
            return ProcessingResult(
                completed=True,
                final_code=current_position,
                is_final=True,
                processing_path=processing_path,
                user_feedback=f"✅ Processing complete! Final code: {current_position}",
                next_steps=[]
            )
        else:
            suggestions = self._get_user_suggestions(current_position)
            return ProcessingResult(
                completed=False,
                final_code=current_position,
                is_final=False,
                processing_path=processing_path,
                user_feedback=f"📍 Agent reached: {current_position}\n\nConsider adding more clinical details to help the agent reach a more specific code.",
                next_steps=suggestions
            )
    
    def _agent_decision(self, clinical_note: str, current_position: str, available_options):
        """Simulate realistic agent decision-making - shows both successes and limitations."""
        clinical_lower = clinical_note.lower()
        
        # Basic clinical keyword matching (realistic for simple deterministic agent)
        if current_position == "ROOT":
            # Look for diabetes mention to route to endocrine chapter
            if "diabetes" in clinical_lower:
                for option_id, _ in available_options:
                    target_node = self.tree.get(NodeId(str(option_id)))
                    if target_node and target_node.name:
                        name_lower = target_node.name.lower()
                        if "endocrine" in name_lower:
                            return (option_id, _)
            
            # For vague notes, agent can't make good choices - should pause
            if "medical condition" in clinical_lower or len(clinical_lower.split()) < 4:
                return None  # Too vague for agent to decide
        
        elif current_position == "4":  # Endocrine chapter
            # Look for diabetes-related blocks
            if "diabetes" in clinical_lower:
                for option_id, _ in available_options:
                    target_node = self.tree.get(NodeId(str(option_id)))
                    if target_node and target_node.name:
                        name_lower = target_node.name.lower()
                        if "diabetes" in name_lower:
                            return (option_id, _)
        
        elif "E08-E13" in current_position:  # Diabetes block
            # Agent needs diabetes type to continue - realistic limitation
            if "type 1" not in clinical_lower and "type 2" not in clinical_lower:
                return None  # Agent can't decide without type info
                
            # Try to select appropriate type
            for option_id, _ in available_options:
                target_node = self.tree.get(NodeId(str(option_id)))
                if target_node and target_node.name:
                    name_lower = target_node.name.lower()
                    if "type 1" in clinical_lower and ("type 1" in name_lower or "e10" in str(option_id).lower()):
                        return (option_id, _)
                    elif "type 2" in clinical_lower and ("type 2" in name_lower or "e11" in str(option_id).lower()):
                        return (option_id, _)
        
        # Realistic agent behavior: Default to first option when no specific match
        # This shows where simple agents fail and need improvement
        if available_options:
            return available_options[0]
        return None
    
    def _get_user_suggestions(self, current_position: str) -> list[str]:
        """Get user-friendly suggestions for improving their clinical note."""
        children = self.tree.children(NodeId(current_position))
        suggestions = []
        
        for child_id in children[:3]:
            child = self.tree.get(child_id)
            if child:
                suggestions.append(f"• {child_id} - {child.name}")
        
        if not suggestions:
            suggestions = ["Consider adding more specific clinical details"]
        
        return suggestions

def demonstrate_user_facing_processing():
    """Demonstrate user-facing clinical note processing."""
    print("🏥 CODR Clinical Note Processor")
    print("=" * 60)
    print("Your clinical notes act like high-level syntax that gets processed")
    print("into medical codes. The agent will work with what you provide!")
    print()
    
    processor = ClinicalNoteProcessor()
    
    # User scenarios with different levels of detail
    user_scenarios = [
        {
            "name": "Complete Clinical Note",
            "note": "45-year-old patient with Type 2 diabetes mellitus with diabetic nephropathy",
            "expectation": "Should process successfully to specific code"
        },
        {
            "name": "Incomplete Clinical Note", 
            "note": "Patient has diabetes mellitus",
            "expectation": "Agent should pause and ask for diabetes type"
        },
        {
            "name": "Very General Note",
            "note": "Patient has medical condition",
            "expectation": "Agent should pause early and ask for specifics"
        }
    ]
    
    for scenario in user_scenarios:
        print(f"📝 USER SCENARIO: {scenario['name']}")
        print("=" * 50)
        print(f"YOUR INPUT:")
        print(f"   {scenario['note']}")
        print()
        print(f"EXPECTATION:")
        print(f"   {scenario['expectation']}")
        print()
        
        # Process the note
        result = processor.process_note(scenario['note'])
        
        print("PROCESSING RESULT:")
        print("=" * 30)
        if result.completed:
            print(f"✅ SUCCESS")
            print(f"   Final Code: {result.final_code}")
        else:
            print(f"⏸️  PAUSED")
            print(f"   Agent Stopped At: {result.final_code}")
        
        # Format path with chapter numbers
        formatted_path = []
        for code in result.processing_path:
            if code.isdigit() and 1 <= int(code) <= 22:
                formatted_path.append(f"Chapter {code}")
            else:
                formatted_path.append(code)
        print(f"   Path: {' → '.join(formatted_path)}")
        print()
        
        print("FEEDBACK FOR YOU:")
        print(result.user_feedback)
        print()
        
        if result.next_steps:
            print("NEXT STEPS:")
            for step in result.next_steps:
                print(f"   {step}")
            print()
        
        print("HOW TO IMPROVE YOUR NOTE:")
        if result.is_final:
            print("   ✅ Your clinical note provided perfect detail!")
        else:
            print("   🔧 Add more specific medical details")
            print("   💡 Include diagnosis types, complications, or anatomical specifics")
            print("   📝 The more specific your note, the more specific the final code")
        
        print("\n" + "=" * 80 + "\n")

def show_user_experience_philosophy():
    """Show the user experience design philosophy."""
    print("🎯 USER EXPERIENCE DESIGN")
    print("=" * 60)
    print()
    
    print("YOUR CLINICAL NOTES AS HIGH-LEVEL SYNTAX:")
    print("   'Type 1 diabetes with neuropathy' → Agent processes → E10.40")
    print("   'Patient has diabetes' → Agent pauses → 'Need diabetes type'")
    print("   'Medical condition' → Agent pauses → 'Need condition details'")
    print()
    
    print("NATURAL PROCESSING FLOW:")
    print("   📝 You write clinical note")
    print("   🤖 Agent processes through medical coding workflow")  
    print("   ⏸️  Agent pauses when it needs more details from you")
    print("   💡 You get friendly suggestions on what to add")
    print("   ✅ Agent reaches final code when note is complete")
    print()
    
    print("WHY AGENTS PAUSE:")
    print("   • Your note needs more diagnostic specificity")
    print("   • Multiple valid paths exist and agent needs guidance")
    print("   • Medical terminology requires additional clinical context")
    print()
    
    print("USER-FRIENDLY DESIGN PRINCIPLES:")
    print("   🚫 No technical error messages")
    print("   💡 Helpful suggestions for note improvement")  
    print("   🎯 Focus on what YOU need to add, not system problems")
    print("   📋 Clear feedback on why the agent paused")

if __name__ == "__main__":
    demonstrate_user_facing_processing()
    show_user_experience_philosophy()