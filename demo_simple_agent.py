"""
Simple Agent Demo for ICD-10-CM Coding
======================================

Hackathon-ready agent that traverses ICD tree based on medical documents.
Uses simple prompting to determine next steps in coding traversal.
"""

import json
from icd_tree import (
    create_simple_navigator, 
    get_chapters_for_selection, 
    get_node_details, 
    get_children_with_context,
    find_codes_by_search
)


class SimpleCodingAgent:
    """
    Simple agent for ICD-10-CM coding traversal.
    
    Agent workflow:
    1. Start with chapter selection based on medical document
    2. Navigate through hierarchy using context and document
    3. Stop when node is appropriate for documented condition
    4. Allow multiple selections at each step
    """
    
    def __init__(self):
        self.navigator = create_simple_navigator()
        self.current_codes = []  # Can select multiple codes
        self.traversal_history = []
        
    def analyze_document_for_chapters(self, medical_document: str) -> list[dict]:
        """
        Analyze medical document to suggest relevant chapters.
        
        In a real implementation, this would use an LLM.
        For demo purposes, we'll use keyword matching.
        """
        chapters = get_chapters_for_selection(self.navigator)
        
        # Simple keyword-based chapter suggestion (replace with LLM call)
        chapter_keywords = {
            '1': ['infectious', 'infection', 'sepsis', 'pneumonia', 'covid'],
            '2': ['cancer', 'tumor', 'malignant', 'neoplasm', 'carcinoma'],
            '3': ['anemia', 'blood', 'bleeding', 'coagulation', 'hemoglobin'],
            '4': ['diabetes', 'thyroid', 'metabolic', 'nutrition', 'obesity'],
            '5': ['depression', 'anxiety', 'mental', 'psychosis', 'dementia'],
            '6': ['stroke', 'seizure', 'neurological', 'paralysis', 'headache'],
            '7': ['vision', 'hearing', 'eye', 'ear', 'blind', 'deaf'],
            '8': ['heart', 'cardiac', 'hypertension', 'coronary', 'arrhythmia'],
            '9': ['respiratory', 'lung', 'asthma', 'copd', 'breathing'],
            '10': ['digestive', 'stomach', 'liver', 'bowel', 'gastric'],
            '11': ['skin', 'rash', 'dermatitis', 'ulcer', 'wound'],
            '12': ['joint', 'arthritis', 'bone', 'fracture', 'muscle'],
            '13': ['genitourinary', 'kidney', 'bladder', 'reproductive'],
            '14': ['pregnancy', 'birth', 'delivery', 'prenatal', 'obstetric'],
            '15': ['newborn', 'perinatal', 'infant', 'neonatal'],
            '16': ['congenital', 'birth defect', 'chromosomal', 'genetic'],
            '17': ['abnormal', 'laboratory', 'finding', 'symptom'],
            '18': ['symptom', 'sign', 'abnormal', 'complaint'],
            '19': ['injury', 'trauma', 'accident', 'poisoning', 'burn'],
            '20': ['external', 'cause', 'accident', 'violence'],
            '21': ['health status', 'screening', 'examination', 'history']
        }
        
        document_lower = medical_document.lower()
        suggested_chapters = []
        
        for chapter in chapters:
            chapter_code = chapter['code']
            if chapter_code in chapter_keywords:
                keywords = chapter_keywords[chapter_code]
                if any(keyword in document_lower for keyword in keywords):
                    chapter['relevance_score'] = sum(1 for kw in keywords if kw in document_lower)
                    suggested_chapters.append(chapter)
        
        # Sort by relevance score
        suggested_chapters.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return suggested_chapters[:5]  # Top 5 suggestions
    
    def get_llm_context(self, current_code: str, medical_document: str) -> str:
        """
        Build context string for LLM decision making.
        """
        details = get_node_details(self.navigator, current_code)
        
        context = f"""
Medical Document: {medical_document}

Current Position in ICD Tree:
- Code: {details['current_node']['code']}
- Name: {details['current_node']['name']}
- Type: {details['current_node']['element_type']}

Path to Current Position:
{' → '.join(details['path_to_root'])}

Ancestor Context:
"""
        
        for ancestor in details['ancestors']:
            context += f"- {ancestor['code']}: {ancestor['name']} ({ancestor['element_type']})\n"
        
        if details['children']:
            context += f"\nAvailable Child Nodes ({len(details['children'])}):\n"
            for child in details['children']:
                context += f"- {child['code']}: {child['name']}\n"
        else:
            context += "\n✓ This is a leaf node (most specific available)\n"
        
        return context.strip()
    
    def simulate_llm_decision(self, context: str, medical_document: str) -> dict:
        """
        Simulate LLM decision making for traversal.
        
        In a real implementation, this would call an actual LLM.
        For demo purposes, we'll use rule-based logic.
        """
        lines = context.split('\n')
        children_section = False
        available_children = []
        current_code = ""
        is_leaf = "leaf node" in context.lower()
        
        # Parse context to extract information
        for line in lines:
            if line.startswith("- Code:"):
                current_code = line.split(": ")[1]
            elif line.startswith("Available Child Nodes"):
                children_section = True
            elif children_section and line.startswith("- "):
                parts = line[2:].split(": ", 1)
                if len(parts) == 2:
                    child_code = parts[0]
                    child_name = parts[1]
                    available_children.append({'code': child_code, 'name': child_name})
        
        # Simple decision logic (replace with actual LLM)
        medical_lower = medical_document.lower()
        
        # If we're at a leaf or the condition seems well-matched, consider stopping
        if is_leaf or len(available_children) == 0:
            return {
                'decision': 'stop',
                'reasoning': f'Code {current_code} is sufficiently specific for the documented condition',
                'selected_codes': [current_code]
            }
        
        # Look for specific matches in children
        selected_children = []
        
        for child in available_children[:3]:  # Limit to first 3 for demo
            child_name_lower = child['name'].lower()
            
            # Simple keyword matching logic
            if any(word in child_name_lower for word in medical_lower.split() if len(word) > 3):
                selected_children.append(child['code'])
        
        if selected_children:
            return {
                'decision': 'continue',
                'reasoning': f'Found specific matches in children: {", ".join(selected_children)}',
                'selected_codes': selected_children
            }
        elif len(available_children) <= 3:
            # If few children and no specific matches, might be getting too specific
            return {
                'decision': 'stop',
                'reasoning': f'Current code {current_code} appropriate - children may be too specific',
                'selected_codes': [current_code]
            }
        else:
            # Select first child as default
            return {
                'decision': 'continue', 
                'reasoning': f'Continuing traversal to {available_children[0]["code"]} for more specificity',
                'selected_codes': [available_children[0]['code']]
            }
    
    def traverse_for_document(self, medical_document: str, max_steps: int = 10) -> dict:
        """
        Main traversal method for coding a medical document.
        """
        print(f"🔍 Analyzing document: {medical_document[:100]}...\n")
        
        # Step 1: Chapter selection
        print("Step 1: Chapter Selection")
        print("-" * 40)
        suggested_chapters = self.analyze_document_for_chapters(medical_document)
        
        if not suggested_chapters:
            return {'error': 'No relevant chapters found'}
        
        print("Suggested chapters:")
        for chapter in suggested_chapters:
            score = chapter.get('relevance_score', 0)
            print(f"   Chapter {chapter['code']}: {chapter['name']} (score: {score})")
        
        # Select top chapter (in real implementation, LLM would choose)
        selected_chapter = suggested_chapters[0]['code']
        print(f"\n🎯 Selected: Chapter {selected_chapter}")
        
        # Step 2: Hierarchical traversal
        current_codes = [selected_chapter]
        final_codes = []
        step = 2
        
        while current_codes and step <= max_steps:
            print(f"\nStep {step}: Traversal Analysis")
            print("-" * 40)
            
            new_codes = []
            
            for code in current_codes:
                print(f"\n📍 Analyzing code: {code}")
                
                # Get LLM context
                context = self.get_llm_context(code, medical_document)
                
                # Simulate LLM decision
                decision = self.simulate_llm_decision(context, medical_document)
                
                print(f"   Decision: {decision['decision']}")
                print(f"   Reasoning: {decision['reasoning']}")
                print(f"   Selected: {decision['selected_codes']}")
                
                if decision['decision'] == 'stop':
                    final_codes.extend(decision['selected_codes'])
                else:
                    new_codes.extend(decision['selected_codes'])
                
                # Store history
                self.traversal_history.append({
                    'step': step,
                    'current_code': code,
                    'decision': decision,
                    'context_length': len(context)
                })
            
            current_codes = new_codes
            step += 1
        
        # Final results
        if current_codes:  # Ran out of steps
            final_codes.extend(current_codes)
        
        return {
            'final_codes': final_codes,
            'traversal_steps': step - 2,
            'history': self.traversal_history,
            'document_analyzed': medical_document
        }
    
    def get_code_details(self, codes: list[str]) -> list[dict]:
        """Get detailed information for final codes."""
        details = []
        for code in codes:
            node_details = get_node_details(self.navigator, code)
            if 'error' not in node_details:
                details.append({
                    'code': code,
                    'name': node_details['current_node']['name'],
                    'path': ' → '.join(node_details['path_to_root']),
                    'is_leaf': node_details['is_leaf']
                })
        return details


def demo_agent_workflow():
    """Demonstrate the simple agent workflow."""
    print("🤖 Simple ICD Coding Agent Demo\n")
    
    # Create agent
    agent = SimpleCodingAgent()
    
    # Example medical documents
    test_documents = [
        "Patient presents with Type 1 diabetes mellitus with diabetic nephropathy. HbA1c is elevated at 9.2%. Patient shows signs of proteinuria and decreased kidney function.",
        
        "45-year-old male presents to ED with acute myocardial infarction. ST-elevation noted on EKG. Patient has history of hypertension and hyperlipidemia.",
        
        "Newborn infant delivered at 32 weeks gestation with respiratory distress syndrome. Requires mechanical ventilation and surfactant therapy."
    ]
    
    # Process each document
    for i, document in enumerate(test_documents, 1):
        print(f"\n{'='*60}")
        print(f"🔬 Test Case {i}")
        print(f"{'='*60}")
        
        # Run traversal
        result = agent.traverse_for_document(document, max_steps=8)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        # Display results
        print(f"\n📋 Final Results:")
        print(f"   Steps taken: {result['traversal_steps']}")
        print(f"   Codes found: {len(result['final_codes'])}")
        
        # Get detailed information
        code_details = agent.get_code_details(result['final_codes'])
        
        print(f"\n🎯 Final Codes:")
        for detail in code_details:
            leaf_indicator = "🍃" if detail['is_leaf'] else "🌿"
            print(f"   {leaf_indicator} {detail['code']}: {detail['name']}")
            print(f"      Path: {detail['path']}")
        
        # Reset for next test
        agent.current_codes = []
        agent.traversal_history = []


if __name__ == "__main__":
    demo_agent_workflow()