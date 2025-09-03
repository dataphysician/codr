"""
Simple Agent Actions for ICD-10 Medical Document Processing
===========================================================

This module provides LLM-based agent actions for processing medical documents
and navigating the ICD-10 tree structure using the existing llm_client API.
"""

import json
import os
from icd_tree import ICDTreeNavigator, create_simple_navigator, get_node_details, get_chapters_for_selection
from llm_client import completion, get_response_text




def create_medical_document_message(medical_document: str) -> dict[str, str]:
    """
    Create a message object for the medical document.
    
    Args:
        medical_document: The medical document text
        
    Returns:
        Message object with role and content
    """
    return {"role": "user", "content": medical_document}


def create_node_context_message(
    navigator: ICDTreeNavigator, 
    code: str | None = None, 
    include_children: bool = True
) -> dict[str, str]:
    """
    Create a concatenated node context message from Full Path, Ancestor Context, 
    Current Position, and Children.
    
    Args:
        navigator: ICD tree navigator instance
        code: Current node code (None for initial chapter selection)
        include_children: Whether to include children in context
        
    Returns:
        Message object with concatenated node context
    """
    context_parts = []
    
    if code is None:
        # Initial chapter selection
        context_parts.append("NAVIGATION CONTEXT:")
        context_parts.append("Position: ROOT - Select initial chapter")
        
        chapters = get_chapters_for_selection(navigator)
        context_parts.append(f"Available Chapters ({len(chapters)}):")
        for chapter in chapters:
            context_parts.append(f"  {chapter['code']}: {chapter['name']}")
            
    else:
        # Navigate to specific node
        node_details = get_node_details(navigator, code)
        
        if 'error' in node_details:
            context_parts.append(f"ERROR: {node_details['error']}")
        else:
            context_parts.append("NAVIGATION CONTEXT:")
            
            # Full Path
            if node_details.get('path_to_root'):
                context_parts.append(f"Full Path: {' → '.join(node_details['path_to_root'])}")
            
            # Ancestor Context
            if node_details.get('ancestors'):
                context_parts.append("Ancestor Context:")
                for ancestor in node_details['ancestors']:
                    context_parts.append(f"  {ancestor['code']}: {ancestor['name']} ({ancestor['element_type']})")
            
            # Current Position
            current = node_details['current_node']
            context_parts.append(f"Current Position: {current['code']} - {current['name']} ({current['element_type']})")
            
            # Children (if requested and available)
            if include_children and node_details.get('children'):
                context_parts.append(f"Available Children ({len(node_details['children'])}):")
                for child in node_details['children']:
                    context_parts.append(f"  {child['code']}: {child['name']} ({child['element_type']})")
            elif include_children:
                context_parts.append("No children available (leaf node)")
    
    concatenated_context = "\n".join(context_parts)
    return {"role": "user", "content": concatenated_context}


def create_agent_prompt() -> dict[str, str]:
    """
    Create the system prompt for the ICD coding agent.
    
    Returns:
        System message with agent instructions
    """
    prompt = """You are an expert ICD-10-CM medical coding assistant. Your task is to analyze medical documents and navigate the ICD-10 hierarchy to find appropriate diagnosis codes.

The ICD-10 hierarchy follows this structure:
1. CHAPTERS (numbered 1, 2, 3, etc.) - broad disease categories
2. BLOCKS (like E10-E16) - groups of related conditions  
3. CATEGORIES (like E10) - specific disease categories
4. SUBCATEGORIES (like E10.2) - more specific conditions
5. CODES (like E10.21) - precise diagnosis codes

Given a medical document and navigation context, you should:

1. Analyze the medical conditions described in the document
2. Consider the current position in the ICD-10 hierarchy  
3. Select the most appropriate child node(s) to continue navigation toward the most specific applicable codes
4. Return your selections in the specified JSON format

CRITICAL NAVIGATION CONSTRAINTS:
- You CANNOT jump to codes or skip levels in the hierarchy
- You MUST ONLY select from the "Available Children" provided in the navigation context
- You CANNOT select codes outside of the explicitly listed children options
- If no child adequately matches the medical documentation due to ambiguity or insufficient detail, you may choose to STOP navigation
- Navigate systematically: Chapter → Block → Category → Subcategory → Final Code

IMPORTANT NAVIGATION RULES:
- For chapters, use the NUMERIC code (1, 2, 3, etc.) NOT the range codes
- For blocks, categories, and subcategories, use the EXACT code shown (E10-E16, E10, E10.2, etc.)
- Choose the path that leads to the most clinically accurate and specific codes
- Only select codes that are explicitly listed as available children

RESPONSE FORMAT:
Return a JSON array of objects, where each object contains:
- "node": the exact code of the child node to select (MUST be from available children list) OR "STOP" to halt navigation
- "description": brief explanation of why this node matches the medical document OR why navigation should stop
- "citation": specific quote or reference from the medical document that supports this selection (MANDATORY)
  * For regular navigation: quote from medical document
  * For STOP navigation: return the available children with code and description, newline separated per child

Example responses:
For chapter selection: [{"node": "4", "description": "Endocrine diseases chapter - patient has diabetes", "citation": "Patient presents with Type 1 diabetes mellitus"}]
For block selection: [{"node": "E10-E16", "description": "Diabetes mellitus block matches condition", "citation": "Type 1 diabetes mellitus with diabetic nephropathy"}] 
For category: [{"node": "E10", "description": "Type 1 diabetes mellitus matches patient's condition", "citation": "Type 1 diabetes mellitus"}]
To stop navigation: [{"node": "STOP", "description": "Medical documentation lacks sufficient detail to distinguish between subcategories", "citation": "E10.0: Type 1 diabetes mellitus with coma\nE10.1: Type 1 diabetes mellitus with ketoacidosis\nE10.2: Type 1 diabetes mellitus with kidney complications\nE10.3: Type 1 diabetes mellitus with ophthalmic complications"}]

Focus on accuracy and clinical relevance. Only navigate to children that are explicitly provided in the context."""

    return {"role": "system", "content": prompt}


def process_medical_document_with_llm(
    medical_document: str,
    navigator: ICDTreeNavigator,
    current_code: str | None = None,
    provider: str = "openai",
    **llm_kwargs
) -> list[dict[str, str]]:
    """
    Process a medical document with LLM to get next navigation steps.
    
    Args:
        medical_document: The medical document text
        navigator: ICD tree navigator instance  
        current_code: Current position code (None for initial selection)
        provider: LLM provider to use ("openai", "anthropic", "gemini", "cerebras")
        **llm_kwargs: Additional parameters for the LLM call
        
    Returns:
        List of node selections with format [{"node": code, "description": desc}, ...]
    """
    # Create message objects
    system_prompt = create_agent_prompt()
    medical_msg = create_medical_document_message(medical_document)
    context_msg = create_node_context_message(navigator, current_code)
    
    # Combine messages
    messages = [system_prompt, medical_msg, context_msg]
    
    # Call LLM
    response = completion(messages, provider=provider, **llm_kwargs)
    response_text = get_response_text(response)
    
    # Parse JSON response
    try:
        # Clean up response text and extract JSON
        response_text = response_text.strip()
        
        # Handle cases where LLM includes markdown formatting
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            response_text = response_text.replace('```', '').strip()
        
        # Extract just the JSON array part if there's extra text
        import re
        json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
            
        selections = json.loads(response_text)
        
        # Validate format
        if not isinstance(selections, list):
            raise ValueError("Response must be a list")
            
        for selection in selections:
            if not isinstance(selection, dict):
                raise ValueError("Each selection must be a dictionary")
            if 'node' not in selection or 'description' not in selection:
                raise ValueError("Each selection must have 'node' and 'description' keys")
            # Citation is optional but preferred
            if 'citation' not in selection:
                selection['citation'] = "No specific citation provided"
                
        return selections
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"   ❌ JSON PARSING ERROR: {e}")
        print(f"   📄 Raw LLM Response: {response_text}")
        return []


def navigate_with_agent(
    medical_document: str,
    max_steps: int = 10,
    provider: str = "openai",
    navigator: ICDTreeNavigator | None = None,
    **llm_kwargs
) -> dict[str, any]:
    """
    Perform complete navigation using LLM agent decisions.
    
    Args:
        medical_document: The medical document text
        max_steps: Maximum navigation steps to prevent infinite loops
        provider: LLM provider to use
        navigator: Optional pre-created navigator (will create if None)
        **llm_kwargs: Additional parameters for LLM calls
        
    Returns:
        Dictionary with navigation results and path taken
    """
    if navigator is None:
        navigator = create_simple_navigator()
    
    navigation_path = []
    current_code = None
    hierarchy_levels = ["ROOT", "Chapter", "Block", "Category", "Subcategory", "Code"]
    
    print(f"\n🏥 MEDICAL DOCUMENT:")
    print(f"   {medical_document}")
    print(f"\n📍 NAVIGATION TRAJECTORY:")
    print(f"   Target: Chapter → Block → Category → Subcategory → Final Code")
    
    for step in range(max_steps):
        # Determine current hierarchy level
        if current_code is None:
            current_level = "ROOT"
            next_level = "Chapter"
        else:
            node_details = get_node_details(navigator, current_code)
            if 'error' not in node_details:
                element_type = node_details['current_node']['element_type']
                if element_type == 'chapter':
                    current_level = "Chapter"
                    next_level = "Block"
                elif element_type == 'block':
                    current_level = "Block"  
                    next_level = "Category"
                elif element_type == 'diagnosis':
                    # Determine if this is category, subcategory, or final code
                    code = current_code
                    if '.' not in code:
                        current_level = "Category"
                        next_level = "Subcategory"
                    elif code.count('.') == 1:
                        current_level = "Subcategory"
                        next_level = "Final Code"
                    else:
                        current_level = "Final Code"
                        next_level = "Complete"
                else:
                    current_level = "Unknown"
                    next_level = "Next"
        
        print(f"\n🤖 STEP {step + 1}: {current_level} → {next_level}")
        print(f"   Current Position: {'ROOT (Start)' if current_code is None else current_code}")
        
        # Get LLM decision
        selections = process_medical_document_with_llm(
            medical_document, navigator, current_code, provider, **llm_kwargs
        )
        
        if not selections:
            print(f"   ❌ NAVIGATION HALTED: No valid selections from LLM")
            print(f"   📋 Reason: LLM response could not be parsed or contained no valid selections")
            
            # Show available children and clinical guidance
            if current_code:
                children_details = get_node_details(navigator, current_code)
                if 'children' in children_details and children_details['children']:
                    children_list = []
                    for child in children_details['children']:
                        children_list.append(f"{child['code']}: {child['name']}")
                    print(f"   📋 Available Children:\n{chr(10).join(children_list)}")
                    
                    # Provide clinical guidance based on current position
                    print(f"\n   🩺 CLINICAL GUIDANCE:")
                    print(f"   To proceed with coding, additional clinical information may be needed:")
                    
                    current_name = children_details['current_node']['name']
                    if "fracture" in current_name.lower():
                        print(f"   • Is this an initial encounter, subsequent encounter, or sequela?")
                        print(f"   • Is the fracture displaced or non-displaced?")
                        print(f"   • Which specific anatomical location/pole is affected?")
                        print(f"   • Is this an open or closed fracture?")
                    elif "diabetes" in current_name.lower():
                        print(f"   • What specific complications are present (retinopathy, nephropathy, etc.)?")
                        print(f"   • Is this with or without complications?")
                        print(f"   • What is the severity/stage of complications?")
                    elif "asthma" in current_name.lower() or "respiratory" in current_name.lower():
                        print(f"   • Is this mild, moderate, or severe?")
                        print(f"   • Is this intermittent or persistent?")
                        print(f"   • Are there acute exacerbations present?")
                        print(f"   • What triggers are involved (allergic, non-allergic)?")
                    elif "hypertension" in current_name.lower():
                        print(f"   • Is this primary (essential) or secondary hypertension?")
                        print(f"   • Are there target organ complications?")
                        print(f"   • Is this benign or malignant?")
                    else:
                        print(f"   • Review clinical documentation for specific manifestations")
                        print(f"   • Identify anatomical location, severity, or encounter type")
                        print(f"   • Determine if additional qualifiers apply")
                        
                    print(f"   • Consult clinical notes and diagnostic reports for specificity")
            break
            
        print(f"   🧠 LLM Decision: {len(selections)} option(s)")
        for selection in selections:
            print(f"      → {selection['node']}: {selection['description']}")
        
        # Take the first selection for navigation (could be extended to handle multiple paths)
        selected = selections[0]
        next_code = selected['node']
        
        # Check if LLM decided to stop navigation
        if next_code == "STOP":
            print(f"   🛑 NAVIGATION HALTED: LLM decided to stop")
            print(f"   📋 Reason: {selected['description']}")
            if 'citation' in selected and selected['citation']:
                print(f"   📄 Available Options: \"{selected['citation']}\"")
            
            print(f"\n   🩺 CLINICAL GUIDANCE:")
            print(f"   The LLM determined that the medical documentation is insufficient")
            print(f"   to distinguish between the available subcategories. Consider:")
            print(f"   • Reviewing additional clinical notes or diagnostic reports")
            print(f"   • Obtaining more specific details about the condition")
            print(f"   • Clarifying anatomical location, severity, or encounter type")
            print(f"   • Using 'unspecified' codes when clinical details are truly unavailable")
            print(f"   💡 This indicates appropriate clinical judgment to avoid miscoding")
            break
        
        # Verify the selection is valid
        node_details = get_node_details(navigator, next_code)
        if 'error' in node_details:
            print(f"   ❌ NAVIGATION HALTED: Invalid selection")
            print(f"   📋 Selected Code: {next_code}")
            print(f"   🚫 Error: {node_details['error']}")
            print(f"   💡 This indicates the LLM selected a code not available in the current context")
            break
            
        # Display the navigation step
        node_name = node_details['current_node']['name']
        node_type = node_details['current_node']['element_type']
        print(f"   ✅ Selected: {next_code} ({node_type.title()}) - {node_name}")
        
        # Record the navigation step
        navigation_path.append({
            'step': step + 1,
            'from_code': current_code,
            'selected_code': next_code,
            'selection_reason': selected['description'],
            'node_details': node_details,
            'hierarchy_level': next_level
        })
        
        current_code = next_code
        
        # Check if we've reached a leaf node or maximum specificity
        # For ICD-10, final codes are typically diagnosis elements without children
        # Block codes (like S60-S69) are not final codes even if they're 7 characters
        is_final_code = (
            node_details['is_leaf'] or 
            len(node_details.get('children', [])) == 0 or
            (len(current_code) >= 7 and node_details['current_node']['element_type'] == 'diagnosis')
        )
        
        if is_final_code:
            print(f"   🎯 NAVIGATION COMPLETE: Reached final diagnosis")
            print(f"   📋 Final Code: {current_code}")
            if len(node_details.get('children', [])) == 0:
                print(f"   💡 Reason: No further subcategories available (leaf node)")
            elif len(current_code) >= 7 and node_details['current_node']['element_type'] == 'diagnosis':
                print(f"   💡 Reason: Maximum ICD-10 specificity reached (7+ characters)")
            else:
                print(f"   💡 Reason: Node marked as leaf in hierarchy")
            break
    
    # Display complete trajectory summary
    print(f"\n📊 COMPLETE TRAJECTORY SUMMARY:")
    trajectory_display = ["ROOT"]
    for step in navigation_path:
        trajectory_display.append(step['selected_code'])
    
    print(f"   Path: {' → '.join(trajectory_display)}")
    print(f"   Levels: {' → '.join([step['hierarchy_level'] for step in navigation_path])}")
    
    # Prepare final results
    final_node = get_node_details(navigator, current_code) if current_code else None
    
    return {
        'medical_document': medical_document,
        'navigation_path': navigation_path,
        'final_code': current_code,
        'final_diagnosis': final_node['current_node']['name'] if final_node and 'current_node' in final_node else None,
        'total_steps': len(navigation_path),
        'completed': final_node['is_leaf'] if final_node else False,
        'all_selections': [step['selected_code'] for step in navigation_path],
        'trajectory_summary': ' → '.join(trajectory_display)
    }


# Convenience functions for specific use cases

def quick_code_medical_document(
    medical_document: str, 
    provider: str = "openai",
    **llm_kwargs
) -> str | None:
    """
    Quick function to get a single ICD code for a medical document.
    
    Args:
        medical_document: Medical document text
        provider: LLM provider to use
        **llm_kwargs: Additional LLM parameters
        
    Returns:
        Final ICD code or None if navigation failed
    """
    result = navigate_with_agent(medical_document, provider=provider, **llm_kwargs)
    return result.get('final_code')


def batch_process_documents(
    documents: list[str],
    provider: str = "openai", 
    **llm_kwargs
) -> list[dict[str, any]]:
    """
    Process multiple medical documents in batch.
    
    Args:
        documents: List of medical document texts
        provider: LLM provider to use
        **llm_kwargs: Additional LLM parameters
        
    Returns:
        List of navigation results for each document
    """
    navigator = create_simple_navigator()  # Reuse navigator for efficiency
    results = []
    
    for i, doc in enumerate(documents):
        print(f"\n{'='*50}")
        print(f"Processing Document {i+1}/{len(documents)}")
        print(f"{'='*50}")
        
        result = navigate_with_agent(doc, navigator=navigator, provider=provider, **llm_kwargs)
        results.append(result)
        
    return results


if __name__ == "__main__":
    # Demo usage
    # medical_document = """Patient presents with Type 1 diabetes mellitus with diabetic nephropathy. 
    # HbA1c is elevated at 9.2%. Patient shows proteinuria and decreased kidney function with GFR at 14 
    # which clearly meets CKD-4 criteria."""
    medical_document ="""
Chief Complaint:
“Pain and swelling in my left wrist after a fall.”

History of Present Illness:
Patient reports falling onto an outstretched left hand earlier today while [walking / sports activity]. Immediately noted pain and swelling around the wrist. Difficulty moving the hand due to pain. No loss of consciousness. No open wounds seen.

Past Medical History:
Denies prior fractures of this wrist. No significant chronic illnesses reported.

Medications:
None regularly. Took acetaminophen at home.

Allergies:
None known.

Family/Social History:
Lives with family, right-hand dominant. No tobacco, alcohol, or drug use reported.

Review of Systems (focused):

Musculoskeletal: Pain, swelling, limited movement of left wrist.

Neurological: No numbness or tingling reported.

General: No fever or systemic complaints.

Physical Exam (as observed):

Left wrist visibly swollen, tender over distal radius.

Limited range of motion due to pain.

No obvious deformity noted.

Sensation intact in fingers, capillary refill <2 seconds.

Imaging:
X-ray of the left wrist reportedly shows a fracture near the distal radius, likely the scaphoid. No mention of displacement or open wound.

Assessment:
Patient with wrist injury after fall, x-ray consistent with fracture of the distal radius, closed type. No evidence of neurovascular compromise.

Plan (to be confirmed by attending):

Immobilization with splint.

Pain management.

Orthopedic consultation for definitive management.

Instructions given to keep arm elevated, ice as tolerated.
"""
    medical_document2 = """
Chief Complaint:
“Cough and shortness of breath on and off for a while.”

History of Present Illness:
Patient reports having had frequent respiratory infections for over 2 years. Episodes described as recurrent cough, wheezing, and chest tightness, usually worse with cold weather or when exposed to smoke/dust. States that symptoms have been ongoing for years, with gradual persistence of daily cough. Prior spirometry reportedly showed reduced FEV1 with partial reversibility. Findings consistent with airway obstruction noted over multiple visits. No recent acute flare-up reported. Currently denies fever or chills.

Past Medical History:
Known history of asthma since childhood. Patient recalls multiple episodes of “bronchitis” in the past, treated with antibiotics and inhalers.
No recent hospital admissions reported.

Medications:
Patient states using an inhaler “as needed.” Specific name not recalled. Reports occasional use of cough syrup in the past.

Allergies:
Denies drug allergies.

Family/Social History:
Father with history of asthma.
Non-smoker currently, but exposed to secondhand smoke at home.
Works indoors, reports dust exposure at workplace.

Review of Systems (limited):
Respiratory: Chronic cough, intermittent wheeze, shortness of breath on exertion.
General: No recent weight loss, no fever.
Cardiovascular: Denies chest pain.
Physical Exam (as observed):
Patient speaking in full sentences, no acute distress.
Mild wheeze noted on expiration.
No cyanosis.

Assessment:
Patient with history of asthma and long-standing episodes of recurrent respiratory infections likely obstructive in nature, currently presenting with persistent cough and wheezing.No signs of acute exacerbation at this time.
Plan (to be confirmed by attending):

Continue current inhaler use as prescribed.
Monitor for worsening symptoms (fever, increased shortness of breath).
Attending to evaluate for possible need of maintenance therapy.
Patient advised to avoid smoke exposure.
"""

    print("="*80)
    print("🏥 SIMPLE AGENT ACTIONS DEMO - ICD-10 Navigation")
    print("="*80)
    
    # Navigate with agent
    result = navigate_with_agent(medical_document, provider="cerebras", model="qwen-3-235b-a22b-instruct-2507")#"llama-4-maverick-17b-128e-instruct")
    
    print(f"\n" + "="*80)
    print(f"📋 FINAL RESULTS")
    print("="*80)
    print(f"🎯 Final Code: {result['final_code']}")
    print(f"📝 Final Diagnosis: {result['final_diagnosis']}")
    print(f"📈 Navigation Steps: {result['total_steps']}")
    print(f"🗺️  Complete Trajectory: {result.get('trajectory_summary', 'N/A')}")
    print(f"✅ Navigation Complete: {result['completed']}")
    
    if not result['completed']:
        print(f"\n⚠️  NAVIGATION STATUS:")
        print(f"   Navigation was halted before reaching a final diagnosis code.")
        print(f"   Review the step-by-step output above for specific halt reasons.")
    
    # Display step-by-step breakdown
    if result['navigation_path']:
        print(f"\n📚 STEP-BY-STEP BREAKDOWN:")
        for step in result['navigation_path']:
            print(f"   Step {step['step']}: {step.get('hierarchy_level', 'Unknown')} → {step['selected_code']}")
            print(f"      Reason: {step['selection_reason']}")
    
    print(f"\n💡 This demo shows the systematic navigation:")
    print(f"   ROOT → Chapter → Block → Category → Subcategory → Final Code")
    print(f"   Each step uses LLM reasoning to select the most appropriate path")