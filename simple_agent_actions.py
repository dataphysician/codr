"""
Simple Agent Actions for ICD-10 Medical Document Processing
===========================================================

This module provides LLM-based agent actions for processing medical documents
and navigating the ICD-10 tree structure using the existing llm_client API.
"""

import json
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
        provider: LLM provider to use ("openai", "anthropic", "gemini", "keywell")
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
        print(f"   ❌ Failed to parse LLM response: {e}")
        print(f"   Raw response: {response_text}")
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
            print(f"   ❌ No valid selections from LLM - stopping navigation")
            break
            
        print(f"   🧠 LLM Decision: {len(selections)} option(s)")
        for selection in selections:
            print(f"      → {selection['node']}: {selection['description']}")
        
        # Take the first selection for navigation (could be extended to handle multiple paths)
        selected = selections[0]
        next_code = selected['node']
        
        # Verify the selection is valid
        node_details = get_node_details(navigator, next_code)
        if 'error' in node_details:
            print(f"   ❌ Invalid selection {next_code}: {node_details['error']}")
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
        
        # Check if we've reached a leaf node or specific enough diagnosis
        if node_details['is_leaf'] or len(node_details.get('children', [])) == 0:
            print(f"   🎯 Reached final diagnosis: {current_code}")
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
    medical_document = """Patient presents with Type 1 diabetes mellitus with diabetic nephropathy. 
    HbA1c is elevated at 9.2%. Patient shows proteinuria and decreased kidney function with GFR at 14 
    which clearly meets CKD-4 criteria."""
    
    print("="*80)
    print("🏥 SIMPLE AGENT ACTIONS DEMO - ICD-10 Navigation")
    print("="*80)
    
    # Navigate with agent
    result = navigate_with_agent(medical_document, provider="openai", model="gpt-5-mini")
    
    print(f"\n" + "="*80)
    print(f"📋 FINAL RESULTS")
    print("="*80)
    print(f"🎯 Final Code: {result['final_code']}")
    print(f"📝 Final Diagnosis: {result['final_diagnosis']}")
    print(f"📈 Navigation Steps: {result['total_steps']}")
    print(f"🗺️  Complete Trajectory: {result.get('trajectory_summary', 'N/A')}")
    print(f"✅ Navigation Complete: {result['completed']}")
    
    # Display step-by-step breakdown
    if result['navigation_path']:
        print(f"\n📚 STEP-BY-STEP BREAKDOWN:")
        for step in result['navigation_path']:
            print(f"   Step {step['step']}: {step.get('hierarchy_level', 'Unknown')} → {step['selected_code']}")
            print(f"      Reason: {step['selection_reason']}")
    
    print(f"\n💡 This demo shows the systematic navigation:")
    print(f"   ROOT → Chapter → Block → Category → Subcategory → Final Code")
    print(f"   Each step uses LLM reasoning to select the most appropriate path")