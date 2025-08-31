"""
ICD-10-CM Tree Navigation Interface
==================================

Standalone tree interface for hackathon demos and agentic workflows.
Contains ICD tree parsing, navigation, and agent-friendly context methods.
"""

import xml.etree.ElementTree as ET
import time
from anytree import Node


# Core ICD Tree Classes and Functions
# ====================================

def parse_icd10_file(file_path: str = "icd10cm_tabular_2026.txt"):
    """Parses the ICD-10-CM tabular XML file and builds a tree structure."""
    
    # Parse the XML file
    tree = ET.parse(file_path)
    root_element = tree.getroot()
    
    # Create the root node for our tree
    root = Node("ICD-10-CM Root", code="ROOT", notes={}, element_type="root")
    
    def parse_code_description(note_text: str) -> list[tuple[str, str]]:
        """Parse code and description from note text, returning (code, description) tuples."""
        import re
        
        # Find all ICD codes in parentheses
        code_pattern = r'\(([A-Z][0-9][0-9](?:\.[0-9A-X-]+)?(?:-[A-Z][0-9][0-9](?:\.[0-9A-X-]+)?)?(?:, ?[A-Z][0-9][0-9](?:\.[0-9A-X-]+)?)*(?:\.?-)?)\)'
        
        matches = re.findall(code_pattern, note_text)
        
        if matches:
            # Split multiple codes by comma and clean up
            codes = []
            for match in matches:
                # Handle comma-separated codes like "J09.X3, J10.2, J11.2"
                for code in match.split(','):
                    code = code.strip()
                    if code:
                        codes.append(code)
            
            # Extract description (text before the first parenthesis)
            desc_match = re.match(r'^(.*?)\s*\([A-Z][0-9]', note_text)
            description = desc_match.group(1).strip() if desc_match else note_text
            
            # Return list of (code, description) tuples
            return [(code, description) for code in codes]
        else:
            # No codes found, return the whole text as description with empty code
            return [("", note_text)]

    def extract_notes(element) -> dict[str, list[tuple[str, str]]]:
        """Extract notes from various note elements like includes, excludes1, etc."""
        notes = {}
        
        # Check for different types of notes
        note_types = ['includes', 'excludes1', 'excludes2', 'useAdditionalCode', 'codeFirst', 'codeAlso', 'sevenChrNote']
        
        for note_type in note_types:
            note_elements = element.findall(note_type)
            if note_elements:
                note_list = []
                for note_elem in note_elements:
                    for note in note_elem.findall('note'):
                        if note.text:
                            parsed_notes = parse_code_description(note.text.strip())
                            note_list.extend(parsed_notes)
                if note_list:
                    notes[note_type] = note_list
        
        # Also check for inclusionTerm
        inclusion_terms = element.findall('inclusionTerm')
        if inclusion_terms:
            inclusion_list = []
            for term_elem in inclusion_terms:
                for note in term_elem.findall('note'):
                    if note.text:
                        parsed_notes = parse_code_description(note.text.strip())
                        inclusion_list.extend(parsed_notes)
            if inclusion_list:
                notes['inclusionTerm'] = inclusion_list
                
        return notes
    
    def create_diag_nodes(diag_element, parent_node):
        """Recursively create nodes for diagnosis elements."""
        name_elem = diag_element.find('name')
        desc_elem = diag_element.find('desc')
        
        if name_elem is not None and desc_elem is not None:
            code = name_elem.text.strip() if name_elem.text else ""
            description = desc_elem.text.strip() if desc_elem.text else ""
            
            # Extract notes for this diagnosis
            notes = extract_notes(diag_element)
            
            # Create the node
            diag_node = Node(
                name=description,
                parent=parent_node,
                code=code,
                notes=notes,
                element_type="diagnosis"
            )
            
            # Process any nested diagnosis elements
            nested_diags = diag_element.findall('diag')
            for nested_diag in nested_diags:
                create_diag_nodes(nested_diag, diag_node)
    
    def create_section_nodes(section_element, parent_node):
        """Create nodes for section elements (blocks)."""
        desc_elem = section_element.find('desc')
        
        if desc_elem is not None:
            # Extract range from description (like "A00-A09")
            section_desc = desc_elem.text.strip() if desc_elem.text else ""
            
            # Try to extract the range pattern
            import re
            range_match = re.match(r'([A-Z][0-9][0-9](?:\.[0-9A-X-]+)?-[A-Z][0-9][0-9](?:\.[0-9A-X-]+)?)', section_desc)
            
            if range_match:
                section_code = range_match.group(1)
            else:
                # Fallback to section ID if no range found
                section_id = section_element.get('id', '')
                section_code = section_id if section_id else section_desc[:10]
            
            # Extract notes for this section
            notes = extract_notes(section_element)
            
            # Create the section node
            section_node = Node(
                name=section_desc,
                parent=parent_node,
                code=section_code,
                notes=notes,
                element_type="block"
            )
            
            # Process diagnosis elements within this section
            diag_elements = section_element.findall('diag')
            for diag_element in diag_elements:
                create_diag_nodes(diag_element, section_node)
    
    # Process the XML structure
    for chapter_element in root_element.findall('chapter'):
        chapter_desc = chapter_element.find('desc')
        if chapter_desc is not None:
            chapter_name = chapter_desc.text.strip() if chapter_desc.text else ""
            chapter_num = chapter_element.get('num', '')
            
            # Extract notes for this chapter
            notes = extract_notes(chapter_element)
            
            # Create chapter node
            chapter_node = Node(
                name=chapter_name,
                parent=root,
                code=chapter_num,
                notes=notes,
                element_type="chapter"
            )
            
            # Process sections within the chapter
            section_elements = chapter_element.findall('section')
            for section_element in section_elements:
                create_section_nodes(section_element, chapter_node)
    
    return root


class ICDTreeNavigator:
    """Efficient navigation utilities for the ICD tree structure."""
    
    def __init__(self, root):
        self.root = root
        self._build_indexes()
    
    def _build_indexes(self):
        """Build indexes for O(1) lookups by code."""
        from anytree import PreOrderIter
        
        self.code_to_node = {}
        self.chapters = {}
        self.blocks = {}
        self.diagnoses = {}
        
        for node in PreOrderIter(self.root):
            if hasattr(node, 'code') and node.code:
                self.code_to_node[node.code] = node
                
                element_type = getattr(node, 'element_type', None)
                if element_type == 'chapter':
                    self.chapters[node.code] = node
                elif element_type == 'block':
                    self.blocks[node.code] = node
                elif element_type == 'diagnosis':
                    self.diagnoses[node.code] = node
    
    def find_by_code(self, code: str):
        """O(1) lookup by ICD code."""
        return self.code_to_node.get(code)
    
    def get_path_to_code(self, code: str) -> list[str] | None:
        """Get full path from root to specified code."""
        node = self.find_by_code(code)
        if node:
            return [ancestor.code for ancestor in node.path]
        return None
    
    def search_by_name(self, search_term: str, max_results: int = 100) -> list:
        """Search for codes by name/description using case-insensitive substring matching."""
        results = []
        search_term = search_term.lower()
        
        for code, node in self.code_to_node.items():
            if hasattr(node, 'name') and node.name and search_term in node.name.lower():
                results.append(node)
                if len(results) >= max_results:
                    break
        
        return results


def create_navigator(file_path: str = "icd10cm_tabular_2026.txt") -> ICDTreeNavigator:
    """Create and return a fully initialized navigator."""
    root = parse_icd10_file(file_path)
    return ICDTreeNavigator(root)


# Agent-Friendly Navigation Functions
# ===================================

def _get_children_direct(node) -> list[dict[str, str]]:
    """Helper function to get children directly from a node."""
    children = []
    for child in node.children:
        child_info = {
            'code': child.code,
            'name': child.name,
            'element_type': getattr(child, 'element_type', 'unknown')
        }
        children.append(child_info)
    return children


def get_ancestors_with_context(navigator: ICDTreeNavigator, code: str) -> list[dict[str, str]]:
    """
    Get ancestor hierarchy with context for LLM prompting.
    
    Returns:
        List of ancestor nodes with code, name, and element_type for context
    """
    node = navigator.find_by_code(code)
    if not node:
        return []
    
    ancestors = []
    current = node
    
    # Walk up the tree to collect ancestors
    while current and current.parent:
        ancestor_info = {
            'code': current.parent.code,
            'name': current.parent.name,
            'element_type': getattr(current.parent, 'element_type', 'unknown')
        }
        ancestors.append(ancestor_info)
        current = current.parent
    
    return list(reversed(ancestors))  # Root to current order


def get_children_with_context(navigator: ICDTreeNavigator, code: str) -> list[dict[str, str]]:
    """
    Get child nodes with context for LLM decision making.
    
    Returns:
        List of child nodes with code, name, and element_type
    """
    node = navigator.find_by_code(code)
    
    # Handle chapter lookups by number
    if not node and code.isdigit():
        chapters = get_chapters_for_selection(navigator)
        chapter_num = int(code)
        if 1 <= chapter_num <= len(chapters):
            # Get the actual chapter node
            root = navigator.code_to_node.get('ROOT')
            if root and chapter_num <= len(root.children):
                node = root.children[chapter_num - 1]
    
    if not node:
        return []
    
    children = []
    for child in node.children:
        child_info = {
            'code': child.code,
            'name': child.name,
            'element_type': getattr(child, 'element_type', 'unknown')
        }
        children.append(child_info)
    
    return children


def get_chapters_for_selection(navigator: ICDTreeNavigator) -> list[dict[str, str]]:
    """
    Get all chapters for initial agent selection.
    
    Returns:
        List of chapter nodes for agent to choose from
    """
    chapters = []
    
    # Get root node and find its chapter children
    root = navigator.code_to_node.get('ROOT')
    if root:
        for i, child in enumerate(root.children):
            if getattr(child, 'element_type', None) == 'chapter':
                # Use chapter number as code (1-based index)
                chapter_num = str(i + 1)
                chapter_info = {
                    'code': chapter_num,
                    'name': child.name,
                    'element_type': 'chapter',
                    'actual_node_code': child.code  # Store the actual empty code for navigation
                }
                chapters.append(chapter_info)
    
    return chapters


def get_node_details(navigator: ICDTreeNavigator, code: str) -> dict[str, any]:
    """
    Get comprehensive node details for agent context.
    
    Returns:
        Complete node information including ancestors, children, and notes
    """
    node = navigator.find_by_code(code)
    
    # Handle chapter lookups by number
    if not node and code.isdigit():
        chapters = get_chapters_for_selection(navigator)
        chapter_num = int(code)
        if 1 <= chapter_num <= len(chapters):
            # Get the actual chapter node
            root = navigator.code_to_node.get('ROOT')
            if root and chapter_num <= len(root.children):
                node = root.children[chapter_num - 1]
                # Use the chapter number as the display code
                code = str(chapter_num)
    
    if not node:
        return {'error': f'Code {code} not found'}
    
    return {
        'current_node': {
            'code': code,  # Use the lookup code (could be chapter number)
            'name': node.name,
            'element_type': getattr(node, 'element_type', 'unknown')
        },
        'ancestors': get_ancestors_with_context(navigator, node.code),
        'children': _get_children_direct(node),
        'has_children': len(node.children) > 0,
        'is_leaf': len(node.children) == 0,
        'path_to_root': [code] if getattr(node, 'element_type', None) == 'chapter' else navigator.get_path_to_code(node.code) or []
    }


def find_codes_by_search(navigator: ICDTreeNavigator, search_term: str, max_results: int = 10) -> list[dict[str, str]]:
    """
    Search for codes by name for agent exploration.
    
    Returns:
        List of matching nodes with context
    """
    results = navigator.search_by_name(search_term, max_results)
    
    search_results = []
    for node in results:
        result_info = {
            'code': node.code,
            'name': node.name,
            'element_type': getattr(node, 'element_type', 'unknown')
        }
        search_results.append(result_info)
    
    return search_results


def create_simple_navigator(file_path: str = "icd10cm_tabular_2026.txt") -> ICDTreeNavigator:
    """Create navigator instance for hackathon demos."""
    return create_navigator(file_path)


# Demo Function
# =============

def demo_simple_navigation():
    """Demonstrate agent workflow with LLM context at each traversal step."""
    print("=== Agent Traversal Context Demo ===\n")
    print("This demo shows the context provided to an LLM at each decision point.\n")
    
    # Create navigator
    print("Building ICD-10-CM tree and navigation indexes...")
    start_time = time.time()
    navigator = create_simple_navigator()
    build_time = time.time() - start_time
    print(f"Build time: {build_time:.2f} seconds\n")
    
    # Simulate medical document context
    medical_document = "Patient presents with Type 1 diabetes mellitus with diabetic nephropathy. HbA1c is elevated at 9.2%. Patient shows proteinuria and decreased kidney function with GFR at 14 which clearly meets CKD-4 criteria."
    print(f"📄 Medical Document Context:")
    print(f"   {medical_document}\n")
    
    # Step 1: Initial chapter selection context
    print("🤖 LLM DECISION POINT 1: Chapter Selection")
    print("-" * 50)
    chapters = get_chapters_for_selection(navigator)
    print(f"Context sent to LLM:")
    print(f"Medical Document: {medical_document}")
    print(f"Available Chapters ({len(chapters)}):")
    for chapter in chapters:  # Show relevant chapters
        print(f"   Chapter {chapter['code']}: {chapter['name']}")
    print(f"   ... and {len(chapters) - 8} more")
    print("LLM sees the medical_document and the available choices for the next step and decides the next node/s.") # NOTE: Code this part where the LLM calls a client API.
    print(f"\n[LLM would select: Chapter 4 - Endocrine diseases]")
    
    # Step 2: Navigate to Chapter 4 - show context for next decision
    print(f"\n🤖 LLM DECISION POINT 2: Chapter 4 Exploration")
    print("-" * 50)
    chapter4_details = get_node_details(navigator, '4') # TODO: Make this part dynamic
    if 'error' not in chapter4_details: # TODO: Make the variables generic for dynamic calls/reusability
        print(f"Context sent to LLM:")
        print(f"Medical Document: {medical_document}")
        print(f"Established Context: ROOT") # TODO: Do not include ROOT as an ancestor
        print(f"Current Position: {chapter4_details['current_node']['name']}") # TODO: Make the variables generic for dynamic calls/reusability
        print(f"Available Children ({len(chapter4_details['children'])}):") # TODO: Make the variables generic for dynamic calls/reusability
        for child in chapter4_details['children']: # TODO: Make the variables generic for dynamic calls/reusability
            print(f"   {child['code']}: {child['name']}")
        print(f"\n[LLM would select: E08-E13 - Diabetes mellitus]") # TODO: LLM could return a Tuple, just the code (for passing along the node traversal), and the desc, which is for reporting as context/ancestry.
    
    # Step 3: Navigate to diabetes block - show context for next decision
    print(f"\n🤖 LLM DECISION POINT 3: Diabetes Block Exploration")
    print("-" * 50)
    diabetes_details = get_node_details(navigator, 'E08-E13')
    if 'error' not in diabetes_details:
        print(f"Context sent to LLM:")
        print(f"Medical Document: {medical_document}")
        print(f"Ancestor Context:") # TODO: Put the ancestors before the current code so that the LLM can understand that the medical document is grounded on the already established ancestor diagnosis.
        for ancestor in diabetes_details['ancestors']: # TODO: Make the variables generic for dynamic calls/reusability
            print(f"   {ancestor['code']}: {ancestor['name']} ({ancestor['element_type']})")
        print(f"Current Position: {diabetes_details['current_node']['name']}") # TODO: Make the variables generic for dynamic calls/reusability
        print(f"Available Children ({len(diabetes_details['children'])}):") # TODO: Make the variables generic for dynamic calls/reusability
        for child in diabetes_details['children']: # TODO: Make the variables generic for dynamic calls/reusability
            print(f"   {child['code']}: {child['name']}")
        print(f"\n[LLM would select: E10 - Type 1 diabetes mellitus]") # TODO: LLM could return a Tuple, just the code (for passing along the node traversal), and the desc, which is for reporting as context/ancestry.
    
    # Step 4: Navigate to E10 - show context for next decision  
    print(f"\n🤖 LLM DECISION POINT 4: Type 1 Diabetes Exploration")
    print("-" * 50)
    e10_details = get_node_details(navigator, 'E10')
    if 'error' not in e10_details: # TODO: Make the variables generic for dynamic calls/reusability
        print(f"Context sent to LLM:")
        print(f"Medical Document: {medical_document}")
        print(f"Full Path: {' → '.join(e10_details['path_to_root'])}") # TODO: Make the variables generic for dynamic calls/reusability
        print(f"Ancestor Context:")
        for ancestor in e10_details['ancestors']:
            print(f"   {ancestor['code']}: {ancestor['name']} ({ancestor['element_type']})")
        print(f"Available Children ({len(e10_details['children'])}):") # TODO: Make the variables generic for dynamic calls/reusability
        print(f"Current Position: {e10_details['current_node']['name']}") # TODO: Make the variables generic for dynamic calls/reusability
        for child in e10_details['children']: # TODO: Make the variables generic for dynamic calls/reusability
            print(f"   {child['code']}: {child['name']}")
        print(f"\n[LLM would select: E10.2 - with kidney complications]")
    
    # Step 5: Navigate to E10.2 - show context for final decision
    print(f"\n🤖 LLM DECISION POINT 5: Kidney Complications")
    print("-" * 50)
    e10_2_details = get_node_details(navigator, 'E10.2')
    if 'error' not in e10_2_details: # TODO: Make the variables generic for dynamic calls/reusability
        print(f"Context sent to LLM:")
        print(f"Medical Document: {medical_document}")
        print(f"Full Path: {' → '.join(e10_2_details['path_to_root'])}") # TODO: Make the variables generic for dynamic calls/reusability
        print(f"Ancestor Context:")
        for ancestor in e10_2_details['ancestors']: # TODO: Make the variables generic for dynamic calls/reusability
            print(f"   {ancestor['code']}: {ancestor['name']} ({ancestor['element_type']})")
        print(f"Current Position: {e10_2_details['current_node']['name']}") # TODO: Make the variables generic for dynamic calls/reusability
        print(f"Available Children ({len(e10_2_details['children'])}):")
        for child in e10_2_details['children']: # TODO: Make the variables generic for dynamic calls/reusability
            print(f"   {child['code']}: {child['name']}")
        print(f"\n[LLM would select: E10.21 - diabetic nephropathy]")
    
    # Step 6: Final code - E10.21
    print(f"\n✅ FINAL DECISION: Reached Target Code")
    print("-" * 50)
    e10_21_details = get_node_details(navigator, 'E10.21')
    if 'error' not in e10_21_details:
        print(f"Final Code: {e10_21_details['current_node']['code']}")
        print(f"Description: {e10_21_details['current_node']['name']}")
        print(f"Complete Path: {' → '.join(e10_21_details['path_to_root'])}")
        print(f"Is Leaf Node: {e10_21_details['is_leaf']}")
        print(f"✓ Perfect match for documented condition!")
    
    print(f"\n=== Agent Context Functions Available ===")
    print("• create_simple_navigator() - Create navigator instance")
    print("• get_chapters_for_selection(nav) - Step 1: Chapter selection context")
    print("• get_node_details(nav, code) - Each step: Complete context for LLM")
    print("• get_ancestors_with_context(nav, code) - Hierarchy for reasoning")
    print("• get_children_with_context(nav, code) - Options for next step")
    print("• find_codes_by_search(nav, term) - Alternative: direct search")
    
    print(f"\n💡 LLM Prompt Pattern:")
    print("   Given medical document + current position + ancestors + children")
    print("   → LLM decides which child(ren) to traverse next")
    print("   → Repeat until appropriate specificity reached")


if __name__ == "__main__":
    demo_simple_navigation()