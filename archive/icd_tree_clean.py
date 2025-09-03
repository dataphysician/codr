"""
ICD-10-CM Tree Parser and Navigator
===================================

This module handles ICD-10-CM tree building, searching, and reporting functionality.
Contains only the core tree management without any agentic or LLM-based functionality.
"""

import xml.etree.ElementTree as ET
import time
from anytree import Node, RenderTree

filepath = "icd10cm_tabular_2026.txt"


def parse_icd10_file(file_path):
    """Parses the ICD-10-CM tabular XML file and builds a tree structure."""
    
    # Parse the XML file
    tree = ET.parse(file_path)
    root_element = tree.getroot()
    
    # Create the root node for our tree
    root = Node("ICD-10-CM Root", code="ROOT", notes={}, element_type="root")
    
    def parse_code_description(note_text):
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

    def extract_notes(element):
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
    
    def find_by_code(self, code):
        """O(1) lookup by ICD code."""
        return self.code_to_node.get(code)
    
    def find_chapter(self, chapter_number):
        """Find chapter by number (e.g., '1', '2')."""
        return self.chapters.get(str(chapter_number))
    
    def find_block(self, block_code):
        """Find block by code range (e.g., 'A00-A09')."""
        return self.blocks.get(block_code)
    
    def find_diagnosis(self, diag_code):
        """Find diagnosis by code (e.g., 'A00', 'A00.0')."""
        return self.diagnoses.get(diag_code)
    
    def get_path_to_code(self, code):
        """Get full path from root to specified code."""
        node = self.find_by_code(code)
        if node:
            return [ancestor.code for ancestor in node.path]
        return None
    
    def get_ancestors_by_type(self, code, element_type):
        """Get the first ancestor of specified type (chapter, block, diagnosis)."""
        node = self.find_by_code(code)
        if node:
            for ancestor in reversed(node.path):
                if getattr(ancestor, 'element_type', None) == element_type:
                    return ancestor
        return None
    
    def get_siblings(self, code):
        """Get sibling nodes at the same level."""
        node = self.find_by_code(code)
        if node and node.parent:
            return [child for child in node.parent.children if child != node]
        return []
    
    def search_by_name(self, search_term, max_results=100):
        """Search for codes by name/description using case-insensitive substring matching."""
        results = []
        search_term = search_term.lower()
        
        for code, node in self.code_to_node.items():
            if hasattr(node, 'name') and node.name and search_term in node.name.lower():
                results.append(node)
                if len(results) >= max_results:
                    break
        
        return results
    
    def get_all_diagnoses_in_block(self, block_code):
        """Get all diagnosis codes within a specific block."""
        block_node = self.find_block(block_code)
        if not block_node:
            return []
        
        diagnoses = []
        from anytree import PreOrderIter
        
        for node in PreOrderIter(block_node):
            if getattr(node, 'element_type', None) == 'diagnosis':
                diagnoses.append(node)
        
        return diagnoses
    
    def get_all_diagnoses_in_chapter(self, chapter_number):
        """Get all diagnosis codes within a specific chapter."""
        chapter_node = self.find_chapter(chapter_number)
        if not chapter_node:
            return []
        
        diagnoses = []
        from anytree import PreOrderIter
        
        for node in PreOrderIter(chapter_node):
            if getattr(node, 'element_type', None) == 'diagnosis':
                diagnoses.append(node)
        
        return diagnoses
    
    def print_tree_structure(self, max_depth=3):
        """Print the tree structure up to specified depth."""
        for pre, _, node in RenderTree(self.root):
            depth = len(pre) // 4  # Estimate depth from prefix
            if depth <= max_depth:
                element_type = getattr(node, 'element_type', 'unknown')
                code = getattr(node, 'code', 'NO_CODE')
                print(f"{pre}{element_type.upper()}: {code} - {node.name}")
    
    def get_tree_statistics(self):
        """Get statistics about the tree structure."""
        stats = {
            'total_nodes': 0,
            'chapters': 0,
            'blocks': 0,
            'diagnoses': 0,
            'nodes_with_notes': 0
        }
        
        from anytree import PreOrderIter
        
        for node in PreOrderIter(self.root):
            stats['total_nodes'] += 1
            
            element_type = getattr(node, 'element_type', None)
            if element_type == 'chapter':
                stats['chapters'] += 1
            elif element_type == 'block':
                stats['blocks'] += 1
            elif element_type == 'diagnosis':
                stats['diagnoses'] += 1
            
            if hasattr(node, 'notes') and node.notes:
                stats['nodes_with_notes'] += 1
        
        return stats


def create_navigator():
    """Create and return a fully initialized navigator."""
    root = parse_icd10_file(filepath)
    return ICDTreeNavigator(root)


def demo_efficient_navigation():
    """Demonstrate efficient ICD tree navigation and search capabilities."""
    print("Building ICD-10-CM tree and navigation indexes...")
    start_time = time.time()
    navigator = create_navigator()
    build_time = time.time() - start_time
    print(f"Build time: {build_time:.2f} seconds")
    
    # Get statistics
    stats = navigator.get_tree_statistics()
    print(f"\nTree Statistics:")
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value:,}")
    
    # 1. Direct code lookup
    print(f"\n1. Direct code lookup (A00):")
    a00 = navigator.find_by_code('A00')
    if a00:
        print(f"   {a00.code}: {a00.name}")
        chapter = navigator.get_ancestors_by_type('A00', 'chapter')
        block = navigator.get_ancestors_by_type('A00', 'block')
        print(f"   Chapter: {chapter.name if chapter else 'Unknown'}")
        print(f"   Block: {block.name if block else 'Unknown'}")
    
    # 2. Chapter navigation
    print(f"\n2. Chapter 1 overview:")
    chapter1 = navigator.find_chapter('1')
    if chapter1:
        print(f"   {chapter1.code}: {chapter1.name}")
        print(f"   Child blocks: {len(chapter1.children)}")
        for child in chapter1.children[:2]:
            print(f"     {child.code}: {child.name}")
    
    # 3. Block exploration
    print(f"\n3. Block A00-A09 diagnoses (first 5):")
    block_diags = navigator.get_all_diagnoses_in_block('A00-A09')
    for diag in block_diags[:5]:
        print(f"   {diag.code}: {diag.name}")
    print(f"   ... and {len(block_diags) - 5} more")
    
    # 4. Search by name
    print(f"\n4. Search for 'diabetes' (first 3 results):")
    diabetes_results = navigator.search_by_name('diabetes')
    for result in diabetes_results[:3]:
        print(f"   {result.code}: {result.name}")
    print(f"   ... found {len(diabetes_results)} total matches")
    
    # 5. Get siblings
    print(f"\n5. Siblings of A00:")
    siblings = navigator.get_siblings('A00')
    print(f"   Found {len(siblings)} siblings (first 3):")
    for sib in siblings[:3]:
        print(f"   {sib.code}: {sib.name}")

    return navigator


if __name__ == "__main__":
    file_path = "icd10cm_tabular_2026.txt"
    
    print("=== ICD-10-CM Tree Parser ===\n")
    
    # Parse and create navigator
    print("Parsing XML and building navigation indexes...")
    navigator = demo_efficient_navigation()
    
    print(f"\n=== Additional Navigation Examples ===\n")
    
    # Example: Access notes and detailed information
    print("Detailed information for E10 (Type 1 diabetes mellitus):")
    e10 = navigator.find_diagnosis('E10')
    if e10:
        print(f"  Code: {e10.code}")
        print(f"  Name: {e10.name}")
        print(f"  Notes available: {list(e10.notes.keys())}")
        if e10.notes:
            for note_type, note_list in e10.notes.items():
                print(f"    {note_type}:")
                for code, description in note_list[:2]:  # Show first 2 examples
                    if code:
                        print(f"      - {description} ({code})")
                    else:
                        print(f"      - {description}")
                if len(note_list) > 2:
                    print(f"      ... and {len(note_list) - 2} more")
        print(f"  Children: {len(e10.children)} subcategories")
        
        # Show parent hierarchy
        chapter = navigator.get_ancestors_by_type('E10', 'chapter')
        block = navigator.get_ancestors_by_type('E10', 'block')
        print(f"  Chapter: {chapter.name if chapter else 'None'}")
        print(f"  Block: {block.name if block else 'None'}")
    
    # Performance comparison
    print(f"\n=== Performance Comparison ===")
    
    # Measure indexed lookup performance
    start = time.time()
    for _ in range(1000):
        node = navigator.find_by_code('A00.0')
    indexed_time = time.time() - start
    
    # Measure tree traversal performance (simulating without index)
    from anytree import PreOrderIter
    start = time.time()
    for _ in range(10):  # Fewer iterations because it's much slower
        for node in PreOrderIter(navigator.root):
            if getattr(node, 'code', '') == 'A00.0':
                break
    traversal_time = (time.time() - start) * 100  # Scale to compare with 1000 iterations
    
    print(f"1000 indexed lookups: {indexed_time:.4f} seconds")
    print(f"1000 tree traversals: {traversal_time:.4f} seconds")
    print(f"Speedup: {traversal_time/indexed_time:.1f}x faster with indexing")
    
    # Final Statistics
    print(f"\n=== Final Statistics ===")
    total_nodes = sum(1 for _ in RenderTree(navigator.root))
    print(f"Total nodes: {total_nodes}")
    print(f"Chapters: {len(navigator.chapters)}")
    print(f"Blocks: {len(navigator.blocks)}")
    print(f"Diagnoses: {len(navigator.diagnoses)}")
    print(f"Index build time: Completed during initialization")
    print(f"Memory usage: {len(navigator.code_to_node)} codes indexed")