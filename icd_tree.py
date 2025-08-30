import xml.etree.ElementTree as ET
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
        note_types = ['includes', 'excludes1', 'excludes2', 'useAdditionalCode', 'codeFirst', 'codeAlso']
        
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
    
    # Process chapters
    chapters = root_element.findall('chapter')
    for chapter_elem in chapters:
        name_elem = chapter_elem.find('name')
        desc_elem = chapter_elem.find('desc')
        
        if name_elem is not None and desc_elem is not None:
            chapter_number = name_elem.text.strip() if name_elem.text else ""
            chapter_desc = desc_elem.text.strip() if desc_elem.text else ""
            chapter_name = f"Chapter {chapter_number}: {chapter_desc}"
            
            # Extract notes for the chapter
            chapter_notes = extract_notes(chapter_elem)
            
            # Create chapter node
            chapter_node = Node(
                name=chapter_name,
                parent=root,
                code=chapter_number,
                notes=chapter_notes,
                element_type="chapter"
            )
            
            # Process blocks within this chapter
            blocks = chapter_elem.findall('section')
            for block_elem in blocks:
                block_id = block_elem.get('id', '')
                desc_elem = block_elem.find('desc')
                block_desc = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""
                
                # Create block node
                block_node = Node(
                    name=block_desc,
                    parent=chapter_node,
                    code=block_id,
                    notes={},
                    element_type="block"
                )
                
                # Process diagnoses within this block
                diags = block_elem.findall('diag')
                for diag_elem in diags:
                    create_diag_nodes(diag_elem, block_node)
    
    return root

# Efficient navigation helper functions
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
    
    def get_all_diagnoses_in_block(self, block_code):
        """Get all diagnoses within a block."""
        block = self.find_block(block_code)
        if not block:
            return []
        
        from anytree import PreOrderIter
        return [node for node in PreOrderIter(block) 
                if getattr(node, 'element_type', None) == 'diagnosis']
    
    def get_all_diagnoses_in_chapter(self, chapter_number):
        """Get all diagnoses within a chapter."""
        chapter = self.find_chapter(chapter_number)
        if not chapter:
            return []
        
        from anytree import PreOrderIter
        return [node for node in PreOrderIter(chapter) 
                if getattr(node, 'element_type', None) == 'diagnosis']
    
    def search_by_name(self, search_term, case_sensitive=False):
        """Search for nodes by name substring."""
        from anytree import PreOrderIter
        
        if not case_sensitive:
            search_term = search_term.lower()
        
        results = []
        for node in PreOrderIter(self.root):
            name = node.name if case_sensitive else node.name.lower()
            if search_term in name:
                results.append(node)
        
        return results
    
    def get_siblings(self, code):
        """Get all sibling nodes of the specified code."""
        node = self.find_by_code(code)
        if node and node.parent:
            return [child for child in node.parent.children if child != node]
        return []
    
    def get_ancestors_by_type(self, code, element_type):
        """Get ancestor of specific type (e.g., chapter, block)."""
        node = self.find_by_code(code)
        if not node:
            return None
        
        for ancestor in node.ancestors:
            if getattr(ancestor, 'element_type', None) == element_type:
                return ancestor
        return None

    def is_leaf(self, code):
        """Check if a node is a leaf (terminal node with no children)."""
        node = self.find_by_code(code)
        if not node:
            return False
        return len(node.children) == 0

# Helper functions for common operations
def create_navigator(file_path="icd10cm_tabular_2026.txt"):
    """Create an ICD tree and return a navigator instance."""
    root = parse_icd10_file(file_path)
    return ICDTreeNavigator(root)

def demo_efficient_navigation():
    """Demonstrate efficient navigation methods."""
    nav = create_navigator()
    
    print("=== Efficient Navigation Demo ===\n")
    
    # 1. Direct code lookup (O(1))
    print("1. Direct code lookup:")
    a00 = nav.find_by_code('A00')
    print(f"   A00: {a00.name if a00 else 'Not found'}")
    
    a00_0 = nav.find_diagnosis('A00.0')
    print(f"   A00.0: {a00_0.name if a00_0 else 'Not found'}")
    
    # 2. Get path to diagnosis
    print(f"\n2. Path to A00.0:")
    path = nav.get_path_to_code('A00.0')
    if path:
        for i, code in enumerate(path):
            node = nav.find_by_code(code)
            indent = "   " * i
            print(f"{indent}{code}: {node.name}")
    
    # 3. Find parent chapter and block
    print(f"\n3. Find parents of A00.0:")
    chapter = nav.get_ancestors_by_type('A00.0', 'chapter')
    block = nav.get_ancestors_by_type('A00.0', 'block')
    print(f"   Chapter: {chapter.name if chapter else 'None'}")
    print(f"   Block: {block.name if block else 'None'}")
    
    # 4. Get all diagnoses in a block
    print(f"\n4. All diagnoses in block A00-A09 (first 5):")
    block_diags = nav.get_all_diagnoses_in_block('A00-A09')
    for diag in block_diags[:5]:
        print(f"   {diag.code}: {diag.name}")
    print(f"   ... and {len(block_diags) - 5} more")
    
    # 5. Search by name
    print(f"\n5. Search for 'diabetes' (first 3 results):")
    diabetes_results = nav.search_by_name('diabetes')
    for result in diabetes_results[:3]:
        print(f"   {result.code}: {result.name}")
    print(f"   ... found {len(diabetes_results)} total matches")
    
    # 6. Get siblings
    print(f"\n6. Siblings of A00:")
    siblings = nav.get_siblings('A00')
    print(f"   Found {len(siblings)} siblings (first 3):")
    for sib in siblings[:3]:
        print(f"   {sib.code}: {sib.name}")

    return nav

# --- Main Execution ---
if __name__ == "__main__":
    file_path = "icd10cm_tabular_2026.txt" # Path to your file
    
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
    import time
    
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
    
    # --- Final Statistics ---
    print(f"\n=== Final Statistics ===")
    total_nodes = sum(1 for _ in RenderTree(navigator.root))
    print(f"Total nodes: {total_nodes}")
    print(f"Chapters: {len(navigator.chapters)}")
    print(f"Blocks: {len(navigator.blocks)}")
    print(f"Diagnoses: {len(navigator.diagnoses)}")
    print(f"Index build time: Completed during initialization")
    print(f"Memory usage: {len(navigator.code_to_node)} codes indexed")
