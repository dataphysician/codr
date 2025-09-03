"""
Optimized ICD-10-CM Tree Parser with Fast Loading
Implements disk caching, streaming parsing, and lightweight data structures.
"""

import xml.etree.ElementTree as ET
import pickle
import os
import time
from dataclasses import dataclass, field


@dataclass
class ICDNode:
    """Lightweight node structure for ICD tree."""
    code: str
    name: str
    element_type: str  # 'root', 'chapter', 'block', 'diagnosis'
    notes: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    parent_code: str | None = None
    children_codes: list[str] = field(default_factory=list)
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children)."""
        return len(self.children_codes) == 0


class ICDTreeLightNavigator:
    """Fast, lightweight navigator for ICD-10-CM tree with caching."""
    
    def __init__(self, xml_file_path: str = "icd10cm_tabular_2026.txt"):
        self.xml_file_path = xml_file_path
        self.cache_file = f"{xml_file_path}.cache"
        self.nodes: dict[str, ICDNode] = {}
        self.root_code = "ROOT"
        
        # Load data (from cache or parse fresh)
        self._load_data()
    
    def _should_rebuild_cache(self) -> bool:
        """Check if cache needs rebuilding based on file modification time."""
        if not os.path.exists(self.cache_file):
            return True
        
        if not os.path.exists(self.xml_file_path):
            return False
        
        xml_mtime = os.path.getmtime(self.xml_file_path)
        cache_mtime = os.path.getmtime(self.cache_file)
        
        return xml_mtime > cache_mtime
    
    def _load_data(self):
        """Load tree data from cache or parse from XML."""
        if self._should_rebuild_cache():
            print("Parsing XML and building cache...")
            start_time = time.time()
            self._parse_xml()
            self._save_cache()
            parse_time = time.time() - start_time
            print(f"Parsing completed in {parse_time:.2f} seconds")
        else:
            print("Loading from cache...")
            start_time = time.time()
            self._load_cache()
            load_time = time.time() - start_time
            print(f"Cache loaded in {load_time:.3f} seconds")
    
    def _parse_code_description(self, note_text: str) -> list[tuple[str, str]]:
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

    def _extract_notes(self, element) -> dict[str, list[tuple[str, str]]]:
        """Extract all note types from XML element."""
        notes = {}
        
        # Standard note types
        note_types = ['includes', 'excludes1', 'excludes2', 'useAdditionalCode', 
                     'codeFirst', 'codeAlso']
        
        for note_type in note_types:
            note_elements = element.findall(note_type)
            if note_elements:
                note_list = []
                for note_elem in note_elements:
                    for note in note_elem.findall('note'):
                        if note.text:
                            parsed_notes = self._parse_code_description(note.text.strip())
                            note_list.extend(parsed_notes)
                if note_list:
                    notes[note_type] = note_list
        
        # Inclusion terms
        inclusion_terms = element.findall('inclusionTerm')
        if inclusion_terms:
            inclusion_list = []
            for term_elem in inclusion_terms:
                for note in term_elem.findall('note'):
                    if note.text:
                        parsed_notes = self._parse_code_description(note.text.strip())
                        inclusion_list.extend(parsed_notes)
            if inclusion_list:
                notes['inclusionTerm'] = inclusion_list
        
        return notes
    
    def _parse_xml(self):
        """Parse XML using streaming parser for memory efficiency."""
        self.nodes = {}
        
        # Create root node
        root_node = ICDNode(
            code=self.root_code,
            name="ICD-10-CM Root",
            element_type="root"
        )
        self.nodes[self.root_code] = root_node
        
        # Use iterparse for memory-efficient streaming
        try:
            for event, elem in ET.iterparse(self.xml_file_path, events=('start', 'end')):
                if event == 'end':
                    if elem.tag == 'chapter':
                        self._process_chapter(elem, self.root_code)
                        elem.clear()  # Free memory
                    elif elem.tag == 'section':
                        # Blocks are processed within chapters
                        pass
        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
            # Fallback to full tree parsing
            self._parse_xml_full_tree()
    
    def _parse_xml_full_tree(self):
        """Fallback: parse entire XML tree (slower but more reliable)."""
        tree = ET.parse(self.xml_file_path)
        root_element = tree.getroot()
        
        # Process chapters
        for chapter_elem in root_element.findall('chapter'):
            self._process_chapter(chapter_elem, self.root_code)
    
    def _process_chapter(self, chapter_elem, parent_code: str):
        """Process a chapter element."""
        name_elem = chapter_elem.find('name')
        desc_elem = chapter_elem.find('desc')
        
        if name_elem is not None and desc_elem is not None:
            chapter_number = name_elem.text.strip() if name_elem.text else ""
            chapter_desc = desc_elem.text.strip() if desc_elem.text else ""
            chapter_name = f"Chapter {chapter_number}: {chapter_desc}"
            
            notes = self._extract_notes(chapter_elem)
            
            chapter_node = ICDNode(
                code=chapter_number,
                name=chapter_name,
                element_type="chapter",
                notes=notes,
                parent_code=parent_code
            )
            
            self.nodes[chapter_number] = chapter_node
            self.nodes[parent_code].children_codes.append(chapter_number)
            
            # Process blocks within chapter
            for block_elem in chapter_elem.findall('section'):
                self._process_block(block_elem, chapter_number)
    
    def _process_block(self, block_elem, parent_code: str):
        """Process a block element."""
        block_id = block_elem.get('id', '')
        desc_elem = block_elem.find('desc')
        block_desc = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""
        
        block_node = ICDNode(
            code=block_id,
            name=block_desc,
            element_type="block",
            parent_code=parent_code
        )
        
        self.nodes[block_id] = block_node
        self.nodes[parent_code].children_codes.append(block_id)
        
        # Process diagnoses within block
        for diag_elem in block_elem.findall('diag'):
            self._process_diagnosis(diag_elem, block_id)
    
    def _process_diagnosis(self, diag_elem, parent_code: str):
        """Process a diagnosis element recursively."""
        name_elem = diag_elem.find('name')
        desc_elem = diag_elem.find('desc')
        
        if name_elem is not None and desc_elem is not None:
            code = name_elem.text.strip() if name_elem.text else ""
            description = desc_elem.text.strip() if desc_elem.text else ""
            
            notes = self._extract_notes(diag_elem)
            
            diag_node = ICDNode(
                code=code,
                name=description,
                element_type="diagnosis",
                notes=notes,
                parent_code=parent_code
            )
            
            self.nodes[code] = diag_node
            self.nodes[parent_code].children_codes.append(code)
            
            # Process nested diagnoses
            for nested_diag in diag_elem.findall('diag'):
                self._process_diagnosis(nested_diag, code)
    
    def _save_cache(self):
        """Save parsed data to cache file."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.nodes, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _load_cache(self):
        """Load parsed data from cache file."""
        try:
            with open(self.cache_file, 'rb') as f:
                self.nodes = pickle.load(f)
        except Exception as e:
            print(f"Cache loading failed: {e}")
            print("Falling back to XML parsing...")
            self._parse_xml()
            self._save_cache()
    
    # Navigation Methods
    
    def find_by_code(self, code: str) -> ICDNode | None:
        """Find node by ICD code (O(1) lookup)."""
        return self.nodes.get(code)
    
    def search_by_name(self, search_term: str, case_sensitive: bool = False) -> list[ICDNode]:
        """Search nodes by name substring."""
        if not case_sensitive:
            search_term = search_term.lower()
        
        results = []
        for node in self.nodes.values():
            name = node.name if case_sensitive else node.name.lower()
            if search_term in name:
                results.append(node)
        
        return results
    
    def get_parent(self, code: str) -> ICDNode | None:
        """Get direct parent of a node."""
        node = self.find_by_code(code)
        if node and node.parent_code:
            return self.find_by_code(node.parent_code)
        return None
    
    def get_children(self, code: str) -> list[ICDNode]:
        """Get all direct children of a node."""
        node = self.find_by_code(code)
        if not node:
            return []
        
        children = []
        for child_code in node.children_codes:
            child = self.find_by_code(child_code)
            if child:
                children.append(child)
        
        return children
    
    def get_ancestors(self, code: str) -> list[ICDNode]:
        """Get all ancestors from node to root."""
        ancestors = []
        current = self.find_by_code(code)
        
        while current and current.parent_code:
            parent = self.find_by_code(current.parent_code)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        
        return ancestors
    
    def get_ancestor_by_type(self, code: str, element_type: str) -> ICDNode | None:
        """Get ancestor of specific type (chapter, block, etc.)."""
        for ancestor in self.get_ancestors(code):
            if ancestor.element_type == element_type:
                return ancestor
        return None
    
    def is_leaf(self, code: str) -> bool:
        """Check if node is a leaf (has no children)."""
        node = self.find_by_code(code)
        return node.is_leaf() if node else False
    
    def get_all_descendants(self, code: str) -> list[ICDNode]:
        """Get all descendants of a node recursively."""
        descendants = []
        children = self.get_children(code)
        
        for child in children:
            descendants.append(child)
            descendants.extend(self.get_all_descendants(child.code))
        
        return descendants
    
    def get_siblings(self, code: str) -> list[ICDNode]:
        """Get all sibling nodes."""
        parent = self.get_parent(code)
        if not parent:
            return []
        
        siblings = []
        for sibling_code in parent.children_codes:
            if sibling_code != code:
                sibling = self.find_by_code(sibling_code)
                if sibling:
                    siblings.append(sibling)
        
        return siblings
    
    def get_path_to_root(self, code: str) -> list[str]:
        """Get path from node to root as list of codes."""
        path = [code]
        ancestors = self.get_ancestors(code)
        path.extend([ancestor.code for ancestor in ancestors])
        return path
    
    # Specialized lookups
    
    def find_chapter(self, chapter_number: str) -> ICDNode | None:
        """Find chapter by number."""
        return self.find_by_code(str(chapter_number))
    
    def find_block(self, block_code: str) -> ICDNode | None:
        """Find block by code range."""
        return self.find_by_code(block_code)
    
    def find_diagnosis(self, diag_code: str) -> ICDNode | None:
        """Find diagnosis by code."""
        node = self.find_by_code(diag_code)
        return node if node and node.element_type == "diagnosis" else None
    
    def get_all_diagnoses_in_block(self, block_code: str) -> list[ICDNode]:
        """Get all diagnoses within a block."""
        diagnoses = []
        descendants = self.get_all_descendants(block_code)
        
        for node in descendants:
            if node.element_type == "diagnosis":
                diagnoses.append(node)
        
        return diagnoses
    
    def get_all_diagnoses_in_chapter(self, chapter_number: str) -> list[ICDNode]:
        """Get all diagnoses within a chapter."""
        diagnoses = []
        descendants = self.get_all_descendants(str(chapter_number))
        
        for node in descendants:
            if node.element_type == "diagnosis":
                diagnoses.append(node)
        
        return diagnoses
    
    # Statistics and info
    
    def get_stats(self) -> dict[str, int]:
        """Get tree statistics."""
        stats = {
            'total_nodes': len(self.nodes),
            'chapters': 0,
            'blocks': 0,
            'diagnoses': 0,
            'leaves': 0
        }
        
        for node in self.nodes.values():
            if node.element_type == 'chapter':
                stats['chapters'] += 1
            elif node.element_type == 'block':
                stats['blocks'] += 1
            elif node.element_type == 'diagnosis':
                stats['diagnoses'] += 1
            
            if node.is_leaf():
                stats['leaves'] += 1
        
        return stats


def create_light_navigator(xml_file_path: str = "icd10cm_tabular_2026.txt") -> ICDTreeLightNavigator:
    """Create optimized ICD tree navigator."""
    return ICDTreeLightNavigator(xml_file_path)


def demo_light_navigation():
    """Demonstrate the lightweight navigator."""
    print("=== ICD Tree Light Navigation Demo ===\n")
    
    # Create navigator (uses cache if available)
    start_time = time.time()
    nav = create_light_navigator()
    init_time = time.time() - start_time
    
    print(f"Navigator initialized in {init_time:.3f} seconds")
    print(f"Statistics: {nav.get_stats()}\n")
    
    # Test core functionality
    print("1. Finding node by code:")
    e10 = nav.find_by_code('E10')
    if e10:
        print(f"   {e10.code}: {e10.name}")
        print(f"   Type: {e10.element_type}")
        print(f"   Is leaf: {e10.is_leaf()}")
        print(f"   Notes available: {list(e10.notes.keys())}")
        if e10.notes:
            for note_type, note_list in e10.notes.items():
                print(f"     {note_type}:")
                for code, description in note_list[:2]:  # Show first 2 examples
                    if code:
                        print(f"       - {description} ({code})")
                    else:
                        print(f"       - {description}")
                if len(note_list) > 2:
                    print(f"       ... and {len(note_list) - 2} more")
    
    print(f"\n2. Getting parent:")
    parent = nav.get_parent('E10')
    if parent:
        print(f"   Parent of E10: {parent.code}: {parent.name}")
    
    print(f"\n3. Getting children:")
    children = nav.get_children('E10')
    print(f"   E10 has {len(children)} children:")
    for child in children[:3]:
        print(f"      {child.code}: {child.name}")
    
    print(f"\n4. Ancestry chain for E10.21:")
    ancestors = nav.get_ancestors('E10.21')
    print(f"   Ancestors: {[a.code for a in ancestors]}")
    
    print(f"\n5. Search by name:")
    results = nav.search_by_name('diabetes')
    print(f"   Found {len(results)} matches for 'diabetes' (showing first 3):")
    for result in results[:3]:
        print(f"      {result.code}: {result.name}")
    
    return nav


if __name__ == "__main__":
    demo_light_navigation()