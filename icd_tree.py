"""
ICD-10-CM Tree Navigation Interface
==================================

Standalone tree interface for hackathon demos and agentic workflows.
Contains ICD tree parsing, navigation, and agent-friendly context methods.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from typing import Any
from anytree import Node

# -------------------------------------------------------------------
# Canonical patterns (shared meaning across the system)
# -------------------------------------------------------------------

# Extract parenthesized, actionable ICD references embedded in prose notes.
CODE_PATTERN_RE: str = (
    r'\(([A-Z][0-9][0-9](?:\.[0-9A-X-]+)?'
    r'(?:-[A-Z][0-9][0-9](?:\.[0-9A-X-]+)?)?'
    r'(?:, ?[A-Z][0-9][0-9](?:\.[0-9A-X-]+)?)*'
    r'(?:\.?-)?)\)'
)

# Validate the INNER operational form you actually store/use (no parentheses).
NORMALIZED_CODE_RE: str = (
    r'^[A-Z][0-9][0-9](?:\.[0-9A-X-]+)?'
    r'(?:-[A-Z][0-9][0-9](?:\.[0-9A-X-]+)?)?'
    r'(?:, ?[A-Z][0-9][0-9](?:\.[0-9A-X-]+)?)*'
    r'(?:\.?-)?$'
)

def normalize_code(s: str) -> str:
    """
    Accept '(X..)' or 'X..' → return normalized inner 'X..'; raise if invalid.
    Used to sanitize codes extracted from notes and any external lookup.
    """
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if not re.match(NORMALIZED_CODE_RE, s):
        raise ValueError(f"Invalid ICD code/range/list: {s!r}")
    return s


# Core ICD Tree Classes and Functions
# ====================================

def parse_icd10_file(file_path: str = "icd10cm_tabular_2026.xml") -> Node:
    """Parses the ICD-10-CM tabular XML file and builds a tree structure."""
    tree = ET.parse(file_path)
    root_element = tree.getroot()

    # Create the root node for our tree
    root = Node("ICD-10-CM Root", code="ROOT", notes={}, element_type="root")

    def parse_code_description(note_text: str) -> list[tuple[str, str]]:
        """
        Parse code(s) and description from note text, returning (code, description) tuples.

        - 'code' is the INNER normalized form (no parentheses) when present and valid
        - If a note line has no valid actionable code, return [("", description)]
        """
        matches = re.findall(CODE_PATTERN_RE, note_text or "")
        if matches:
            codes: list[str] = []
            for match in matches:
                # Handle comma-separated codes like "J09.X3, J10.2, J11.2"
                for raw in match.split(","):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        codes.append(normalize_code(raw))
                    except ValueError:
                        # Skip malformed/non-actionable tokens inside ()
                        continue

            if codes:
                # Description: text before the first '(' if available
                desc_match = re.match(r'^(.*?)\s*\([A-Z][0-9]', note_text or "")
                description = desc_match.group(1).strip() if desc_match else (note_text or "").strip()
                return [(code, description) for code in codes]

        # No valid actionable codes found → return prose as context only
        return [("", (note_text or "").strip())]

    def extract_notes(element: ET.Element) -> dict[str, list[tuple[str, str]]]:
        """Extract notes from note elements like includes, excludes1, etc."""
        notes: dict[str, list[tuple[str, str]]] = {}

        note_types = [
            "includes", "excludes1", "excludes2",
            "useAdditionalCode", "codeFirst", "codeAlso", "sevenChrNote"
        ]

        for note_type in note_types:
            note_elements = element.findall(note_type)
            if not note_elements:
                continue
            bucket: list[tuple[str, str]] = []
            for note_elem in note_elements:
                for note in note_elem.findall("note"):
                    if not note.text:
                        continue
                    parsed = parse_code_description(note.text.strip())
                    bucket.extend(parsed)  # includes ("", prose) for context-only lines
            if bucket:
                notes[note_type] = bucket

        # inclusionTerm (contextual synonyms/phrases)
        inclusion_terms = element.findall("inclusionTerm")
        if inclusion_terms:
            incl_bucket: list[tuple[str, str]] = []
            for term_elem in inclusion_terms:
                for note in term_elem.findall("note"):
                    if not note.text:
                        continue
                    parsed = parse_code_description(note.text.strip())
                    incl_bucket.extend(parsed)
            if incl_bucket:
                notes["inclusionTerm"] = incl_bucket

        return notes

    def create_diag_nodes(diag_element: ET.Element, parent_node: Node) -> None:
        """Recursively create nodes for diagnosis elements."""
        name_elem = diag_element.find("name")
        desc_elem = diag_element.find("desc")

        if name_elem is not None and desc_elem is not None:
            code = name_elem.text.strip() if name_elem.text else ""
            description = desc_elem.text.strip() if desc_elem.text else ""

            # Best-effort normalization (do not fail tree build if code looks odd)
            try:
                if code:
                    code = normalize_code(code)
            except ValueError:
                pass

            # Extract notes for this diagnosis
            notes = extract_notes(diag_element)

            # Create the node
            diag_node = Node(
                name=description,
                parent=parent_node,
                code=code,
                notes=notes,
                element_type="diagnosis",
            )

            # Process any nested diagnosis elements
            nested_diags = diag_element.findall("diag")
            for nested_diag in nested_diags:
                create_diag_nodes(nested_diag, diag_node)

    def create_section_nodes(section_element: ET.Element, parent_node: Node) -> None:
        """Create nodes for section elements (blocks)."""
        desc_elem = section_element.find("desc")

        if desc_elem is not None:
            section_desc = desc_elem.text.strip() if desc_elem.text else ""

            # Try to extract a range pattern like "A00-A09"
            range_match = re.match(
                r"([A-Z][0-9][0-9](?:\.[0-9A-X-]+)?-[A-Z][0-9][0-9](?:\.[0-9A-X-]+)?)",
                section_desc,
            )
            if range_match:
                section_code = range_match.group(1)
            else:
                # Fallback to section ID if no range found
                section_id = section_element.get("id", "")
                section_code = section_id if section_id else section_desc[:10]

            # Best-effort normalization for section ranges
            try:
                if section_code:
                    section_code = normalize_code(section_code)
            except ValueError:
                pass

            # Extract notes for this section
            notes = extract_notes(section_element)

            # Create the section node
            section_node = Node(
                name=section_desc,
                parent=parent_node,
                code=section_code,
                notes=notes,
                element_type="block",
            )

            # Process diagnosis elements within this section
            diag_elements = section_element.findall("diag")
            for diag_element in diag_elements:
                create_diag_nodes(diag_element, section_node)

    # Build chapters → blocks → diagnoses
    for chapter_element in root_element.findall("chapter"):
        chapter_desc = chapter_element.find("desc")
        if chapter_desc is not None:
            chapter_name = chapter_desc.text.strip() if chapter_desc.text else ""
            chapter_num = chapter_element.get("num", "")

            # Extract notes for this chapter
            notes = extract_notes(chapter_element)

            # Create chapter node (chapter_num may not be an ICD code; keep as provided)
            chapter_node = Node(
                name=chapter_name,
                parent=root,
                code=chapter_num,
                notes=notes,
                element_type="chapter",
            )

            # Process sections within the chapter
            section_elements = chapter_element.findall("section")
            for section_element in section_elements:
                create_section_nodes(section_element, chapter_node)

    return root


class ICDTreeNavigator:
    """Efficient navigation utilities for the ICD tree structure."""

    def __init__(self, root: Node) -> None:
        self.root: Node = root
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build indexes for O(1) lookups by code."""
        from anytree import PreOrderIter

        self.code_to_node: dict[str, Node] = {}
        self.chapters: dict[str, Node] = {}
        self.blocks: dict[str, Node] = {}
        self.diagnoses: dict[str, Node] = {}

        for node in PreOrderIter(self.root):
            if hasattr(node, "code") and node.code:
                self.code_to_node[node.code] = node

                element_type = getattr(node, "element_type", None)
                if element_type == "chapter":
                    self.chapters[node.code] = node
                elif element_type == "block":
                    self.blocks[node.code] = node
                elif element_type == "diagnosis":
                    self.diagnoses[node.code] = node

    def find_by_code(self, code: str) -> Node | None:
        """O(1) lookup by ICD code (expects INNER normalized form)."""
        return self.code_to_node.get(code)

    def get_path_to_code(self, code: str) -> list[str] | None:
        """Get full path from root to specified code (list of node.codes)."""
        node = self.find_by_code(code)
        if node:
            return [ancestor.code for ancestor in node.path]
        return None

    def search_by_name(self, search_term: str, max_results: int = 100) -> list[Node]:
        """Search for codes by name/description using case-insensitive substring matching."""
        results: list[Node] = []
        s = search_term.lower()
        for _code, node in self.code_to_node.items():
            if hasattr(node, "name") and node.name and s in node.name.lower():
                results.append(node)
                if len(results) >= max_results:
                    break
        return results


def create_navigator(file_path: str = "icd10cm_tabular_2026.xml") -> ICDTreeNavigator:
    """Create and return a fully initialized navigator."""
    root = parse_icd10_file(file_path)
    return ICDTreeNavigator(root)


def create_simple_navigator(file_path: str = "icd10cm_tabular_2026.xml") -> ICDTreeNavigator:
    """Alias kept for demos."""
    return create_navigator(file_path)


# Agent-Friendly Navigation Functions
# ===================================

def _get_children_direct(node: Node) -> list[dict[str, str]]:
    """Helper to get children directly from a node."""
    children: list[dict[str, str]] = []
    for child in node.children:
        child_info = {
            "code": child.code,
            "name": child.name,
            "element_type": getattr(child, "element_type", "unknown"),
        }
        children.append(child_info)
    return children


def get_ancestors_with_context(navigator: ICDTreeNavigator, code: str) -> list[dict[str, str]]:
    """
    Get ancestor hierarchy with context for LLM prompting.

    Returns:
        List of ancestor nodes with code, name, and element_type
    """
    node = navigator.find_by_code(code)
    if not node:
        return []

    ancestors: list[dict[str, str]] = []
    current: Node | None = node
    # Walk up the tree to collect ancestors
    while current and current.parent:
        ancestor_info = {
            "code": current.parent.code,
            "name": current.parent.name,
            "element_type": getattr(current.parent, "element_type", "unknown"),
        }
        ancestors.append(ancestor_info)
        current = current.parent

    return list(reversed(ancestors))  # Root to current order


def get_children_with_context(navigator: ICDTreeNavigator, code: str) -> list[dict[str, str]]:
    """
    Get child nodes with context for agent decision making.

    Returns:
        List of child nodes with code, name, and element_type
    """
    node = navigator.find_by_code(code)

    # Handle chapter lookups by number (back-compat)
    if not node and code.isdigit():
        chapters = get_chapters_for_selection(navigator)
        chapter_num = int(code)
        if 1 <= chapter_num <= len(chapters):
            root = navigator.code_to_node.get("ROOT")
            if root and chapter_num <= len(root.children):
                node = root.children[chapter_num - 1]

    if not node:
        return []

    return _get_children_direct(node)


def get_chapters_for_selection(navigator: ICDTreeNavigator) -> list[dict[str, str]]:
    """
    Get all chapters for initial agent selection.

    Returns:
        List of chapter nodes for agent to choose from
    """
    chapters: list[dict[str, str]] = []
    root = navigator.code_to_node.get("ROOT")
    if root:
        for i, child in enumerate(root.children):
            if getattr(child, "element_type", None) == "chapter":
                chapter_info = {
                    "code": str(i + 1),            # 1-based index for UI selection
                    "name": child.name,
                    "element_type": "chapter",
                    "actual_node_code": child.code,  # stored for navigation if needed
                }
                chapters.append(chapter_info)
    return chapters


def get_node_details(navigator: ICDTreeNavigator, code: str) -> dict[str, Any]:
    """
    Get comprehensive node details for agent context.

    Returns:
        Complete node information including ancestors, children, and notes
    """
    node = navigator.find_by_code(code)

    # Handle chapter lookups by number (back-compat)
    if not node and code.isdigit():
        chapters = get_chapters_for_selection(navigator)
        chapter_num = int(code)
        if 1 <= chapter_num <= len(chapters):
            root = navigator.code_to_node.get("ROOT")
            if root and chapter_num <= len(root.children):
                node = root.children[chapter_num - 1]
                code = str(chapter_num)

    if not node:
        return {"error": f"Code {code} not found"}

    return {
        "current_node": {
            "code": code,  # Use lookup code (could be chapter number for UI)
            "name": node.name,
            "element_type": getattr(node, "element_type", "unknown"),
        },
        "ancestors": get_ancestors_with_context(navigator, node.code),
        "children": _get_children_direct(node),
        "has_children": len(node.children) > 0,
        "is_leaf": len(node.children) == 0,
        "path_to_root": (
            [code] if getattr(node, "element_type", None) == "chapter"
            else navigator.get_path_to_code(node.code) or []
        ),
        "notes": node.notes,
    }


def find_codes_by_search(navigator: ICDTreeNavigator, search_term: str, max_results: int = 10) -> list[dict[str, str]]:
    """
    Search for codes by name for agent exploration.

    Returns:
        List of matching nodes with context
    """
    results = navigator.search_by_name(search_term, max_results)
    search_results: list[dict[str, str]] = []
    for node in results:
        search_results.append({
            "code": node.code,
            "name": node.name,
            "element_type": getattr(node, "element_type", "unknown"),
        })
    return search_results


# Additional tiny helpers for traversal modules
# ============================================

def note_codes_only(node: Node, key: str) -> list[str]:
    """
    Return only the codes from node.notes[key], ignoring descriptions.
    Useful for guards like excludes1/2, codeFirst, useAdditionalCode, codeAlso.
    """
    pairs: list[tuple[str, str]] = node.notes.get(key, [])
    return [c for (c, _desc) in pairs if c]

def children_codes(navigator: ICDTreeNavigator, code: str) -> list[str]:
    """
    Return child codes for a given node code. Empty list if not found.
    """
    node = navigator.find_by_code(code)
    if not node:
        return []
    return [child.code for child in node.children]

def is_ancestor(navigator: ICDTreeNavigator, ancestor_code: str, code: str) -> bool:
    """
    True if 'ancestor_code' lies on the path (root→code) of 'code'.
    Used to ensure 7th-character handoff only occurs within lineage.
    """
    path = navigator.get_path_to_code(code) or []
    return ancestor_code in path



# demo.py (or append to icd_tree.py below the definitions)

import time
from icd_tree import (
    create_simple_navigator,
    get_chapters_for_selection,
    get_node_details,
)

def _print_children(children, limit=12):
    n = len(children)
    take = min(n, limit)
    for child in children[:take]:
        print(f"   {child['code']}: {child['name']}")
    more = n - take
    if more > 0:
        print(f"   ... and {more} more")

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

    # Simulated medical document context for the agent
    medical_document = (
        "Patient presents with Type 1 diabetes mellitus with diabetic nephropathy. "
        "HbA1c is elevated at 9.2%. Patient shows proteinuria and decreased kidney "
        "function with GFR at 14 which clearly meets CKD-4 criteria."
    )
    print("📄 Medical Document Context:")
    print(f"   {medical_document}\n")

    # ----------------------------
    # DECISION POINT 1: Chapter(s)
    # ----------------------------
    print("🤖 LLM DECISION POINT 1: Chapter Selection")
    print("-" * 50)
    chapters = get_chapters_for_selection(navigator)
    print("Context sent to LLM:")
    print(f"Medical Document: {medical_document}")
    print(f"Available Chapters ({len(chapters)}):")
    _print_children(chapters, limit=8)
    print("LLM sees the medical_document and the available choices for the next step "
          "and decides the next node(s).  [Stub: replace with actual LLM call]\n")
    print("[LLM would select: Chapter 4 - Endocrine, nutritional and metabolic diseases]")

    # ----------------------------
    # DECISION POINT 2: Chapter 4
    # ----------------------------
    print("\n🤖 LLM DECISION POINT 2: Chapter 4 Exploration")
    print("-" * 50)
    chapter4_details = get_node_details(navigator, "4")  # LLM-chosen; can be dynamic
    if "error" in chapter4_details:
        print("Could not load Chapter 4 details. Stopping demo.")
        return

    print("Context sent to LLM:")
    print(f"Medical Document: {medical_document}")
    print(f"Current Position: {chapter4_details['current_node']['name']}")
    children = chapter4_details["children"]
    print(f"Available Children ({len(children)}):")
    _print_children(children)
    print("\n[LLM would select: E08-E13 - Diabetes mellitus]")

    # ---------------------------------
    # DECISION POINT 3: Diabetes block
    # ---------------------------------
    print("\n🤖 LLM DECISION POINT 3: Diabetes Block Exploration")
    print("-" * 50)
    diabetes_details = get_node_details(navigator, "E08-E13")
    if "error" in diabetes_details:
        print("Could not load 'E08-E13' block. Your XML vintage may differ.")
        return

    print("Context sent to LLM:")
    print(f"Medical Document: {medical_document}")
    print("Ancestor Context:")
    for a in diabetes_details["ancestors"]:
        # Avoid printing ROOT here
        if a["code"] != "ROOT":
            print(f"   {a['code']}: {a['name']} ({a['element_type']})")
    print(f"Current Position: {diabetes_details['current_node']['name']}")
    children = diabetes_details["children"]
    print(f"Available Children ({len(children)}):")
    _print_children(children)
    print("\n[LLM would select: E10 - Type 1 diabetes mellitus]")

    # -----------------------------
    # DECISION POINT 4: E10 branch
    # -----------------------------
    print("\n🤖 LLM DECISION POINT 4: Type 1 Diabetes Exploration")
    print("-" * 50)
    e10_details = get_node_details(navigator, "E10")
    if "error" in e10_details:
        print("Could not load 'E10'. Stopping demo.")
        return

    path = " → ".join([p for p in (e10_details["path_to_root"] or []) if p != "ROOT"])
    print("Context sent to LLM:")
    print(f"Medical Document: {medical_document}")
    if path:
        print(f"Full Path: {path}")
    print("Ancestor Context:")
    for a in e10_details["ancestors"]:
        if a["code"] != "ROOT":
            print(f"   {a['code']}: {a['name']} ({a['element_type']})")
    print(f"Current Position: {e10_details['current_node']['name']}")
    children = e10_details["children"]
    print(f"Available Children ({len(children)}):")
    _print_children(children)
    print("\n[LLM would select: E10.2 - with kidney complications]")

    # ----------------------------------
    # DECISION POINT 5: E10.2 sub-branch
    # ----------------------------------
    print("\n🤖 LLM DECISION POINT 5: Kidney Complications")
    print("-" * 50)
    e10_2_details = get_node_details(navigator, "E10.2")
    if "error" in e10_2_details:
        print("Could not load 'E10.2'. Stopping demo.")
        return

    path = " → ".join([p for p in (e10_2_details["path_to_root"] or []) if p != "ROOT"])
    print("Context sent to LLM:")
    print(f"Medical Document: {medical_document}")
    if path:
        print(f"Full Path: {path}")
    print("Ancestor Context:")
    for a in e10_2_details["ancestors"]:
        if a["code"] != "ROOT":
            print(f"   {a['code']}: {a['name']} ({a['element_type']})")
    print(f"Current Position: {e10_2_details['current_node']['name']}")
    children = e10_2_details["children"]
    print(f"Available Children ({len(children)}):")
    _print_children(children)
    print("\n[LLM would select: E10.21 - diabetic nephropathy]")

    # --------------------
    # FINAL: E10.21 leaf?
    # --------------------
    print("\n✅ FINAL DECISION: Reached Target Code")
    print("-" * 50)
    e10_21_details = get_node_details(navigator, "E10.21")
    if "error" in e10_21_details:
        print("Could not load 'E10.21'. Your XML vintage may differ.")
        return

    print(f"Final Code: {e10_21_details['current_node']['code']}")
    print(f"Description: {e10_21_details['current_node']['name']}")
    final_path = " → ".join([p for p in (e10_21_details["path_to_root"] or []) if p != 'ROOT'])
    print(f"Complete Path: {final_path}")
    print(f"Is Leaf Node: {e10_21_details['is_leaf']}")
    print("✓ Perfect match for documented condition!\n")

    print("=== Agent Context Functions Available ===")
    print("• create_simple_navigator() - Create navigator instance")
    print("• get_chapters_for_selection(nav) - Step 1: Chapter selection context")
    print("• get_node_details(nav, code) - Each step: Complete context for LLM")
    print("• get_ancestors_with_context(nav, code) - Hierarchy for reasoning")
    print("• get_children_with_context(nav, code) - Options for next step")
    print("• find_codes_by_search(nav, term) - Alternative: direct search\n")

    print("💡 LLM Prompt Pattern:")
    print("   Given medical document + current position + ancestors + children")
    print("   → LLM decides which child(ren) to traverse next")
    print("   → Repeat until appropriate specificity reached")

if __name__ == "__main__":
    demo_simple_navigation()