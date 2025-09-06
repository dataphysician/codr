"""
Test Suite for TreeIndex Interface

Tests the TreeIndex interface capabilities:
- Node lookup and normalization
- Tree structure navigation (children, ancestors, paths)  
- Search functionality
- Node details and metadata
- Performance characteristics
"""

import pytest
import time

from core import NodeId, TreeIndex
from core.domains.medical.trees.icd_tree import create_navigator


@pytest.fixture
def tree():
    """Create tree instance for tests."""
    return create_navigator()


class TestBasicNavigation:
    """Test basic tree navigation operations."""
    
    def test_node_normalization(self, tree: TreeIndex):
        """Test node ID normalization."""
        raw_codes = ["E10.21", "(E10.21)", "  E10.21  "]
        
        for raw in raw_codes:
            normalized = tree.normalize_id(raw)
            assert normalized
            assert isinstance(normalized, NodeId)
    
    def test_invalid_normalization(self, tree: TreeIndex):
        """Test invalid node normalization raises appropriate errors."""
        with pytest.raises(Exception):
            tree.normalize_id("INVALID")
    
    def test_node_lookup(self, tree: TreeIndex):
        """Test node lookup functionality."""
        test_codes = ["E10.21", "E10", "E08-E13", "4"]
        
        for code in test_codes:
            node_id = NodeId(code)
            node = tree.get(node_id)
            
            assert node is not None
            assert node.id == node_id
            assert node.name
            assert node.element_type
    
    def test_nonexistent_node(self, tree: TreeIndex):
        """Test lookup of nonexistent node returns None."""
        node = tree.get(NodeId("NONEXISTENT"))
        assert node is None


class TestTreeStructure:
    """Test tree structure navigation."""
    
    def test_children_access(self, tree: TreeIndex):
        """Test children access for different node types."""
        # Test chapter node (should have children)
        chapter_id = NodeId("chapter_4")
        children = tree.children(chapter_id)
        assert isinstance(children, list)
        
        # Test specific code (may or may not have children)
        code_id = NodeId("E10.21")
        children = tree.children(code_id)
        assert isinstance(children, list)
    
    def test_ancestors_access(self, tree: TreeIndex):
        """Test ancestors access."""
        node_id = NodeId("E10.21")
        ancestors = tree.ancestors(node_id)
        
        assert isinstance(ancestors, list)
        # Specific codes should have ancestors
        if tree.get(node_id):
            assert len(ancestors) > 0
    
    def test_path_to_root(self, tree: TreeIndex):
        """Test path to root functionality."""
        node_id = NodeId("E10.21")
        path = tree.path_to_root(node_id)
        
        assert isinstance(path, list)
        # Path should start with the node itself
        if path:
            assert path[0] == str(node_id)
    
    def test_is_leaf(self, tree: TreeIndex):
        """Test leaf detection."""
        # Test different types of nodes
        node_id = NodeId("4")  # Chapter should not be leaf
        is_leaf = tree.is_leaf(node_id)
        assert isinstance(is_leaf, bool)
        assert is_leaf == False  # Chapter should not be leaf


class TestSearchFunctionality:
    """Test search capabilities."""
    
    def test_basic_search(self, tree: TreeIndex):
        """Test basic search functionality."""
        results = tree.search("diabetes", k=5)
        
        assert isinstance(results, list)
        assert len(results) <= 5
        
        for result in results:
            assert hasattr(result, 'id')
            assert hasattr(result, 'name')
            assert hasattr(result, 'element_type')
            # Should contain search term in name (case insensitive)
            assert "diabetes" in result.name.lower()
    
    def test_search_limit(self, tree: TreeIndex):
        """Test search result limiting."""
        # Test different k values
        for k in [1, 5, 10]:
            results = tree.search("infection", k=k)
            assert len(results) <= k
    
    def test_empty_search_results(self, tree: TreeIndex):
        """Test search with no results."""
        results = tree.search("NONEXISTENT_MEDICAL_TERM_12345", k=5)
        assert isinstance(results, list)
        assert len(results) == 0


class TestNodeDetails:
    """Test detailed node information."""
    
    def test_details_structure(self, tree: TreeIndex):
        """Test node details return proper structure."""
        node_id = NodeId("E10")
        details = tree.details(node_id)
        
        assert isinstance(details, dict)
        
        if "error" not in details:
            assert "current_node" in details
            assert "is_leaf" in details
            
            current = details["current_node"]
            assert "name" in current
            assert "element_type" in current
    
    def test_details_for_different_node_types(self, tree: TreeIndex):
        """Test details for different node types."""
        test_nodes = ["4", "E08-E13", "E10", "E10.21"]
        
        for node_code in test_nodes:
            node_id = NodeId(node_code)
            details = tree.details(node_id)
            
            assert isinstance(details, dict)
            # Should either have valid details or error
            assert ("current_node" in details) or ("error" in details)


class TestPerformance:
    """Test performance characteristics."""
    
    def test_lookup_performance(self, tree: TreeIndex):
        """Test that lookups are reasonably fast."""
        node_id = NodeId("E10")
        
        start_time = time.time()
        for _ in range(100):
            tree.get(node_id)
        lookup_time = time.time() - start_time
        
        # Should complete 100 lookups in under 1 second
        assert lookup_time < 1.0
    
    def test_search_performance(self, tree: TreeIndex):
        """Test search performance is reasonable."""
        start_time = time.time()
        results = tree.search("diabetes", k=10)
        search_time = time.time() - start_time
        
        # Search should complete in under 1 second
        assert search_time < 1.0
        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])