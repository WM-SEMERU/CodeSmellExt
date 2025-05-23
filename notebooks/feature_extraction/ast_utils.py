import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from typing import Optional
import traceback

def get_ast_height(python_code: str) -> int:
    """
    Calculate the height of the AST for a Python method using tree-sitter.
    
    Args:
        python_code (str): Python method code as a string
        
    Returns:
        int: Height of the AST (maximum depth from root to any leaf)
    """
    # Initialize the parser with Python language
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    
    # Parse the code
    tree = parser.parse(bytes(python_code, "utf8"))
    
    def calculate_height(node):
        """Recursively calculate the height of a tree-sitter node."""
        if node.child_count == 0:
            return 1
        
        max_child_height = 0
        for child in node.children:
            child_height = calculate_height(child)
            max_child_height = max(max_child_height, child_height)
        
        return max_child_height + 1
    
    return calculate_height(tree.root_node)