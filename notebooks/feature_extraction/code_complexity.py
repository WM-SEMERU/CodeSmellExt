import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from typing import Optional
import traceback
import ast
from radon.complexity import cc_visit


def get_cyclomatic_complexity(python_code: str) -> int:
    """
    Calculate the cyclomatic complexity of a Python function.
    
    Args:
        python_code (str): Python function code as a string
        
    Returns:
        int: Cyclomatic complexity of the function
    """
    
    def calculate_with_radon(code: str) -> Optional[int]:
        """Calculate cyclomatic complexity using radon."""
        try:
            ast.parse(code)
            complexity_blocks = cc_visit(code)
            if complexity_blocks:
                return complexity_blocks[0].complexity
            return 1
            
        except SyntaxError:
            return None
        except Exception:
            return None
    
    def calculate_with_tree_sitter(code: str) -> int:
        """Calculate cyclomatic complexity using tree-sitter parsing."""
        try:
            # Initialize the parser with Python language
            PY_LANGUAGE = Language(tspython.language())
            parser = Parser(PY_LANGUAGE)
            tree = parser.parse(bytes(code, "utf8"))
            
            def count_decision_points(node):
                """Count decision points in the AST that contribute to cyclomatic complexity."""
                decision_points = 0
                
                # Node types that increase cyclomatic complexity
                decision_node_types = {
                    'if_statement',
                    'elif_clause', 
                    'while_statement',
                    'for_statement',
                    'try_statement',
                    'except_clause',
                    'with_statement',
                    'match_statement',
                    'case_clause',
                    'boolean_operator',
                    'conditional_expression', 
                    'list_comprehension',
                    'dictionary_comprehension',
                    'set_comprehension',
                    'generator_expression'
                }
                decision_keywords = {'if', 'elif', 'while', 'for', 'try', 'except', 'with', 'match', 'case'}
                
                if node.type in decision_node_types:
                    decision_points += 1
                elif node.type == 'ERROR':
                    # For ERROR nodes, look for decision keywords in immediate children
                    # Only count once per ERROR node to avoid double counting
                    for child in node.children:
                        if child.type in decision_keywords:
                            decision_points += 1
                            break
                elif node.type in decision_keywords:
                    pass
                
                # Recursively count in children, but skip if we already counted this ERROR node
                for child in node.children:
                    # Skip recursing into keywords we already counted in ERROR nodes
                    if node.type == 'ERROR' and child.type in decision_keywords:
                        continue
                    decision_points += count_decision_points(child)
                
                return decision_points
            
            base_complexity = 1
            decision_points = count_decision_points(tree.root_node)
            
            return base_complexity + decision_points
            
        except Exception as e:
            return 1
    
    radon_result = calculate_with_radon(python_code)
    if radon_result is not None:
        return radon_result
    
    return calculate_with_tree_sitter(python_code)


if __name__ == "__main__":
    # Test cases
    test_codes = [
        # Simple function
        """
def simple_function():
    return 1
        """,
        
        # Function with if statement
        """
def function_with_if(x):
    if x > 0:
        return x
    else:
        return -x
        """,
        
        # Function with multiple decision points
        """
def complex_function(x, y):
    if x > 0
        if y > 0
            return x + y
        elif y < 0:
            return x - y
        else:
            return x
    elif x < 0:
        for i in range(abs(x))
            if i % 2 == 0:
                y += i
        return y
    else:
        try:
            result = y / x
        except ZeroDivisionError:
            result = 0
        return result
        """,
        
        # Function with syntax error (to test tree-sitter fallback)
        """
def broken_function(:
    if x > 0
        return x
    return 0
        """
    ]
    
    for i, code in enumerate(test_codes, 1):
        print(f"Test {i}:")
        print(f"Code:\n{code}")
        complexity = get_cyclomatic_complexity(code)
        print(f"Cyclomatic Complexity: {complexity}")
        print("-" * 50)