import keyword
import builtins
import re
import ast
import spacy
import textwrap
from collections import defaultdict
from typing import Dict, List, Tuple, Set


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
    raise

EXCLUDED_NAMES = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls"}

def split_identifier(name: str) -> List[str]:
    """
    Split camelCase and snake_case identifiers into individual words.
    
    Args:
        name: The identifier to split
        
    Returns:
        List of lowercase word parts
    """
    # Handle snake_case by replacing underscores with spaces
    name_no_underscore = name.replace('_', ' ')
    
    # Split on camelCase boundaries and numbers
    parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+', name_no_underscore)
    
    # Filter out empty parts and convert to lowercase
    return [part.lower() for part in parts if part]

class IdentifierExtractor(ast.NodeVisitor):
    """AST visitor to extract all identifiers from Python code."""
    
    def __init__(self):
        self.identifiers: Set[str] = set()
    
    def visit_Name(self, node):
        """Visit variable names and identifiers."""
        if node.id not in EXCLUDED_NAMES:
            self.identifiers.add(node.id)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Visit function definitions and their arguments."""
        if node.name not in EXCLUDED_NAMES:
            self.identifiers.add(node.name)
        
        # Extract argument names
        for arg in node.args.args:
            if arg.arg not in EXCLUDED_NAMES:
                self.identifiers.add(arg.arg)
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Visit class definitions."""
        if node.name not in EXCLUDED_NAMES:
            self.identifiers.add(node.name)
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        """Visit attribute access (e.g., obj.attr)."""
        if node.attr not in EXCLUDED_NAMES:
            self.identifiers.add(node.attr)
        self.generic_visit(node)

def analyze_pos_for_tokens(tokens: List[str]) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Analyze POS tags for a list of tokens.
    
    Args:
        tokens: List of word tokens to analyze
        
    Returns:
        Tuple of (token_to_pos_map, pos_counts)
    """
    if not tokens:
        return {}, {}
    
    # Join tokens and process with spaCy
    doc = nlp(" ".join(tokens))
    
    token_pos_map = {}
    pos_counts = defaultdict(int)
    
    for token in doc:
        if token.text in tokens:  # Only count tokens we're interested in
            token_pos_map[token.text] = token.pos_
            pos_counts[token.pos_] += 1
    
    return token_pos_map, dict(pos_counts)

def get_pos_dict_from_code(code: str) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Extract identifiers from Python code and analyze their parts of speech.
    
    Args:
        code: Python source code as a string
        
    Returns:
        Tuple of (pos_to_words_dict, pos_counts_dict)
    """
    # Remove leading/trailing whitespace and dedent the code
    import textwrap
    cleaned_code = textwrap.dedent(code).strip()
    
    try:
        tree = ast.parse(cleaned_code)
    except SyntaxError as e:
        print(f"Syntax error in code: {e}")
        return {}, {}
    
    # Extract identifiers
    extractor = IdentifierExtractor()
    extractor.visit(tree)
    identifiers = extractor.identifiers
    
    if not identifiers:
        return {}, {}
    
    # Process each identifier
    pos_dict = defaultdict(list)
    pos_count = defaultdict(int)
    
    for identifier in identifiers:
        tokens = split_identifier(identifier)
        
        # Handle single character identifiers
        if not tokens and len(identifier) == 1:
            pos_dict["NOUN"].append(identifier)
            pos_count["NOUN"] += 1
            continue
        
        # Handle identifiers that couldn't be split
        if not tokens:
            tokens = [identifier]
        
        # Analyze POS for each token
        token_pos_map, token_pos_counts = analyze_pos_for_tokens(tokens)
        
        # Add to main dictionaries
        for token, pos in token_pos_map.items():
            pos_dict[pos].append(token)
            pos_count[pos] += token_pos_counts.get(pos, 0)
    
    # Remove duplicates for the first dict
    unique_pos_dict = {k: list(set(v)) for k, v in pos_dict.items()}
    
    return unique_pos_dict, dict(pos_count)

def print_analysis_results(pos_dict: Dict[str, List[str]], pos_counts: Dict[str, int], 
                          test_name: str = "Code") -> None:
    """Print formatted analysis results."""
    print(f"\n{test_name} - POS Analysis:")
    print("=" * 50)
    
    if not pos_dict:
        print("No identifiers found or code could not be parsed.")
        return
    
    print("\nUnique words by POS tag:")
    for pos, words in sorted(pos_dict.items()):
        print(f"  {pos}: {words}")
    
    print(f"\nPOS tag counts:")
    for pos, count in sorted(pos_counts.items()):
        print(f"  {pos}: {count}")

# Test cases
if __name__ == '__main__':
    test_cases = [
        ('''
        def calculateTotalPrice(itemList):
            total = 0
            for item in itemList:
                total += item.price
            return total
        ''', "Calculate Total Price Function"),
        
        ('''
        def getDataFromDB(userId):
            data = fetch_user(userId)
            return data
        ''', "Database Data Retrieval"),
        
        ('''
        def processData(x, y):
            result = x * y
            return result
        ''', "Simple Data Processing"),
        
        ('''
        def convertToXmlString(jsonData):
            xml_string = json_to_xml(jsonData)
            return xml_string
        ''', "JSON to XML Conversion"),
        
        ('''
        def get_total_price_from_list(items):
            total_price = sum([item.price for item in items])
            return total_price
        ''', "Price Calculation from List"),
        
        ('''
        class UserManager:
            def __init__(self, database_connection):
                self.db = database_connection
                self.active_users = []
            
            def create_user_profile(self, user_data):
                new_user = User(user_data)
                self.active_users.append(new_user)
                return new_user
        ''', "Class with Methods")
    ]
    
    print("Python Code Identifier POS Analysis")
    print("=" * 60)
    
    for code, description in test_cases:
        unique_pos_dict, pos_count = get_pos_dict_from_code(code)
        print_analysis_results(unique_pos_dict, pos_count, description)
        print("-" * 60)