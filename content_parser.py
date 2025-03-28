from camel_converter import to_snake
from typing import List
import re

class ParseError(Exception):
    pass

class ContentParser:

    @staticmethod
    def _entry_point_variations(entry_point: str) -> List[str]:
        # NOTE: workaround dataset's bug with entry point naming
        return [
            entry_point,
            to_snake(entry_point),
            entry_point[0].lower() + entry_point[1:],
        ]

    def get_function_implementation(self, script: str, func_name: str) -> str:
        """
        Returns the body of the function named `func_name` from the given Rust script,
        not including the function signature.
        
        Note: This parser is still simplified and may not cover all edge cases (e.g., raw string literals in Rust,
        complex nested comments, macros, etc.). For a production solution, consider using a dedicated Rust parser.
        """
        print(f"DEBUG - get_function_implementation - script: {script[:50]}...")
        print(f"DEBUG - get_function_implementation - looking for function: {func_name}")
        
        # Regex to find the function declaration (naively assumes a simple signature)
        pattern = re.compile(r'\bfn\s+' + re.escape(func_name) + r'\s*\(', re.MULTILINE)
        match = pattern.search(script)
        if not match:
            print(f"DEBUG - Function declaration not found for: {func_name}")
            return script

        start_index = match.start()
        # Locate the beginning of the function body.
        body_start = script.find("{", match.end())
        if body_start == -1:
            print("DEBUG - Opening brace not found")
            return script

        print(f"DEBUG - Function signature: {script[start_index:body_start+1]}")
        
        # State variables to handle context.
        brace_counter = 1
        i = body_start + 1

        in_string = False
        in_char = False
        in_single_line_comment = False
        in_multi_line_comment = False

        while i < len(script) and brace_counter > 0:
            char = script[i]
            
            # Check if we're inside a string literal.
            if in_string:
                # End string if we see an unescaped double quote.
                if char == '"' and script[i - 1] != '\\':
                    in_string = False
            # Check if we're inside a character literal.
            elif in_char:
                if char == "'" and script[i - 1] != '\\':
                    in_char = False
            # Check if we're inside a single-line comment.
            elif in_single_line_comment:
                if char == "\n":
                    in_single_line_comment = False
            # Check if we're inside a multi-line comment.
            elif in_multi_line_comment:
                if char == "*" and i + 1 < len(script) and script[i + 1] == "/":
                    in_multi_line_comment = False
                    i += 1  # Skip the '/'
            else:
                # Not inside a string, char literal, or comment.
                # Start of a string literal.
                if char == '"':
                    in_string = True
                # Start of a character literal.
                elif char == "'":
                    in_char = True
                # Check for the start of a comment.
                elif char == "/" and i + 1 < len(script):
                    next_char = script[i + 1]
                    if next_char == "/":
                        in_single_line_comment = True
                        i += 1  # Skip the second '/'
                    elif next_char == "*":
                        in_multi_line_comment = True
                        i += 1  # Skip the '*'
                # Count braces if they are code structure.
                elif char == '{':
                    brace_counter += 1
                elif char == '}':
                    brace_counter -= 1

            i += 1

        # Return the function body including the final closing brace
        result = script[body_start + 1:i].strip()
        print(f"DEBUG - Extracted function body: {result}")
        return result
    

    def __call__(self, prompt: str, content: str, entry_point: str):
        # NOTE: Model doesn't follow instructions directly:
        # adds description of change and sometimes fixes
        # typos, or other "bugs" in description.
        print(f"DEBUG - __call__ - prompt: {prompt}")
        print(f"DEBUG - __call__ - content before extraction: {content[:100]}...")
        print(f"DEBUG - __call__ - entry_point: {entry_point}")
        print(f"DEBUG - __call__ - content has code blocks: {'```' in content}")
        
        if "```" in content:
            # Extract code from markdown code blocks
            code_blocks = re.findall(r'```(?:\w+)?\s*\n(.*?)\n```', content, re.DOTALL)
            print(f"DEBUG - code_blocks found: {len(code_blocks)}")
            if code_blocks:
                content = code_blocks[0]
                print(f"DEBUG - extracted code: {content[:100]}...")
            
        # second parse content with assumption that model wrote code without description
        print(f"DEBUG - trying to find function in whole content")
        for ep in self._entry_point_variations(entry_point):
            print(f"DEBUG - checking content for entry point: {ep}")
            if re.search(rf"\bfn\s+{re.escape(ep)}\s*\(", content):
                print(f"DEBUG - found function definition in content for: {ep}")
                return self.get_function_implementation(content, ep)
            
        print(f"DEBUG - no function found, raising ParseError")
        raise ParseError(f"Prompt is not in content:\n{content}")

