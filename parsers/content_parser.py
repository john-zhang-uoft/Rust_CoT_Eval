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

    def extract_all_functions(self, script: str, entry_point: str) -> str:
        """
        Extracts all functions from the given Rust script, excluding the main function
        and test module. This ensures that all helper functions are included.
        
        Args:
            script: The Rust code as a string
            entry_point: The primary function that's being called (for finding variations)
            
        Returns:
            A string containing all the non-main, non-test functions
        """
        print(f"DEBUG - extract_all_functions - script: {script[:50]}...")
        print(f"DEBUG - extract_all_functions - entry_point: {entry_point}")
        
        # Clean up the script if it contains markdown code blocks
        if "```" in script:
            code_blocks = re.findall(r'```(?:\w+)?\s*\n(.*?)\n```', script, re.DOTALL)
            print(f"DEBUG - code_blocks found: {len(code_blocks)}")
            if code_blocks:
                script = code_blocks[0]
                print(f"DEBUG - extracted code: {script[:100]}...")
        
        # Find all function definitions in the script
        function_pattern = re.compile(r'\bfn\s+([a-zA-Z0-9_]+).*?(?=\bfn\s+|\Z)', re.DOTALL)
        all_functions = []
        
        # First, identify the test module if it exists
        test_module_start = script.find("#[cfg(test)]")
        if test_module_start != -1:
            # Extract the script without the test module
            script_without_tests = script[:test_module_start]
            print(f"DEBUG - Removed test module")
        else:
            script_without_tests = script
        
        # Now extract all functions
        for match in function_pattern.finditer(script_without_tests):
            fn_name = match.group(1)
            fn_content = match.group(0)
            
            # Skip main function
            if fn_name == "main":
                print(f"DEBUG - Skipping main function")
                continue
                
            print(f"DEBUG - Found function: {fn_name}")
            all_functions.append(fn_content)
        
        # Ensure our entry point is included (in case of camel case variations)
        entry_point_found = False
        for fn_content in all_functions:
            for variation in self._entry_point_variations(entry_point):
                if re.search(rf"\bfn\s+{re.escape(variation)}\s*\(", fn_content):
                    entry_point_found = True
                    break
            if entry_point_found:
                break
                
        if not entry_point_found:
            # Try to find entry point with camel/snake case variations
            for variation in self._entry_point_variations(entry_point):
                print(f"DEBUG - Searching for entry point variation: {variation}")
                entry_point_pattern = re.compile(r'\bfn\s+' + re.escape(variation) + r'.*?(?=\bfn\s+|\Z)', re.DOTALL)
                match = entry_point_pattern.search(script_without_tests)
                if match:
                    print(f"DEBUG - Found entry point variation: {variation}")
                    all_functions.append(match.group(0))
                    entry_point_found = True
                    break
        
        if all_functions:
            result = "\n\n".join(all_functions)
            print(f"DEBUG - Extracted {len(all_functions)} functions")
            return result
        else:
            print(f"DEBUG - No functions found")
            return script

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


    def __call__(self, prompt: str, content: str, entry_point: str, extract_all: bool = False):
        """
        Parse content to extract either all functions or a single function implementation
        
        Args:
            prompt: The prompt used to generate the content
            content: The generated code content
            entry_point: The primary function name / entry point
            extract_all: If True, extract all functions excluding main and tests.
                        If False, extract only the specified entry_point function implementation.
        
        Returns:
            The extracted function(s)
        
        Raises:
            ParseError: If the entry point function can't be found in the content
        """
        print(f"DEBUG - __call__ - prompt: {prompt}")
        print(f"DEBUG - __call__ - content before extraction: {content[:100]}...")
        print(f"DEBUG - __call__ - entry_point: {entry_point}")
        print(f"DEBUG - __call__ - extract_all: {extract_all}")
        print(f"DEBUG - __call__ - content has code blocks: {'```' in content}")
        
        if "```" in content:
            # Extract code from markdown code blocks
            code_blocks = re.findall(r'```(?:\w+)?\s*\n(.*?)\n```', content, re.DOTALL)
            print(f"DEBUG - code_blocks found: {len(code_blocks)}")
            if code_blocks:
                content = code_blocks[0]
                print(f"DEBUG - extracted code: {content[:100]}...")
        
        if extract_all:
            # Extract all non-main, non-test functions
            return self.extract_all_functions(content, entry_point)
        else:
            # Extract only the specified function implementation
            # Try to find the function with various name variations
            for ep in self._entry_point_variations(entry_point):
                print(f"DEBUG - checking content for entry point: {ep}")
                if re.search(rf"\bfn\s+{re.escape(ep)}\s*\(", content):
                    print(f"DEBUG - found function definition in content for: {ep}")
                    return self.get_function_implementation(content, ep)
            
            print(f"DEBUG - no function found, raising ParseError")
            raise ParseError(f"Prompt is not in content:\n{content}")

