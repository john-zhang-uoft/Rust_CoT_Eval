from camel_converter import to_snake
from typing import List
import re
from utils.logger import debug, info, warning, error, set_verbose_mode

class ParseError(Exception):
    pass

class ContentParser:
    def __init__(self, verbose: bool = False):
        """
        Initialize the content parser
        
        Args:
            verbose: Enable verbose logging if True
        """
        self.verbose = verbose
        set_verbose_mode(verbose)

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
        debug(f"extract_all_functions - script: {script[:50]}...", self.verbose)
        debug(f"extract_all_functions - entry_point: {entry_point}", self.verbose)
        
        # Clean up the script if it contains markdown code blocks
        if "```" in script:
            code_blocks = re.findall(r'```(?:\w+)?\s*\n(.*?)\n```', script, re.DOTALL)
            debug(f"code_blocks found: {len(code_blocks)}", self.verbose)
            if code_blocks:
                script = code_blocks[0]
                debug(f"extracted code: {script[:100]}...", self.verbose)
        
        # Find all function definitions in the script
        function_pattern = re.compile(r'\bfn\s+([a-zA-Z0-9_]+).*?(?=\bfn\s+|\Z)', re.DOTALL)
        all_functions = []
        
        # First, identify the test module if it exists
        test_module_start = script.find("#[cfg(test)]")
        if test_module_start != -1:
            # Extract the script without the test module
            script_without_tests = script[:test_module_start]
            debug(f"Removed test module", self.verbose)
        else:
            script_without_tests = script
        
        # Now extract all functions
        for match in function_pattern.finditer(script_without_tests):
            fn_name = match.group(1)
            fn_content = match.group(0)
            
            # Skip main function
            if fn_name == "main":
                debug(f"Skipping main function", self.verbose)
                continue
                
            debug(f"Found function: {fn_name}", self.verbose)
            all_functions.append(fn_content)
        
        debug(f"{all_functions}", self.verbose)
        # Ensure our entry point is included (in case of camel case variations)
        entry_point_found = False
        for fn_content in all_functions:
            for variation in self._entry_point_variations(entry_point):
                if fn_content.find(f"fn {variation}") != -1:
                    entry_point_found = True
                    debug(f"Found entry point variation: {variation}", self.verbose)
                    break
            if entry_point_found:
                break
                
        if not entry_point_found:
            info("We didn't find a function with the entry point", self.verbose)
            # Try to find entry point with camel/snake case variations
            for variation in self._entry_point_variations(entry_point):
                debug(f"Searching for entry point variation: {variation}", self.verbose)
                entry_point_pattern = re.compile(r'\bfn\s+' + re.escape(variation) + r'.*?(?=\bfn\s+|\Z)', re.DOTALL)
                match = entry_point_pattern.search(script_without_tests)
                if match:
                    debug(f"Found entry point variation: {variation}", self.verbose)
                    all_functions.append(match.group(0))
                    entry_point_found = True
                    break
        
        if all_functions:
            result = "\n\n".join(all_functions)
            debug(f"Extracted {len(all_functions)} functions", self.verbose)
            return result
        else:
            debug(f"No functions found", self.verbose)
            return script

    def get_function_implementation(self, script: str, func_name: str) -> str:
        """
        Returns the body of the function named `func_name` from the given Rust script,
        not including the function signature.
        
        Note: This parser is still simplified and may not cover all edge cases (e.g., raw string literals in Rust,
        complex nested comments, macros, etc.). For a production solution, consider using a dedicated Rust parser.
        """
        debug(f"get_function_implementation - script: {script[:50]}...", self.verbose)
        debug(f"get_function_implementation - looking for function: {func_name}", self.verbose)
        
        # Regex to find the function declaration (naively assumes a simple signature)
        pattern = re.compile(r'\bfn\s+' + re.escape(func_name) + r'\s*\(', re.MULTILINE)
        match = pattern.search(script)
        if not match:
            debug(f"Function declaration not found for: {func_name}", self.verbose)
            return script

        start_index = match.start()
        # Locate the beginning of the function body.
        body_start = script.find("{", match.end())
        if body_start == -1:
            debug("Opening brace not found", self.verbose)
            return script

        debug(f"Function signature: {script[start_index:body_start+1]}", self.verbose)
        
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
        debug(f"Extracted function body: {result}", self.verbose)
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
        debug(f"__call__ - prompt: {prompt}", self.verbose)
        debug(f"__call__ - content before extraction: {content[:100]}...", self.verbose)
        debug(f"__call__ - entry_point: {entry_point}", self.verbose)
        debug(f"__call__ - extract_all: {extract_all}", self.verbose)
        debug(f"__call__ - content has code blocks: {'```' in content}", self.verbose)
        
        if "```" in content:
            # Extract code from markdown code blocks
            code_blocks = re.findall(r'```(?:\w+)?\s*\n(.*?)\n```', content, re.DOTALL)
            debug(f"code_blocks found: {len(code_blocks)}", self.verbose)
            if code_blocks:
                content = code_blocks[0]
                debug(f"extracted code: {content[:100]}...", self.verbose)
        
        if extract_all:
            # Extract all non-main, non-test functions
            return self.extract_all_functions(content, entry_point)
        else:
            # Extract only the specified function implementation
            # Try to find the function with various name variations
            for ep in self._entry_point_variations(entry_point):
                debug(f"checking content for entry point: {ep}", self.verbose)
                if re.search(rf"\bfn\s+{re.escape(ep)}\s*\(", content):
                    debug(f"found function definition in content for: {ep}", self.verbose)
                    return self.get_function_implementation(content, ep)
            
            error(f"no function found, raising ParseError")
            raise ParseError(f"Prompt is not in content:\n{content}")

