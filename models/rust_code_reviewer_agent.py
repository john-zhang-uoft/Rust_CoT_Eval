#!/usr/bin/env python3
"""
Code reviewer model for evaluating and providing feedback on generated code
"""

import os
import tempfile
import subprocess
import re
import time
from typing import Dict, Any, List, Tuple, Optional

import termcolor

from models.code_generation_models import CodeGenerationModel
from parsers.content_parser import ContentParser, ParseError

DEFAULT_TIMEOUT = 60  # 1 minute timeout for compilation and test execution

class RustCodeReviewerAgent(CodeGenerationModel):
    """Agent that reviews code, tests it, and provides feedback"""
    
    def __init__(self, model: CodeGenerationModel, timeout: int = DEFAULT_TIMEOUT, sample_idx: int = 0, rust_dir: Optional[str] = None, thread_id: Optional[int] = None, replace_generated_function_signature: bool = False):
        self.model = model
        self.timeout = timeout
        self.sample_idx = sample_idx  # Store the sample index for file naming
        self.thread_id = thread_id or os.getpid()  # Use process ID as default thread ID
        self.parser = ContentParser()  # Initialize ContentParser for parsing code
        self._verbose = True  # Always enable verbose logging for code reviewer agent
        self._replace_generated_function_signature = replace_generated_function_signature  # Whether to keep function signatures in the generated code
        
        # Get absolute path to project root (where this script is located)
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Set up paths for Rust project
        if rust_dir:
            self.rust_dir = rust_dir  # Use custom rust_dir if provided
        else:
            self.rust_dir = os.path.join(self.project_root, "rust")
            
        self.rust_src = os.path.join(self.rust_dir, "src")
        self.rust_bin = os.path.join(self.rust_src, "bin")
            
        # Create src/bin directory if it doesn't exist
        os.makedirs(self.rust_bin, exist_ok=True)
        
        # Check if Cargo.toml exists
        cargo_toml = os.path.join(self.rust_dir, "Cargo.toml")
        if not os.path.exists(cargo_toml):
            raise FileNotFoundError(f"Cargo.toml not found in {self.rust_dir}")
        
        self._log(f"Using Rust project at: {self.rust_dir} with thread ID {self.thread_id}")
        self._log(f"Replace generated function signature: {self._replace_generated_function_signature}")
            
    def _log(self, message: str, color: str = None, always: bool = False, attrs=None, separate_section: bool = False):
        """
        Log a message if verbose is true or always is true
        
        Args:
            message: The message to log
            color: Optional color for the message (using termcolor)
            always: Whether to log regardless of verbose setting
            attrs: Additional attributes for termcolor
            separate_section: Whether to add dividers before and after the message for emphasis
        """
        if self._verbose or always:
            if separate_section:
                print("-" * 40)
            if color:
                attrs = attrs or []
                print(termcolor.colored(message, color, attrs=attrs))
            else:
                print(message)
            if separate_section:
                print("-" * 40)
            
    def _parse_code(self, raw_code: str, prompt: str, entry_point: str) -> str:
        """
        Parse code using ContentParser to extract clean implementation
        
        Args:
            raw_code: Raw code from model
            prompt: Original prompt
            entry_point: Function name / entry point
            
        Returns:
            Parsed code
        """
        # Remove any test modules from the raw code before extraction
        test_module_index = raw_code.find("#[cfg(test)]")
        if test_module_index != -1:
            self._log("Removing test module from code before extraction.", "cyan", attrs=["bold"])
            raw_code = raw_code[:test_module_index]
        
        self._log(f"\nATTEMPTING TO PARSE CODE for {entry_point}:", "cyan", attrs=["bold"])
        
        # Check if the raw code directly contains the entry point function
        if f"fn {entry_point}" in raw_code:
            self._log(f"Raw code contains function definition: 'fn {entry_point}'")
        else:
            self._log(f"WARNING: Raw code does not directly contain 'fn {entry_point}' - this may cause parsing issues", "yellow")
        
        # Check for Markdown code blocks
        code_blocks = re.findall(r'```(?:rust)?\s*(.*?)\s*```', raw_code, re.DOTALL)
        if code_blocks:
            self._log(f"Found {len(code_blocks)} Markdown code blocks in raw code")
            # Print largest code block preview
            largest_block = max(code_blocks, key=len)
            preview = largest_block.split('\n')[:5]
            self._log(f"Largest code block preview ({len(largest_block)} chars):")
            self._log('\n'.join(preview))
            if f"fn {entry_point}" in largest_block:
                self._log(f"  - Contains function definition: 'fn {entry_point}'")
                raw_code = largest_block
        else:
            self._log("No Markdown code blocks found in raw code")
        
        # Try to use ContentParser
        try:
            self._log(f"Calling ContentParser with entry_point='{entry_point}', extract_all={self._replace_generated_function_signature}")
            parsed_code = self.parser(prompt, raw_code, entry_point, extract_all=self._replace_generated_function_signature)
            self._log(f"\nPARSE SUCCESSFUL for {entry_point}:", "green", attrs=["bold"])
            self._log("-" * 40)
            self._log(parsed_code)
            self._log("-" * 40)
            return parsed_code
        except ParseError as e:
            self._log(f"\nPARSE ERROR: {str(e)}", "red", attrs=["bold"])
            
            # Fallback: Direct extraction of function
            self._log("Attempting fallback extraction...")
            pattern = fr'fn\s+{re.escape(entry_point)}[^{{]*{{[^{{]*(?:{{[^{{]*}}[^{{]*)*}}'
            match = re.search(pattern, raw_code, re.DOTALL)
            if match:
                extracted_code = match.group(0)
                self._log(f"\nFALLBACK EXTRACTION SUCCESSFUL:", "yellow", attrs=["bold"])
                self._log("-" * 40)
                self._log(extracted_code)
                self._log("-" * 40)
                return extracted_code
            
            # If everything fails, try to extract from code blocks
            if code_blocks:
                for block in sorted(code_blocks, key=len, reverse=True):
                    if f"fn {entry_point}" in block:
                        self._log(f"\nEXTRACTING FROM CODE BLOCK:", "yellow", attrs=["bold"])
                        self._log("-" * 40)
                        self._log(block)
                        self._log("-" * 40)
                        return block
            
            # Last resort: return raw code
            self._log("All extraction attempts failed. Returning raw code.", "red")
            return raw_code

    def generate_code(self, prompt: str, n: int = 1) -> List[str]:
        """
        Generate code using the model
        """
        # Not implemented
        raise NotImplementedError("RustCodeReviewerAgent does not support code generation")
    
    
    def generate_feedback(self, original_prompt: str, declaration: str, implementation: str, entry_point: str, n: int = 1) -> Tuple[List[str], List[Any], List[bool]]:
        """
        Review the provided code implementation and return a list with the feedback
        """
        # Parse the implementation using ContentParser with correct extraction mode
        implementation = self._parse_code(implementation, original_prompt, entry_point)
        
        feedbacks = []
        detail_list = []   # Use a separate list variable for review details
        successes = []

        for i in range(n):
            success, feedback, review_detail = self.review_rust(original_prompt, declaration, implementation, entry_point)
            feedbacks.append(feedback)
            detail_list.append(review_detail)
            successes.append(success)
        return feedbacks, detail_list, successes

    def review_rust(self, prompt: str, declaration: str, implementation: str, entry_point: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Review Rust code"""
        details = {}
        
        # Handle based on replace_generated_function_signature setting
        if self._replace_generated_function_signature:
            # When we want to keep function signatures from the implementation
            self._log("Using implementation with its own function signatures (empty declaration)", "cyan")
            full_code = implementation
            empty_declaration = ""
        elif not declaration.strip() and implementation.strip():
            # Backward compatibility: empty declaration but implementation exists
            self._log("Using implementation as complete code (declaration is empty)", "cyan")
            full_code = implementation
            empty_declaration = ""
        else:
            # Normal case: combine declaration and implementation
            self._log("Using declaration + implementation", "cyan")
            full_code = f"{declaration}\n{implementation}"
            empty_declaration = declaration
        
        # Step 1: Check if the code compiles
        compiles, compile_feedback, compile_details = self.check_compilation(empty_declaration, full_code)
        details["compilation"] = compile_details
        
        if not compiles:
            return False, f"Compilation failed: {compile_feedback}", details
        
        # Step 2: Generate and run tests
        tests_generated, test_code, test_gen_details = self.generate_tests(prompt, empty_declaration, full_code, entry_point)
        details["test_generation"] = test_gen_details
        
        if not tests_generated:
            return False, f"Could not generate tests: {test_code}", details
        
        # Step 3: Run the tests
        tests_pass, test_output, test_details = self.run_tests(empty_declaration, full_code, test_code, entry_point)
        details["test_execution"] = test_details
        
        if not tests_pass:
            # Step 4: If tests fail, generate more detailed feedback
            detailed_feedback, analysis_details = self.generate_detailed_feedback(
                prompt, empty_declaration, full_code, test_code, test_output
            )
            details["analysis"] = analysis_details
            return False, detailed_feedback, details
        
        return True, "Code looks good. All tests passed.", details
    
    def check_compilation(self, declaration: str, implementation: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Check if the code compiles"""
        start_time = time.time()
        process = None
        
        try:
            # Create a unique file name based on the sample index only
            file_prefix = f"sample_{self.sample_idx}"
            file_name = f"{file_prefix}.rs"
            file_path = os.path.join(self.rust_bin, file_name)
            
            # Add necessary imports and directives
            prelude = "#![allow(unused_imports)]\n#![allow(unused_variables)]\n"
            
            # Ensure there's a main function for binary compilation
            has_main = "fn main()" in declaration or "fn main()" in implementation
            main_fn = "" if has_main else "fn main(){}\n"
            
            # Combine the code with proper formatting
            combined_code = f"{prelude}{main_fn}{declaration}\n{implementation}"
            
            # Print the code being compiled
            self._log(f"\nCOMPILING CODE:", "cyan", attrs=["bold"])
            self._log("-" * 40)
            self._log(combined_code)
            self._log("-" * 40)
            
            # Write to file
            with open(file_path, "w") as f:
                f.write(combined_code)
            
            # Log what we're compiling
            self._log(f"Compiling: {file_path}")
            
            # Try to compile with warnings about unused imports suppressed
            process = subprocess.run(
                ["cargo", "check", "--bin", file_prefix],
                cwd=self.rust_dir,  # Use main directory
                capture_output=True,
                text=True,
                timeout=self.timeout  # Use instance timeout
            )
        except subprocess.TimeoutExpired:
            self._log(f"Compilation timed out for {file_prefix}.rs after {self.timeout} seconds", "red", attrs=["bold"])
            return False, f"Compilation timed out after {self.timeout} seconds.", {
                "duration": time.time() - start_time,
                "error": "Compilation timed out",
                "file_path": file_path
            }
    
        # Collect details
        details = {
            "duration": time.time() - start_time,
            "command": f"cargo check --bin {file_prefix} --message-format=json",
            "return_code": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "raw_stderr": process.stderr,
            "file_path": file_path
        }
        
        if process.returncode != 0:
            # Use the model to summarize the error
            error_prompt = f"""
Analyze this Rust compilation error and provide a clear, concise explanation of what's wrong and how the user should fix it:

Code:
rust
{combined_code}

Compilation error:
{process.stderr}

Provide a brief explanation of what's wrong and how to fix it.
If there are missing imports, remind the user that their solution must match the problem description and not use any imports not listed in the problem description. Tell them to not use structs absent in the imports and list those offending structs.
Problem description (the user's solution may only use imports listed in this description):
{declaration}
"""
            start_time = time.time()
            error_analysis = self.model.generate_code(error_prompt, n=1)[0]
            details["analysis_duration"] = time.time() - start_time
            details["error_analysis"] = error_analysis
            
            return False, error_analysis, details
        
        # Log successful compilation details
        self._log(f"Compilation succeeded for {file_prefix}.rs in {details['duration']:.2f} seconds.", "green")
        return True, "Code compiles successfully.", details
    
    def generate_tests(self, prompt: str, declaration: str, implementation: str, entry_point: str, implementation_visible: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate unit tests for the code"""
        start_time = time.time()
        combined_code = declaration + "\n" + implementation
        
        # Print the implementation we're generating tests for
        self._log(f"\nGENERATING TESTS FOR IMPLEMENTATION:", "cyan", attrs=["bold"])
        self._log("-" * 40)
        self._log(combined_code)
        self._log("-" * 40)
        
        test_prompt = f"""
You are a Rust testing expert.
You are trying to write comprehensive unit tests to ensure that a solution to the following Rust problem is correct:
Problem:
{prompt}

Read the problem description carefully and write the unit tests to ensure that the solutions solve the problem described.
"""
        if implementation_visible:
            test_prompt += f"""
This is the solution you will be testing:

```rust
{combined_code}
```

The entry point function name is {entry_point}.
"""
        test_prompt += f"""
Format your response as a complete Rust test module wrapped with #[cfg(test)] that can be directly appended to the code.
Only write the tests, not the implementation code. Make sure the tests will run with 'cargo test'.
Include useful test cases that would verify the function works correctly for various inputs.
Do not include any explanations, comments, or markdown formatting in your response - only pure Rust code.
"""
        raw_test_code = self.model.generate_code(test_prompt, n=1)[0]
        
        # Record duration
        details = {
            "duration": time.time() - start_time,
            "raw_test_code": raw_test_code
        }
        
        # Show the raw test code
        self._log(f"\nRAW TEST CODE:", "cyan", attrs=["bold"])
        self._log("-" * 40)
        self._log(raw_test_code)
        self._log("-" * 40)
        
        # Extract test module directly instead of using ContentParser
        # First try to extract from Markdown code blocks
        code_blocks = re.findall(r'```(?:rust)?\s*(.*?)\s*```', raw_test_code, re.DOTALL)
        if code_blocks:
            # Use the largest code block
            test_code = max(code_blocks, key=len)
            self._log(f"\nEXTRACTED TEST CODE FROM CODE BLOCKS:", "cyan", attrs=["bold"])
            self._log("-" * 40)
            self._log(test_code)
            self._log("-" * 40)
        else:
            # Try to extract #[cfg(test)] module directly
            test_module_match = re.search(r'(#\[cfg\(test\)].*)', raw_test_code, re.DOTALL)
            if test_module_match:
                test_code = test_module_match.group(1)
                self._log(f"\nEXTRACTED TEST CODE FROM #[cfg(test)]:", "cyan", attrs=["bold"])
                self._log("-" * 40)
                self._log(test_code)
                self._log("-" * 40)
            else:
                # Just use the raw response, but remove any non-code content
                # Remove any "Here's the test code:" prefixes
                test_code = re.sub(r'^.*?(\bmod tests\b|\buse super::\*)', r'\1', raw_test_code, flags=re.DOTALL)
                self._log(f"\nEXTRACTED TEST CODE FROM RAW RESPONSE:", "cyan", attrs=["bold"])
                self._log("-" * 40)
                self._log(test_code)
                self._log("-" * 40)
        
        # Ensure the test code starts with #[cfg(test)]
        if not test_code.strip().startswith("#[cfg(test)]"):
            # If the test module doesn't start with the proper module definition, add it
            if "mod test" in test_code or "mod tests" in test_code:
                # If it has a module but no cfg attribute
                test_code = "#[cfg(test)]\n" + test_code
            else:
                # Add full wrapper
                test_code = f"#[cfg(test)]\nmod tests {{\n    use super::*;\n\n    {test_code}\n}}"
                
            self._log(f"\nADDED #[cfg(test)] WRAPPER:", "yellow", attrs=["bold"])
            self._log("-" * 40)
            self._log(test_code)
            self._log("-" * 40)
        
        # Check if the test module contains test functions
        if "#[test]" not in test_code:
            details["valid_test_not_found"] = True
            return False, "Generated test code doesn't contain valid test functions.", details
        
        # Validate code structure - check for unbalanced braces
        opening_braces = test_code.count('{')
        closing_braces = test_code.count('}')
        if opening_braces > closing_braces:
            # Add missing closing braces
            test_code += '\n' + '}' * (opening_braces - closing_braces)
            self._log(f"\nFIXED UNBALANCED BRACES:", "yellow", attrs=["bold"])
            self._log("-" * 40)
            self._log(test_code)
            self._log("-" * 40)
        
        details["test_module"] = test_code
        
        # Print only a short preview of the tests to avoid overwhelming output
        test_preview = test_code.split("\n")
        if len(test_preview) > 5:
            test_preview = test_preview[:5] + ["..."]
        test_preview = "\n".join(test_preview)
        self._log(f"\nFINAL TEST CODE (PREVIEW):\n{test_preview}", "green", attrs=["bold"])
        
        return True, test_code, details
    
    def run_tests(self, declaration: str, implementation: str, tests: str, entry_point: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Run the tests and check if they pass"""
        start_time = time.time()
        file_path = "unknown"
        file_prefix = "unknown"
        compile_process = None
        process = None
        combined_code = ""
        
        try:
            # Create a unique file name based on the sample index only
            file_prefix = f"sample_{self.sample_idx}_{self.thread_id}" if self.sample_idx is not None else f"sample_{self.thread_id}"
            file_name = f"{file_prefix}.rs"
            file_path = os.path.join(self.rust_bin, file_name)
            
            # Add necessary imports and directives for better error handling
            prelude = "#![allow(unused_imports)]\n#![allow(unused_variables)]\n#![allow(dead_code)]\n"
            
            # Ensure there's a main function for binary compilation
            has_main = "fn main()" in declaration or "fn main()" in implementation
            main_fn = "" if has_main else "fn main(){}\n"
            
            # Verify the test code is valid Rust and contains test functions
            if "#[test]" not in tests:
                return False, "Generated test code doesn't contain any test functions.", {
                    "duration": time.time() - start_time,
                    "error": "No test functions found"
                }
            
            # Check for module sanity
            if not tests.strip().startswith("#[cfg(test)]"):
                tests = f"#[cfg(test)]\nmod tests {{\n    use super::*;\n\n    {tests}\n}}"
                self._log(f"Fixed test module wrapper", "yellow")
            
            # Combine the code with proper formatting and tests
            combined_code = f"{prelude}{main_fn}{declaration}\n{implementation}\n\n{tests}"
            
            # Clean up test failures that might cause compiler errors
            # Make sure braces are balanced in the test code
            opening_braces = combined_code.count('{')
            closing_braces = combined_code.count('}')
            if opening_braces > closing_braces:
                # Add missing closing braces
                combined_code += '\n' + '}' * (opening_braces - closing_braces)
                self._log(f"Fixed {opening_braces - closing_braces} unbalanced braces", "yellow")
            
            # Print the code and tests being run
            self._log(f"\nRUNNING TESTS ON CODE:", "cyan", attrs=["bold"])
            self._log("-" * 40)
            self._log(combined_code)
            self._log("-" * 40)
            
            # Write to file
            with open(file_path, "w") as f:
                f.write(combined_code)
            
            # Log what we're testing
            self._log(f"Testing: {file_path}")
            
            # First compile to check for syntax errors
            compile_process = subprocess.run(
                ["cargo", "check", "--bin", file_prefix],
                cwd=self.rust_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout  # Use instance timeout
            )
            
            if compile_process.returncode != 0:
                self._log(f"Compilation failed for tests:", "red", attrs=["bold"])
                self._log(compile_process.stderr)
                return False, f"Test compilation failed: {compile_process.stderr}", {
                    "duration": time.time() - start_time,
                    "error": "Test compilation failed",
                    "stderr": compile_process.stderr,
                    "combined_code": combined_code,
                    "file_path": file_path
                }
        except subprocess.TimeoutExpired:
            self._log(f"Compilation timed out for {file_prefix}.rs after {self.timeout} seconds", "red", attrs=["bold"])
            return False, f"Test compilation timed out after {self.timeout} seconds.", {
                "duration": time.time() - start_time,
                "error": "Test compilation timed out",
                "file_path": file_path,
                "combined_code": combined_code
            }
        
        # Run cargo test with a timeout
        try:
            process = subprocess.run(
                ["cargo", "test", "--bin", file_prefix],
                cwd=self.rust_dir,  # Use main directory
                capture_output=True,
                text=True,
                timeout=self.timeout  # Use instance timeout 
            )
        except subprocess.TimeoutExpired:
            self._log(f"Test execution timed out for {file_prefix}.rs after {self.timeout} seconds", "red", attrs=["bold"])
            return False, f"Tests timed out after {self.timeout} seconds. This may indicate an infinite loop or deadlock in the implementation.", {
                "duration": time.time() - start_time,
                "error": "Test execution timed out",
                "file_path": file_path,
                "combined_code": combined_code
            }
        
        # Collect details
        details = {
            "duration": time.time() - start_time,
            "command": f"cargo test --bin {file_prefix}",
            "return_code": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "combined_code": combined_code,
            "file_path": file_path
        }
        
        if process.returncode != 0:
            # Tests failed
            self._log(f"Tests failed for {file_prefix}.rs", "red", attrs=["bold"])
            # Extract relevant test failure info
            test_output = process.stdout
            if "test result: FAILED" in test_output:
                self._log(f"Test failures:", "red")
                # Extract the failing test info for clarity
                failing_tests = re.findall(r'test (.*?) \.\.\. FAILED', test_output)
                for test in failing_tests:
                    self._log(f"  - {test}", "red")
            return False, process.stdout + "\n" + process.stderr, details
        
        self._log(f"Tests passed for {file_prefix}.rs", "green", attrs=["bold"])
        # Extract test summary for clarity
        test_count_match = re.search(r'running (\d+) tests', process.stdout)
        test_count = test_count_match.group(1) if test_count_match else "?"
        self._log(f"All {test_count} tests passed!", "green")
        
        return True, "All tests passed.", details
    
    def generate_detailed_feedback(self, prompt: str, declaration: str, implementation: str, tests: str, test_output: str) -> Tuple[str, Dict[str, Any]]:
        """Generate detailed feedback about why tests failed"""
        start_time = time.time()
        
        # First do some basic analysis of the test failures
        self._log(f"\nANALYZING TEST FAILURES:", "cyan", attrs=["bold"])
        
        # Extract failing tests
        failing_tests = re.findall(r'test (.*?) \.\.\. FAILED', test_output)
        if failing_tests:
            self._log(f"Detected {len(failing_tests)} failing tests:")
            for test in failing_tests:
                self._log(f"  - {test}")
                
            # Try to extract failure messages
            failure_details = re.findall(r'thread .* panicked at (.*?)(,|\n)', test_output)
            if failure_details:
                self._log("Failure messages:")
                for detail in failure_details:
                    self._log(f"  - {detail[0]}")
        
        feedback_prompt = f"""
As a Rust expert, analyze why these tests are failing for the given problem:

Problem description:
{prompt}

Implementation:
```rust
{declaration}

{implementation}
```

Test code:
```rust
{tests}
```

Test output (showing failures):
```
{test_output}
```

Please provide a detailed analysis of the problems in the implementation:
1. Identify which test cases are failing and why they're failing (expected vs. actual behavior)
2. Point out the specific parts of the code that have logical errors
3. Explain clearly how the code should be fixed
4. For each bug, describe both the cause and the solution

Your feedback should be specific and focus on fixing the implementation, not the tests.
"""
        feedback = self.model.generate_code(feedback_prompt, n=1)[0]
        
        details = {
            "duration": time.time() - start_time,
            "feedback": feedback,
            "failing_tests": failing_tests
        }
        
        # Print a preview of the feedback
        self._log(f"\nGENERATED FEEDBACK:", "cyan", attrs=["bold"])
        self._log("-" * 40)
        feedback_preview = feedback.split('\n')
        if len(feedback_preview) > 10:
            self._log('\n'.join(feedback_preview[:10] + ['...']))
        else:
            self._log(feedback)
        self._log("-" * 40)
        
        return feedback, details
    
    def cleanup(self):
        """Cleanup temporary files if needed"""
        if hasattr(self, 'rust_bin') and os.path.exists(self.rust_bin):
            try:
                # Only remove the files created for this thread and sample
                if self.sample_idx is not None:
                    # Pattern: sample_<idx>_<thread_id>.rs
                    thread_files = [f for f in os.listdir(self.rust_bin) 
                                  if f.startswith(f"sample_{self.sample_idx}_{self.thread_id}")]
                else:
                    # Fallback to the old pattern if sample_idx is not provided
                    thread_files = [f for f in os.listdir(self.rust_bin) 
                                  if f.endswith(f"_{self.thread_id}.rs")]
                
                for file_name in thread_files:
                    file_path = os.path.join(self.rust_bin, file_name)
                    os.remove(file_path)
                    self._log(f"Removed temporary file: {file_path}")
            except Exception as e:
                self._log(f"Warning: Failed to clean up resources for thread {self.thread_id}: {str(e)}", "yellow")
                # Don't raise the exception, just log it

    @property
    def model_name(self) -> str:
        return f"Reviewer({self.model.model_name})"
