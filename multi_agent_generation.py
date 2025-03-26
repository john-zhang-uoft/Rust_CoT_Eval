#!/usr/bin/env python3
"""
Multi-agent code generation system with code review feedback
"""

import os
import tempfile
import subprocess
import re
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm
import jsonlines
import time
import termcolor

from code_generation_models import CodeGenerationModel, OpenAIChatModel, LambdaLabsModel
from content_parser import ContentParser, ParseError

# Default timeout for subprocess calls (in seconds)
DEFAULT_TIMEOUT = 60  # 1 minute timeout for compilation and test execution

class CodeGeneratorAgent(CodeGenerationModel):
    """Agent that generates code based on prompts"""
    
    def __init__(self, model: CodeGenerationModel):
        self.model = model

    def generate_code(self, prompt: str, n: int = 1, **kwargs) -> List[str]:
        """
        Generate code based on the prompt
        
        Args:
            prompt: The prompt to generate code from
            n: Number of completions to generate
            **kwargs: Additional arguments including entry_point, feedback, and code for refinement
            
        Returns:
            List containing the generated code completion(s)
        """
        entry_point = kwargs.get("entry_point", "solution")
        feedback = kwargs.get("feedback", None)
        code = kwargs.get("code", None)
        
        if feedback and code:
            # Handle refinement case
            refinement_prompt = f"""
You are a Rust programming expert. Your task is to fix the following code based on the feedback provided.

Original Problem:
{prompt}

Current Implementation:
```rust
{code}
```

Feedback from Code Review:
{feedback}

Please provide an improved version of the code that addresses all the issues mentioned in the feedback.
Only output the complete, corrected code with no additional explanation.
The function name should remain '{entry_point}'.
"""
            return self.model.generate_code(refinement_prompt, n=n)
        else:
            # Handle initial generation case
            return self.model.generate_code(prompt, n=n)
            
    @property
    def model_name(self) -> str:
        return f"Generator({self.model.model_name})"


class CodeReviewerAgent(CodeGenerationModel):
    """Agent that reviews code, tests it, and provides feedback"""
    
    def __init__(self, model: CodeGenerationModel, language: str = "rust", timeout: int = DEFAULT_TIMEOUT, thread_id: Optional[int] = None):
        self.model = model
        self.language = language
        self.timeout = timeout
        self.thread_id = thread_id or os.getpid()  # Use process ID as default thread ID
        self._last_details = {}
        self.parser = ContentParser()  # Initialize ContentParser for parsing code
        
        # Get absolute path to project root (where this script is located)
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Set up paths for Rust project
        if language == "rust":
            self.rust_dir = os.path.join(self.project_root, "rust")
            
            # Create a unique directory for this thread to avoid conflicts
            self.thread_rust_dir = os.path.join(self.rust_dir, f"thread_{self.thread_id}")
            self.rust_src = os.path.join(self.thread_rust_dir, "src")
            self.rust_bin = os.path.join(self.rust_src, "bin")
            
            # Create src/bin directory if it doesn't exist
            os.makedirs(self.rust_bin, exist_ok=True)
            
            # Check if base Cargo.toml exists
            base_cargo_toml = os.path.join(self.rust_dir, "Cargo.toml")
            if not os.path.exists(base_cargo_toml):
                raise FileNotFoundError(f"Cargo.toml not found in {self.rust_dir}")
            
            # Copy Cargo.toml to thread directory if it doesn't exist
            thread_cargo_toml = os.path.join(self.thread_rust_dir, "Cargo.toml")
            if not os.path.exists(thread_cargo_toml):
                import shutil
                shutil.copy2(base_cargo_toml, thread_cargo_toml)
                
            print(f"Using Rust project at: {self.thread_rust_dir}")
            
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
        print(termcolor.colored(f"\nATTEMPTING TO PARSE CODE for {entry_point}:", "cyan", attrs=["bold"]))
        
        # Check if the raw code directly contains the entry point function
        if f"fn {entry_point}" in raw_code:
            print(f"Raw code contains function definition: 'fn {entry_point}'")
        else:
            print(f"WARNING: Raw code does not directly contain 'fn {entry_point}' - this may cause parsing issues")
        
        # Check for Markdown code blocks
        code_blocks = re.findall(r'```(?:rust)?\s*(.*?)\s*```', raw_code, re.DOTALL)
        if code_blocks:
            print(f"Found {len(code_blocks)} Markdown code blocks in raw code")
            # Print largest code block preview
            largest_block = max(code_blocks, key=len)
            preview = largest_block.split('\n')[:5]
            print(f"Largest code block preview ({len(largest_block)} chars):")
            print('\n'.join(preview))
            if f"fn {entry_point}" in largest_block:
                print(f"  - Contains function definition: 'fn {entry_point}'")
        else:
            print("No Markdown code blocks found in raw code")
        
        # Try to use ContentParser
        try:
            print(f"Calling ContentParser with entry_point='{entry_point}'")
            parsed_code = self.parser(prompt, raw_code, entry_point)
            print(termcolor.colored(f"\nPARSE SUCCESSFUL for {entry_point}:", "green", attrs=["bold"]))
            print("-" * 40)
            print(parsed_code)
            print("-" * 40)
            return parsed_code
        except ParseError as e:
            print(termcolor.colored(f"\nPARSE ERROR: {str(e)}", "red", attrs=["bold"]))
            
            # Fallback: Direct extraction of function
            print("Attempting fallback extraction...")
            pattern = fr'fn\s+{re.escape(entry_point)}[^{{]*{{[^{{]*(?:{{[^{{]*}}[^{{]*)*}}'
            match = re.search(pattern, raw_code, re.DOTALL)
            if match:
                extracted_code = match.group(0)
                print(termcolor.colored(f"\nFALLBACK EXTRACTION SUCCESSFUL:", "yellow", attrs=["bold"]))
                print("-" * 40)
                print(extracted_code)
                print("-" * 40)
                return extracted_code
            
            # If everything fails, try to extract from code blocks
            if code_blocks:
                for block in sorted(code_blocks, key=len, reverse=True):
                    if f"fn {entry_point}" in block:
                        print(termcolor.colored(f"\nEXTRACTING FROM CODE BLOCK:", "yellow", attrs=["bold"]))
                        print("-" * 40)
                        print(block)
                        print("-" * 40)
                        return block
            
            # Last resort: return raw code
            print("All extraction attempts failed. Returning raw code.")
            return raw_code
            
    def generate_code(self, prompt: str, n: int = 1, **kwargs) -> List[str]:
        """
        Review the provided code implementation and return a list with the feedback
        
        Args:
            prompt: The problem description/prompt
            n: Number of feedback to generate (ignored, always returns 1)
            **kwargs: Additional arguments including declaration, implementation, and entry_point
            
        Returns:
            List containing the feedback
        """
        # Extract parameters from kwargs
        declaration = kwargs.get("declaration", "")
        raw_implementation = kwargs.get("implementation", "")
        entry_point = kwargs.get("entry_point", "solution")
        
        # Parse the implementation using ContentParser
        implementation = self._parse_code(raw_implementation, prompt, entry_point)
        
        # Review the code
        success, feedback, details = self.review(prompt, declaration, implementation, entry_point)
        
        # Store details for later access, ensuring success is directly accessible
        self._last_details = details
        self._last_details["success"] = success
        
        # Return feedback as a list with one item
        return [feedback]
    
    def get_last_details(self) -> Dict[str, Any]:
        """Get the details from the last review"""
        return self._last_details
    
    def review(self, prompt: str, declaration: str, implementation: str, entry_point: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Review the provided code implementation
        
        Returns:
            Tuple of (success, feedback, details)
        """
        if self.language == "rust":
            return self.review_rust(prompt, declaration, implementation, entry_point)
        else:
            return False, f"Language {self.language} not supported for review", {}
    
    def review_rust(self, prompt: str, declaration: str, implementation: str, entry_point: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Review Rust code"""
        details = {}
        
        # Step 1: Check if the code compiles
        compiles, compile_feedback, compile_details = self.check_compilation(declaration, implementation, entry_point)
        details["compilation"] = compile_details
        
        if not compiles:
            return False, f"Compilation failed: {compile_feedback}", details
        
        # Step 2: Generate and run tests
        tests_generated, test_code, test_gen_details = self.generate_tests(prompt, declaration, implementation, entry_point)
        details["test_generation"] = test_gen_details
        
        if not tests_generated:
            return False, f"Could not generate tests: {test_code}", details
        
        # Step 3: Run the tests
        tests_pass, test_output, test_details = self.run_tests(declaration, implementation, test_code, entry_point)
        details["test_execution"] = test_details
        
        if not tests_pass:
            # Step 4: If tests fail, generate more detailed feedback
            detailed_feedback, analysis_details = self.generate_detailed_feedback(
                prompt,declaration, implementation, test_code, test_output
            )
            details["analysis"] = analysis_details
            return False, detailed_feedback, details
        
        return True, "Code looks good. All tests passed.", details
    
    def check_compilation(self, declaration: str, implementation: str, entry_point: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Check if the code compiles"""
        start_time = time.time()
        
        # Create a unique file name based on the entry point
        file_prefix = f"{entry_point.lower().replace('/', '_')}_{self.thread_id}"
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
        print(termcolor.colored(f"\nCOMPILING CODE:", "cyan", attrs=["bold"]))
        print("-" * 40)
        print(combined_code)
        print("-" * 40)
        
        # Write to file
        with open(file_path, "w") as f:
            f.write(combined_code)
        
        # Log what we're compiling
        print(f"Compiling: {file_path}")
        
        # Try to compile with warnings about unused imports suppressed
        try:
            process = subprocess.run(
                ["cargo", "check", "--bin", file_prefix],
                cwd=self.thread_rust_dir,  # Use thread-specific directory
                capture_output=True,
                text=True,
                timeout=self.timeout  # Use instance timeout
            )
        except subprocess.TimeoutExpired:
            print(termcolor.colored(f"Compilation timed out for {file_prefix}.rs after {self.timeout} seconds", "red", attrs=["bold"]))
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
Analyze this Rust compilation error and provide a clear, concise explanation of what's wrong and how to fix it:

Code:
rust
{combined_code}


Compilation error:
{process.stderr}


Provide a brief explanation of what's wrong and how to fix it. Focus on issues in the implementation, not in the Cargo.toml or other project configuration.
"""
            start_time = time.time()
            error_analysis = self.model.generate_code(error_prompt, n=1)[0]
            details["analysis_duration"] = time.time() - start_time
            details["error_analysis"] = error_analysis
            
            return False, error_analysis, details
        
        # MODIFIED: Log successful compilation details
        print(termcolor.colored(f"Compilation succeeded for {file_prefix}.rs in {details['duration']:.2f} seconds.", "green"))
        return True, "Code compiles successfully.", details
    
    def generate_tests(self, prompt: str, declaration: str, implementation: str, entry_point: str, implementation_visible: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate unit tests for the code"""
        start_time = time.time()
        combined_code = declaration + "\n" + implementation
        
        # Print the implementation we're generating tests for
        print(termcolor.colored(f"\nGENERATING TESTS FOR IMPLEMENTATION:", "cyan", attrs=["bold"]))
        print("-" * 40)
        print(combined_code)
        print("-" * 40)
        
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
        print(termcolor.colored(f"\nRAW TEST CODE:", "cyan", attrs=["bold"]))
        print("-" * 40)
        print(raw_test_code)
        print("-" * 40)
        
        # Extract test module directly instead of using ContentParser
        # First try to extract from Markdown code blocks
        code_blocks = re.findall(r'```(?:rust)?\s*(.*?)\s*```', raw_test_code, re.DOTALL)
        if code_blocks:
            # Use the largest code block
            test_code = max(code_blocks, key=len)
            print(termcolor.colored(f"\nEXTRACTED TEST CODE FROM CODE BLOCKS:", "cyan", attrs=["bold"]))
            print("-" * 40)
            print(test_code)
            print("-" * 40)
        else:
            # Try to extract #[cfg(test)] module directly
            test_module_match = re.search(r'(#\[cfg\(test\)].*)', raw_test_code, re.DOTALL)
            if test_module_match:
                test_code = test_module_match.group(1)
                print(termcolor.colored(f"\nEXTRACTED TEST CODE FROM #[cfg(test)]:", "cyan", attrs=["bold"]))
                print("-" * 40)
                print(test_code)
                print("-" * 40)
            else:
                # Just use the raw response, but remove any non-code content
                # Remove any "Here's the test code:" prefixes
                test_code = re.sub(r'^.*?(\bmod tests\b|\buse super::\*)', r'\1', raw_test_code, flags=re.DOTALL)
                print(termcolor.colored(f"\nEXTRACTED TEST CODE FROM RAW RESPONSE:", "cyan", attrs=["bold"]))
                print("-" * 40)
                print(test_code)
                print("-" * 40)
        
        # Ensure the test code starts with #[cfg(test)]
        if not test_code.strip().startswith("#[cfg(test)]"):
            # If the test module doesn't start with the proper module definition, add it
            if "mod test" in test_code or "mod tests" in test_code:
                # If it has a module but no cfg attribute
                test_code = "#[cfg(test)]\n" + test_code
            else:
                # Add full wrapper
                test_code = f"#[cfg(test)]\nmod tests {{\n    use super::*;\n\n    {test_code}\n}}"
                
            print(termcolor.colored(f"\nADDED #[cfg(test)] WRAPPER:", "yellow", attrs=["bold"]))
            print("-" * 40)
            print(test_code)
            print("-" * 40)
        
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
            print(termcolor.colored(f"\nFIXED UNBALANCED BRACES:", "yellow", attrs=["bold"]))
            print("-" * 40)
            print(test_code)
            print("-" * 40)
        
        details["test_module"] = test_code
        
        # Print only a short preview of the tests to avoid overwhelming output
        test_preview = test_code.split("\n")
        if len(test_preview) > 5:
            test_preview = test_preview[:5] + ["..."]
        test_preview = "\n".join(test_preview)
        print(termcolor.colored(f"\nFINAL TEST CODE (PREVIEW):\n{test_preview}", "green", attrs=["bold"]))
        
        return True, test_code, details
    
    def run_tests(self, declaration: str, implementation: str, tests: str, entry_point: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Run the tests and check if they pass"""
        start_time = time.time()
        
        # Create a unique file name based on the entry point
        file_prefix = f"{entry_point.lower().replace('/', '_')}_{self.thread_id}"
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
            print(termcolor.colored(f"Fixed test module wrapper", "yellow"))
        
        # Combine the code with proper formatting and tests
        combined_code = f"{prelude}{main_fn}{declaration}\n{implementation}\n\n{tests}"
        
        # Clean up test failures that might cause compiler errors
        # Make sure braces are balanced in the test code
        opening_braces = combined_code.count('{')
        closing_braces = combined_code.count('}')
        if opening_braces > closing_braces:
            # Add missing closing braces
            combined_code += '\n' + '}' * (opening_braces - closing_braces)
            print(termcolor.colored(f"Fixed {opening_braces - closing_braces} unbalanced braces", "yellow"))
        
        # Print the code and tests being run
        print(termcolor.colored(f"\nRUNNING TESTS ON CODE:", "cyan", attrs=["bold"]))
        print("-" * 40)
        print(combined_code)
        print("-" * 40)
        
        # Write to file
        with open(file_path, "w") as f:
            f.write(combined_code)
        
        # Log what we're testing
        print(f"Testing: {file_path}")
        
        # First compile to check for syntax errors
        try:
            compile_process = subprocess.run(
                ["cargo", "check", "--bin", file_prefix],
                cwd=self.thread_rust_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout  # Use instance timeout
            )
            
            if compile_process.returncode != 0:
                print(termcolor.colored(f"Compilation failed for tests:", "red", attrs=["bold"]))
                print(compile_process.stderr)
                return False, f"Test compilation failed: {compile_process.stderr}", {
                    "duration": time.time() - start_time,
                    "error": "Test compilation failed",
                    "stderr": compile_process.stderr,
                    "combined_code": combined_code,
                    "file_path": file_path
                }
        except subprocess.TimeoutExpired:
            print(termcolor.colored(f"Compilation timed out for {file_prefix}.rs after {self.timeout} seconds", "red", attrs=["bold"]))
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
                cwd=self.thread_rust_dir,  # Use thread-specific directory
                capture_output=True,
                text=True,
                timeout=self.timeout  # Use instance timeout 
            )
        except subprocess.TimeoutExpired:
            print(termcolor.colored(f"Test execution timed out for {file_prefix}.rs after {self.timeout} seconds", "red", attrs=["bold"]))
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
            print(termcolor.colored(f"Tests failed for {file_prefix}.rs", "red", attrs=["bold"]))
            # Extract relevant test failure info
            test_output = process.stdout
            if "test result: FAILED" in test_output:
                print(termcolor.colored(f"Test failures:", "red"))
                # Extract the failing test info for clarity
                failing_tests = re.findall(r'test (.*?) \.\.\. FAILED', test_output)
                for test in failing_tests:
                    print(termcolor.colored(f"  - {test}", "red"))
            return False, process.stdout + "\n" + process.stderr, details
        
        print(termcolor.colored(f"Tests passed for {file_prefix}.rs", "green", attrs=["bold"]))
        # Extract test summary for clarity
        test_count_match = re.search(r'running (\d+) tests', process.stdout)
        test_count = test_count_match.group(1) if test_count_match else "?"
        print(termcolor.colored(f"All {test_count} tests passed!", "green"))
        
        return True, "All tests passed.", details
    
    def generate_detailed_feedback(self, prompt: str, declaration: str, implementation: str, tests: str, test_output: str) -> Tuple[str, Dict[str, Any]]:
        """Generate detailed feedback about why tests failed"""
        start_time = time.time()
        
        # First do some basic analysis of the test failures
        print(termcolor.colored(f"\nANALYZING TEST FAILURES:", "cyan", attrs=["bold"]))
        
        # Extract failing tests
        failing_tests = re.findall(r'test (.*?) \.\.\. FAILED', test_output)
        if failing_tests:
            print(f"Detected {len(failing_tests)} failing tests:")
            for test in failing_tests:
                print(f"  - {test}")
                
            # Try to extract failure messages
            failure_details = re.findall(r'thread .* panicked at (.*?)(,|\n)', test_output)
            if failure_details:
                print("Failure messages:")
                for detail in failure_details:
                    print(f"  - {detail[0]}")
        
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
        print(termcolor.colored(f"\nGENERATED FEEDBACK:", "cyan", attrs=["bold"]))
        print("-" * 40)
        feedback_preview = feedback.split('\n')
        if len(feedback_preview) > 10:
            print('\n'.join(feedback_preview[:10] + ['...']))
        else:
            print(feedback)
        print("-" * 40)
        
        return feedback, details
    
    def cleanup(self):
        """Cleanup temporary files if needed"""
        if self.language == "rust" and hasattr(self, 'thread_rust_dir') and os.path.exists(self.thread_rust_dir):
            import shutil
            try:
                shutil.rmtree(self.thread_rust_dir)
                print(f"Cleaned up temporary directory: {self.thread_rust_dir}")
            except Exception as e:
                print(f"Warning: Failed to clean up {self.thread_rust_dir}: {str(e)}")
                # Don't raise the exception, just log it

    @property
    def model_name(self) -> str:
        return f"Reviewer({self.model.model_name})"


def create_model(model_type: str, model_name: str, temperature: float, top_p: float) -> CodeGenerationModel:
    """Create a code generation model"""
    if model_type.lower() == 'openai':
        return OpenAIChatModel(model=model_name, temperature=temperature, top_p=top_p)
    elif model_type.lower() == 'lambdalabs':
        return LambdaLabsModel(model=model_name, temperature=temperature, top_p=top_p)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_prompt_from_sample(sample: Dict[str, Any], language: str = "rust") -> str:
    """Get the prompt from a sample"""
    if language == "rust":
        # Add the main function for Rust
        main = "fn main(){}\n" if "fn main()" not in sample["declaration"] else ""
        declaration = sample["declaration"]
        
        # Return the instruction with the declaration
        return f"{sample['instruction']}\n\nYou should implement the function according to this declaration:\n{main}{declaration}"
    else:
        return sample['instruction']


class MultiAgentModel(CodeGenerationModel):
    """CodeGenerationModel implementation using multi-agent approach with code generation and review"""
    
    # Possible exit reasons
    EXIT_REASON_SUCCESS = "success"  # All tests passed
    EXIT_REASON_MAX_ITERATIONS = "max_iterations"  # Reached maximum iterations
    EXIT_REASON_NO_CHANGE = "no_change"  # Code didn't change after refinement
    EXIT_REASON_ERROR = "error"  # Error during refinement
    
    def __init__(
        self,
        gen_model: CodeGenerationModel,
        review_model: CodeGenerationModel,
        language: str = "rust",
        max_iterations: int = 3,
        verbose: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
        thread_id: Optional[int] = None
    ):
        """
        Initialize a multi-agent model
        
        Args:
            gen_model: Model for code generation
            review_model: Model for code review
            language: Programming language
            max_iterations: Maximum number of refinement iterations
            verbose: Whether to print detailed logs
            timeout: Timeout for subprocess calls in seconds
            thread_id: Optional thread ID to use for this model
        """
        self._gen_model = gen_model
        self._review_model = review_model
        self._language = language
        self._max_iterations = max_iterations
        self._verbose = verbose
        self._timeout = timeout
        self._thread_id = thread_id
        
        # Initialize agents
        self._generator = CodeGeneratorAgent(gen_model)
        self._reviewer = CodeReviewerAgent(review_model, language=language, timeout=timeout, thread_id=thread_id)
    
    def generate_code(self, prompt: str, n: int = 1, declaration: str = None, entry_point: str = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Generate code using the multi-agent approach
        
        Args:
            prompt: The prompt to generate code from
            n: Number of completions to generate (not fully supported, will generate one high-quality solution)
            declaration: The function declaration (if None, will try to parse from prompt)
            entry_point: The function name / entry point (if None, will try to parse from declaration)
            
        Returns:
            Tuple containing:
                - List of generated code completions
                - Dictionary with generation details including exit_reason, iterations, success status
        """
        # Parse the prompt to extract declaration and entry point if not provided
        if declaration is None or entry_point is None:
            parsed_decl, parsed_entry = self._parse_prompt(prompt)
            declaration = declaration or parsed_decl
            entry_point = entry_point or parsed_entry
        
        if self._verbose or True:  # Always print these, regardless of verbose flag
            print("\n" + termcolor.colored("GENERATING INITIAL CODE...", "cyan", attrs=["bold"]))
            print(f"Entry point: {entry_point}")
            print(f"Declaration: {declaration}")
            
        # Generate the initial code - raw output from the model
        raw_codes = self._generator.generate_code(prompt, n=1, entry_point=entry_point)
        raw_code = raw_codes[0] if raw_codes else ""
        
        if self._verbose or True:  # Always print these, regardless of verbose flag
            print("\n" + termcolor.colored("GENERATOR RAW OUTPUT:", "yellow", attrs=["bold"]))
            print("-" * 40)
            print(raw_code)
            print("-" * 40)
        
        # Refinement loop
        success = False
        current_raw_code = raw_code
        exit_reason = self.EXIT_REASON_MAX_ITERATIONS  # Default exit reason
        iterations_data = []  # Track data for each iteration
        
        for i in range(self._max_iterations):
            if self._verbose or True:  # Always print these, regardless of verbose flag
                print(f"\n" + termcolor.colored(f"REVIEW ITERATION {i+1}/{self._max_iterations}...", "cyan", attrs=["bold"]))
                
            # Review the code
            try:
                # The reviewer will parse the raw code and compile/test it
                print(termcolor.colored(f"\nSending raw code to reviewer for parsing and testing:", "cyan", attrs=["bold"]))
                
                feedback_list = self._reviewer.generate_code(
                    prompt, 
                    n=1, 
                    declaration=declaration, 
                    implementation=current_raw_code, 
                    entry_point=entry_point
                )
                feedback = feedback_list[0] if feedback_list else ""
                review_details = self._reviewer.get_last_details()
                
                # Get success value from review result
                try:
                    # Try accessing the success value from the review method result
                    success_value = review_details.get("success")
                    if success_value is not None:
                        success = success_value
                    else:
                        # Check if it's in test_execution or a different path
                        test_execution = review_details.get("test_execution", {})
                        compile_status = review_details.get("compilation", {})
                        
                        # Consider success if both compilation and tests passed
                        success = (
                            test_execution.get("return_code", 1) == 0 and 
                            compile_status.get("return_code", 1) == 0
                        )
                except Exception:
                    # Fallback for safety
                    success = False
                
                # Store iteration data
                iteration_data = {
                    "iteration": i,
                    "raw_code": current_raw_code,
                    "feedback": feedback,
                    "review_details": review_details,
                    "success": success
                }
                iterations_data.append(iteration_data)
                
            except Exception as e:
                # Handle any unexpected errors during review (including subprocess timeouts)
                error_msg = f"Error during code review: {str(e)}"
                print(termcolor.colored(error_msg, "red", attrs=["bold"]))
                
                if self._verbose:
                    import traceback
                    traceback.print_exc()
                
                # Create generic feedback about the error
                feedback = f"There was an error during code review: {str(e)}. The code might have timing or resource issues. Please review the implementation for possible infinite loops, resource leaks, or excessive computation."
                review_details = {"error": str(e)}
                success = False
                
                # Store iteration data for error case
                iteration_data = {
                    "iteration": i,
                    "raw_code": current_raw_code,
                    "feedback": feedback,
                    "review_details": review_details,
                    "success": False,
                    "error": str(e)
                }
                iterations_data.append(iteration_data)
                
                # Set exit reason to error
                exit_reason = self.EXIT_REASON_ERROR
            
            if self._verbose or True:  # Always print these, regardless of verbose flag
                print("\n" + termcolor.colored("REVIEWER OUTPUT:", "yellow", attrs=["bold"]))
                print("-" * 40)
                print(feedback)
                print("-" * 40)
                
                # Print compilation output if available
                if "compilation" in review_details:
                    comp_details = review_details["compilation"]
                    if comp_details.get("return_code", 0) != 0:
                        print("\n" + termcolor.colored("COMPILATION ERROR:", "red", attrs=["bold"]))
                        print("-" * 40)
                        print(comp_details.get("stderr", "No error output available"))
                        print("-" * 40)
                
                # Print test output if available
                if "test_execution" in review_details:
                    test_details = review_details["test_execution"]
                    if test_details.get("return_code", 0) != 0:
                        print("\n" + termcolor.colored("TEST FAILURE:", "red", attrs=["bold"]))
                        print("-" * 40)
                        print(test_details.get("stdout", "") + "\n" + test_details.get("stderr", ""))
                        print("-" * 40)
            
            if success:
                if self._verbose or True:  # Always print these, regardless of verbose flag
                    print("\n" + termcolor.colored("CODE PASSED ALL REVIEWS!", "green", attrs=["bold"]))
                exit_reason = self.EXIT_REASON_SUCCESS
                break
                
            if i == self._max_iterations - 1:
                # Last iteration, don't refine further
                break
                
            if self._verbose or True:  # Always print these, regardless of verbose flag
                print("\n" + termcolor.colored("REFINING CODE...", "cyan", attrs=["bold"]))
                
            # Refine the code based on feedback - get new raw code
            try:
                refined_codes = self._generator.generate_code(
                    prompt, 
                    n=1, 
                    entry_point=entry_point, 
                    code=current_raw_code,  # Use the current raw code 
                    feedback=feedback
                )
                
                new_raw_code = refined_codes[0] if refined_codes else ""
                
                if self._verbose or True:  # Always print these, regardless of verbose flag
                    print("\n" + termcolor.colored("REFINED CODE:", "green", attrs=["bold"]))
                    print("-" * 40)
                    print(new_raw_code)
                    print("-" * 40)
                
                # Check if the raw code has changed
                if new_raw_code == current_raw_code:
                    if self._verbose or True:  # Always print these, regardless of verbose flag
                        print("\n" + termcolor.colored("CODE DIDN'T CHANGE AFTER REFINEMENT. STOPPING ITERATIONS.", "red", attrs=["bold"]))
                    exit_reason = self.EXIT_REASON_NO_CHANGE
                    break
                
                current_raw_code = new_raw_code
            except Exception as e:
                # Handle any unexpected errors during refinement
                error_msg = f"Error during code refinement: {str(e)}"
                print(termcolor.colored(error_msg, "red", attrs=["bold"]))
                
                if self._verbose:
                    import traceback
                    traceback.print_exc()
                
                # Set exit reason to error
                exit_reason = self.EXIT_REASON_ERROR
                
                # Stop the refinement loop on error
                break
        
        # Get the final parsed implementation from the reviewer
        print(termcolor.colored(f"\nGETTING FINAL PARSED CODE:", "cyan", attrs=["bold"]))
        final_code = self._reviewer._parse_code(current_raw_code, prompt, entry_point)
        
        if self._verbose or True:  # Always print these, regardless of verbose flag
            print("\n" + termcolor.colored("FINAL PARSED CODE:", "green", attrs=["bold"]))
            print("-" * 40)
            print(final_code)
            print("-" * 40)
            print(f"Exit reason: {exit_reason}")
        
        # Prepare generation details
        generation_details = {
            "exit_reason": exit_reason,
            "success": success,
            "iterations": len(iterations_data),
            "iterations_data": iterations_data,
            "final_parsed_code": final_code
        }
        
        # Return the final code and generation details
        return [current_raw_code] * n, generation_details
    
    def _parse_prompt(self, prompt: str) -> Tuple[str, str]:
        """
        Parse the prompt to extract declaration and entry point
        
        Args:
            prompt: The prompt text
            
        Returns:
            Tuple of (declaration, entry_point)
        """
        # Handle the Rust-specific prompt structure
        if self._language == "rust":
            # Look for any function name after fn main(){} that's not main
            fn_defs = re.findall(r'fn\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?\s*\(', prompt)
            fn_defs = [fn_name for fn_name in fn_defs if fn_name != "main"]
            
            if fn_defs:
                # Take the first non-main function as the entry point
                entry_point = fn_defs[0]
                
                # Extract everything after fn main(){} 
                main_match = re.search(r'fn\s+main\s*\(\s*\)\s*\{\s*\}(.*)', prompt, re.DOTALL)
                
                if main_match:
                    content_after_main = main_match.group(1).strip()
                    
                    # Find the complete function signature
                    fn_signature_match = re.search(fr'fn\s+{entry_point}(?:<[^>]*>)?\s*\([^{{]*', content_after_main, re.DOTALL)
                    
                    if fn_signature_match:
                        # Get the text between fn main(){} and the function signature (likely imports)
                        imports_match = re.search(fr'(.*?)fn\s+{entry_point}', content_after_main, re.DOTALL)
                        imports = imports_match.group(1).strip() if imports_match else ""
                        
                        # Complete declaration includes imports and function signature
                        declaration = imports + ("\n" if imports else "") + fn_signature_match.group(0)
                        
                        if self._verbose:
                            print(f"Parsed entry point: {entry_point}")
                        
                        return declaration, entry_point
                
                # Fallback: just try to get the function signature
                declaration_match = re.search(fr'(fn\s+{entry_point}(?:<[^>]*>)?\s*\([^{{]*)', prompt, re.DOTALL)
                declaration = declaration_match.group(1) if declaration_match else ""
                
                if self._verbose:
                    print(f"Fallback parsed entry point: {entry_point}")
                
                return declaration, entry_point
        
        # Generic approach for non-Rust languages or if Rust-specific parsing failed
        # Extract the declaration from the prompt
        declaration_match = re.search(r'declaration:\s*(.+?)(?=\n\n|\Z)', prompt, re.DOTALL | re.IGNORECASE)
        if not declaration_match:
            # Try an alternative pattern if the first one fails
            declaration_match = re.search(r'according to this declaration:\s*(.+?)(?=\n\n|\Z)', prompt, re.DOTALL | re.IGNORECASE)
        
        declaration = declaration_match.group(1).strip() if declaration_match else ""
        
        # Extract or guess the entry point
        entry_point_match = re.search(r'fn\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?\s*\(', declaration)
        if entry_point_match and entry_point_match.group(1) != "main":
            entry_point = entry_point_match.group(1)
        else:
            # Look in the whole prompt if not found in declaration
            entry_point_match = re.search(r'fn\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?\s*\(', prompt)
            entry_point = entry_point_match.group(1) if entry_point_match and entry_point_match.group(1) != "main" else "solution"
        
        if self._verbose:
            print(f"Generic parsed entry point: {entry_point}")
        
        return declaration, entry_point
    
    @property
    def model_name(self) -> str:
        return f"MultiAgent({self._gen_model.model_name}+{self._review_model.model_name})"


def create_multi_agent_model(
    gen_model_type: str,
    gen_model_name: str,
    review_model_type: str,
    review_model_name: str,
    temperature: float,
    top_p: float,
    language: str,
    max_iterations: int,
    verbose: bool,
    skip_review: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    thread_id: Optional[int] = None
) -> MultiAgentModel:
    """
    Create a MultiAgentModel with specified generation and review models
    
    Args:
        gen_model_type: Type of model for code generation ('openai', 'lambdalabs', etc.)
        gen_model_name: Name of the model for code generation
        review_model_type: Type of model for code review
        review_model_name: Name of the model for code review  
        temperature: Temperature parameter for generation
        top_p: Top-p parameter for generation
        language: Programming language
        max_iterations: Maximum number of refinement iterations
        verbose: Whether to print detailed logs
        skip_review: Whether to skip code review when cargo/compiler is not available
        timeout: Timeout for subprocess calls in seconds
        thread_id: Optional thread ID to use for this model
        
    Returns:
        A MultiAgentModel instance
    """
    # Create the models
    gen_model = create_model(
        gen_model_type,
        gen_model_name,
        temperature,
        top_p
    )
    
    review_model = create_model(
        review_model_type,
        review_model_name,
        temperature,
        top_p
    )
    
    if verbose:
        print(f"Using generator model: {gen_model.model_name}")
        print(f"Using reviewer model: {review_model.model_name}")
    
    # Check if cargo is available (for Rust)
    cargo_available = True
    if language == "rust" and not skip_review:
        try:
            # Try to create a reviewer to check if cargo is available
            CodeReviewerAgent(review_model, language=language, timeout=timeout, thread_id=thread_id)
        except FileNotFoundError as e:
            cargo_available = False
            print(f"Warning: {str(e)}")
            print("Running in code generation only mode (no compilation or testing)")
            if not skip_review:
                print("Use --skip_review to suppress this warning")
                max_iterations = 1  # No iterations if cargo not available
    
    # Create the multi-agent model
    multi_agent = MultiAgentModel(
        gen_model=gen_model,
        review_model=review_model,
        language=language,
        max_iterations=max_iterations if cargo_available else 1,
        verbose=verbose,
        timeout=timeout,
        thread_id=thread_id
    )
    
    if verbose:
        print(f"Using multi-agent model: {multi_agent.model_name}")
    return multi_agent


def run_multi_agent_generation(
    language: str,
    gen_model: CodeGenerationModel,
    review_model: CodeGenerationModel,
    samples: List[Dict[str, Any]],
    max_iterations: int = 3,
    verbose: bool = False,
    limit: Optional[int] = None,
    skip_review: bool = False
) -> List[Dict[str, Any]]:
    """
    Run the multi-agent generation process on the provided samples
    
    Args:
        language: Programming language
        gen_model: Model for code generation
        review_model: Model for code review
        samples: List of samples to process
        max_iterations: Maximum number of refinement iterations
        verbose: Whether to print detailed logs
        limit: Optional limit on the number of samples to process
        skip_review: Whether to skip code review
        
    Returns:
        List of processed samples with results
    """
    # Check if cargo is available (for Rust)
    cargo_available = True
    if language == "rust" and not skip_review:
        try:
            # Try to create a reviewer to check if cargo is available
            CodeReviewerAgent(review_model, language=language)
        except FileNotFoundError as e:
            cargo_available = False
            print(f"Warning: {str(e)}")
            print("Running in code generation only mode (no compilation or testing)")
    
    # Create multi-agent model
    multi_agent = MultiAgentModel(
        gen_model=gen_model,
        review_model=review_model,
        language=language,
        max_iterations=max_iterations if cargo_available else 1,  # No iterations if cargo not available
        verbose=verbose
    )
    
    processed_samples = []
    
    # Process each sample
    for idx, sample in enumerate(tqdm(samples)):
        if limit is not None and idx >= limit:
            break
            
        if verbose:
            print(f"\n{'-'*80}")
            print(f"Processing sample {idx+1}/{len(samples)}: {sample['task_id']}")
            print(f"{'-'*80}")
            
        # Get the prompt
        prompt = get_prompt_from_sample(sample, language)
        
        # Get declaration and entry_point directly from the sample
        declaration = sample["declaration"]
        entry_point = sample["entry_point"]
        
        # Generate initial code for tracking
        generator = CodeGeneratorAgent(gen_model)
        initial_raw_code = generator.generate_code(prompt, n=1, entry_point=entry_point)[0]
        
        # Generate code using multi-agent model with explicit declaration and entry_point
        final_code, generation_details = multi_agent.generate_code(
            prompt, 
            declaration=declaration, 
            entry_point=entry_point
        )
        final_code = final_code[0]  # Unpack the list
        
        # Store results with generation details
        result = {
            "task_id": sample["task_id"],
            "entry_point": entry_point,
            "declaration": declaration,
            "prompt": prompt,
            "final_code": final_code,
            "success": generation_details.get("success", False),
            "exit_reason": generation_details.get("exit_reason", "unknown"),
            "iterations": generation_details.get("iterations_data", []),
            "canonical_solution": sample.get("canonical_solution", "")
        }
        
        processed_samples.append(result)
    
    return processed_samples


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Multi-agent code generation with review feedback")
    parser.add_argument("--language", type=str, default="rust", choices=["rust"],
                        help="Programming language (only Rust supported currently)")
    parser.add_argument("--gen_model_type", type=str, default="lambdalabs",
                        choices=["openai", "lambdalabs"],
                        help="Model type for code generation")
    parser.add_argument("--gen_model_name", type=str, default="llama3.3-70b-instruct-fp8",
                        help="Model name for code generation")
    parser.add_argument("--review_model_type", type=str, default="lambdalabs",
                        choices=["openai", "lambdalabs"],
                        help="Model type for code review")
    parser.add_argument("--review_model_name", type=str, default="llama3.3-70b-instruct-fp8",
                        help="Model name for code review")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature parameter for generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p parameter for generation")
    parser.add_argument("--max_iterations", type=int, default=3,
                        help="Maximum number of refinement iterations")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file path (default: multi_agent_results_{language}.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of samples to process")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed logs")
    parser.add_argument("--skip_review", action="store_true", default=False,
                        help="Skip code review")
                        
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Override with environment variables if not set
    gen_model_type = os.getenv("GEN_MODEL_TYPE", args.gen_model_type)
    gen_model_name = os.getenv("GEN_MODEL_NAME", args.gen_model_name)
    review_model_type = os.getenv("REVIEW_MODEL_TYPE", args.review_model_type)
    review_model_name = os.getenv("REVIEW_MODEL_NAME", args.review_model_name)
    
    # Load the dataset
    samples = [s for s in load_dataset("bigcode/humanevalpack", args.language)["test"]]
    print(f"Loaded {len(samples)} samples from HumanEvalPack {args.language} dataset")
    
    # Create the multi-agent model
    multi_agent = create_multi_agent_model(
        gen_model_type=gen_model_type,
        gen_model_name=gen_model_name,
        review_model_type=review_model_type,
        review_model_name=review_model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        language=args.language,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        skip_review=args.skip_review
    )
    
    print(f"Using multi-agent model: {multi_agent.model_name}")
    
    # Process each sample
    results = []
    
    for idx, sample in enumerate(tqdm(samples)):
        if args.limit is not None and idx >= args.limit:
            break
            
        if args.verbose:
            print(f"\n{'-'*80}")
            print(f"Processing sample {idx+1}/{len(samples)}: {sample['task_id']}")
            print(f"{'-'*80}")
            
        # Get the prompt
        prompt = get_prompt_from_sample(sample, args.language)
        
        # Get declaration and entry_point directly from the sample
        declaration = sample["declaration"]
        entry_point = sample["entry_point"]
        
        # Generate code using multi-agent model with explicit declaration and entry_point
        final_code, generation_details = multi_agent.generate_code(
            prompt, 
            declaration=declaration, 
            entry_point=entry_point
        )
        final_code = final_code[0]  # Unpack the list
        
        # Store results with generation details
        result = {
            "task_id": sample["task_id"],
            "entry_point": entry_point,
            "declaration": declaration,
            "prompt": prompt,
            "final_code": final_code,
            "success": generation_details.get("success", False),
            "exit_reason": generation_details.get("exit_reason", "unknown"),
            "iterations": generation_details.get("iterations_data", []),
            "canonical_solution": sample.get("canonical_solution", "")
        }
        
        results.append(result)
    
    # Save results
    output_file = args.output_file or f"multi_agent_results_{args.language}.jsonl"
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(results)
        
    print(f"Results saved to {output_file}")
    
    # Calculate success rate
    success_count = sum(1 for r in results if r["success"])
    success_rate = success_count / len(results) if results else 0
    print(f"Success rate: {success_rate:.2%} ({success_count}/{len(results)})")
    
    # Print exit reason statistics
    exit_reasons = {}
    for r in results:
        exit_reason = r.get("exit_reason", "unknown")
        exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
    
    print("\nExit reason statistics:")
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count} ({count/len(results):.2%})")
