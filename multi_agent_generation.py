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


class CodeGeneratorAgent:
    """Agent that generates code based on prompts"""
    
    def __init__(self, model: CodeGenerationModel, parser: Optional[ContentParser] = None):
        self.model = model
        self.parser = parser

    def _remove_tests(self, code: str) -> str:
        """
        Remove test modules from the code to extract only the function implementation.
        It removes any block starting with #[cfg(test)].
        """
        # Remove any content starting with #[cfg(test)] till the end of the string
        cleaned_code = re.sub(r'(?s)#[\s]*\[cfg\(test\)\].*', '', code)
        return cleaned_code
    
    def generate(self, prompt: str, entry_point: str) -> Tuple[str, str]:
        """
        Generate code based on the prompt
        
        Returns:
            Tuple of (raw_response, parsed_code)
        """
        results = self.model.generate_code(prompt, n=1)
        raw_response = results[0] if results else ""
        
        # Remove tests from the generated code before parsing
        code_for_parsing = self._remove_tests(raw_response)
        
        # Parse the code if we have a parser
        if self.parser:
            try:
                parsed_code = self.parser(prompt, code_for_parsing, entry_point)
                return raw_response, parsed_code
            except ParseError:
                print(f"Warning: Failed to parse generated code for {entry_point}")
                return raw_response, raw_response
        
        return raw_response, raw_response
    
    def refine(self, prompt: str, code: str, feedback: str, entry_point: str) -> Tuple[str, str]:
        """
        Refine code based on feedback
        
        Returns:
            Tuple of (raw_response, parsed_code)
        """
        refinement_prompt = f"""
Your previous code:
{code}


Feedback from code review:
{feedback}

Please improve your code based on this feedback. Provide only the refined code, no explanations.
"""
        full_prompt = prompt + "\n\n" + refinement_prompt
        results = self.model.generate_code(full_prompt, n=1)
        raw_response = results[0] if results else code  # Return original code if refinement fails
        
        # Remove tests from the refined code before parsing
        code_for_parsing = self._remove_tests(raw_response)
        
        # Parse the code if we have a parser
        if self.parser:
            try:
                parsed_code = self.parser(prompt, code_for_parsing, entry_point)
                return raw_response, parsed_code
            except ParseError:
                print(f"Warning: Failed to parse refined code for {entry_point}")
                return raw_response, raw_response
        
        return raw_response, raw_response


class CodeReviewerAgent:
    """Agent that reviews code, tests it, and provides feedback"""
    
    def __init__(self, model: CodeGenerationModel, language: str = "rust"):
        self.model = model
        self.language = language
        
        # Get absolute path to project root (where this script is located)
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Set up paths for Rust project
        if language == "rust":
            self.rust_dir = os.path.join(self.project_root, "rust")
            self.rust_src = os.path.join(self.rust_dir, "src")
            self.rust_bin = os.path.join(self.rust_src, "bin")
            
            # Create src/bin directory if it doesn't exist
            os.makedirs(self.rust_bin, exist_ok=True)
            
            # Check if Cargo.toml exists
            if not os.path.exists(os.path.join(self.rust_dir, "Cargo.toml")):
                raise FileNotFoundError(f"Cargo.toml not found in {self.rust_dir}")
                
            print(f"Using Rust project at: {self.rust_dir}")
    
    def review(self, declaration: str, implementation: str, entry_point: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Review the provided code implementation
        
        Returns:
            Tuple of (success, feedback, details)
        """
        if self.language == "rust":
            return self.review_rust(declaration, implementation, entry_point)
        else:
            return False, f"Language {self.language} not supported for review", {}
    
    def review_rust(self, declaration: str, implementation: str, entry_point: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Review Rust code"""
        details = {}
        
        # Step 1: Check if the code compiles
        compiles, compile_feedback, compile_details = self.check_compilation(declaration, implementation, entry_point)
        details["compilation"] = compile_details
        
        if not compiles:
            return False, f"Compilation failed: {compile_feedback}", details
        
        # Step 2: Generate and run tests
        tests_generated, test_code, test_gen_details = self.generate_tests(declaration, implementation, entry_point)
        details["test_generation"] = test_gen_details
        
        if not tests_generated:
            return False, f"Could not generate tests: {test_code}", details
        
        # Step 3: Run the tests
        tests_pass, test_output, test_details = self.run_tests(declaration, implementation, test_code, entry_point)
        details["test_execution"] = test_details
        
        if not tests_pass:
            # Step 4: If tests fail, generate more detailed feedback
            detailed_feedback, analysis_details = self.generate_detailed_feedback(
                declaration, implementation, test_code, test_output
            )
            details["analysis"] = analysis_details
            return False, detailed_feedback, details
        
        return True, "Code looks good. All tests passed.", details
    
    def check_compilation(self, declaration: str, implementation: str, entry_point: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Check if the code compiles"""
        start_time = time.time()
        
        # Create a unique file name based on the entry point
        file_prefix = entry_point.lower().replace("/", "_")
        file_name = f"{file_prefix}.rs"
        file_path = os.path.join(self.rust_bin, file_name)
        
        # Combine the code and write to a file
        combined_code = "fn main(){}\n" + declaration + "\n" + implementation
        with open(file_path, "w") as f:
            f.write(combined_code)
        
        # Try to compile
        process = subprocess.run(
            ["cargo", "check", "--bin", file_prefix],
            cwd=self.rust_dir,
            capture_output=True,
            text=True
        )
        
        # Collect details
        details = {
            "duration": time.time() - start_time,
            "command": f"cargo check --bin {file_prefix}",
            "return_code": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "file_path": file_path
        }
        
        if process.returncode != 0:
            # Compilation failed, extract error messages
            error_output = process.stderr
            
            # Use the model to summarize the error
            error_prompt = f"""
Analyze this Rust compilation error and provide a clear, concise explanation of what's wrong and how to fix it:

Code:
rust
{combined_code}


Compilation error:
{error_output}


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
    
    def generate_tests(self, declaration: str, implementation: str, entry_point: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate unit tests for the code"""
        start_time = time.time()
        combined_code = declaration + "\n" + implementation
        
        test_prompt = f"""
Write comprehensive unit tests for this Rust function:

rust
{combined_code}


The entry point function name is {entry_point}.

Create multiple test cases that thoroughly test the function, including:
1. Normal cases
2. Edge cases
3. Boundary conditions

Format your response as a Rust test module wrapped with #[cfg(test)] that can be directly appended to the code.
Only write the tests, not the implementation code. Make sure the tests will run with 'cargo test'.
"""
        raw_test_code = self.model.generate_code(test_prompt, n=1)[0]
        
        # Record duration
        details = {
            "duration": time.time() - start_time,
            "raw_test_code": raw_test_code
        }
        
        # Extract the test module
        raw_test_code = raw_test_code.split("```rust")[1].split("```")[0]
        print(termcolor.colored(f"Generated tests: {raw_test_code}", "green", attrs=["bold"]))

        test_module_match = re.search(r'(#\[cfg\(test\)].+)', raw_test_code, re.DOTALL)
        if test_module_match:
            test_module = test_module_match.group(1)
        else:
            # If the pattern didn't match, just use the whole response and hope for the best
            test_module = raw_test_code
            details["module_extraction_failed"] = True
        
        # Verify if the test code looks valid
        if "#[test]" not in test_module:
            details["valid_test_not_found"] = True
            return False, "Generated test code doesn't contain valid test functions.", details
        
        details["test_module"] = test_module
        return True, test_module, details
    
    def run_tests(self, declaration: str, implementation: str, tests: str, entry_point: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Run the tests and check if they pass"""
        start_time = time.time()
        
        # Create a unique file name based on the entry point
        file_prefix = entry_point.lower().replace("/", "_")
        file_name = f"{file_prefix}.rs"
        file_path = os.path.join(self.rust_bin, file_name)
        
        # Combine the code with tests and write to a file
        combined_code = "fn main(){}\n" + declaration + "\n" + implementation + "\n\n" + tests
        with open(file_path, "w") as f:
            f.write(combined_code)
        
        # Run cargo test
        process = subprocess.run(
            ["cargo", "test", "--bin", file_prefix],
            cwd=self.rust_dir,
            capture_output=True,
            text=True
        )
        
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
            return False, process.stdout + "\n" + process.stderr, details
        
        return True, "All tests passed.", details
    
    def generate_detailed_feedback(self, declaration: str, implementation: str, tests: str, test_output: str) -> Tuple[str, Dict[str, Any]]:
        """Generate detailed feedback about why tests failed"""
        start_time = time.time()
        prompt = f"""
Analyze why these tests are failing for the given Rust code:

Code:
rust
{declaration}

{implementation}

{tests}


Test output:
{test_output}


Carefully review the code line by line. Identify what's going wrong with the implementation.
Focus on:
1. What test cases are failing
2. Why they're failing (examine expected vs. actual behavior)
3. What specific parts of the code logic are incorrect
4. How the code should be fixed

Provide a detailed analysis that would help someone fix the issues.
"""
        feedback = self.model.generate_code(prompt, n=1)[0]
        
        details = {
            "duration": time.time() - start_time,
            "feedback": feedback
        }
        
        return feedback, details
    
    def cleanup(self):
        """Cleanup temporary files if needed"""
        # We're using the existing rust project, so no cleanup needed
        pass


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


def run_multi_agent_generation(
    language: str,
    gen_model: CodeGenerationModel,
    review_model: CodeGenerationModel,
    samples: List[Dict[str, Any]],
    max_iterations: int = 3,
    verbose: bool = False,
    limit: Optional[int] = None
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
        
    Returns:
        List of processed samples with results
    """
    # Initialize agents
    parser = ContentParser()
    generator = CodeGeneratorAgent(gen_model, parser=parser)
    reviewer = CodeReviewerAgent(review_model, language=language)
    
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
        
        # Initial code generation
        if verbose:
            print("\n" + termcolor.colored("GENERATING INITIAL CODE...", "cyan", attrs=["bold"]))
            
        declaration = sample["declaration"]
        entry_point = sample["entry_point"]
        
        # Generate the initial code
        raw_code, parsed_code = generator.generate(prompt, entry_point)
        
        if verbose:
            print("\n" + termcolor.colored("GENERATOR RAW OUTPUT:", "yellow", attrs=["bold"]))
            print("-" * 40)
            print(raw_code)
            print("-" * 40)
            print("\n" + termcolor.colored("PARSED CODE:", "green", attrs=["bold"]))
            print("-" * 40)
            print(parsed_code)
            print("-" * 40)
        
        # Track iterations
        iterations = []
        iterations.append({
            "iteration": 0,
            "raw_code": raw_code,
            "parsed_code": parsed_code,
            "feedback": None,
            "review_details": None,
            "success": False
        })
        
        # Use the parsed code for review
        code = parsed_code
        
        # Refinement loop
        success = False
        for i in range(max_iterations):
            if verbose:
                print(f"\n" + termcolor.colored(f"REVIEW ITERATION {i+1}/{max_iterations}...", "cyan", attrs=["bold"]))
                
            # Review the code
            success, feedback, review_details = reviewer.review(declaration, code, entry_point)
            
            # Record this iteration
            iterations[i]["feedback"] = feedback
            iterations[i]["review_details"] = review_details
            iterations[i]["success"] = success
            
            if verbose:
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
                if verbose:
                    print("\n" + termcolor.colored("CODE PASSED ALL REVIEWS!", "green", attrs=["bold"]))
                break
                
            if verbose:
                print("\n" + termcolor.colored("REFINING CODE...", "cyan", attrs=["bold"]))
                
            # Refine the code based on feedback
            raw_new_code, parsed_new_code = generator.refine(prompt, code, feedback, entry_point)
            
            if verbose:
                print("\n" + termcolor.colored("REFINED CODE (RAW):", "yellow", attrs=["bold"]))
                print("-" * 40)
                print(raw_new_code)
                print("-" * 40)
                print("\n" + termcolor.colored("REFINED CODE (PARSED):", "green", attrs=["bold"]))
                print("-" * 40)
                print(parsed_new_code)
                print("-" * 40)
            
            # Check if the parsed code has changed
            if parsed_new_code == code:
                if verbose:
                    print("\n" + termcolor.colored("CODE DIDN'T CHANGE AFTER REFINEMENT. STOPPING ITERATIONS.", "red", attrs=["bold"]))
                break
                
            code = parsed_new_code
            
            # Record the new iteration
            iterations.append({
                "iteration": i+1,
                "raw_code": raw_new_code,
                "parsed_code": parsed_new_code,
                "feedback": None,
                "review_details": None,
                "success": False
            })
        
        # Store results
        result = {
            "task_id": sample["task_id"],
            "entry_point": entry_point,
            "declaration": declaration,
            "prompt": prompt,
            "iterations": iterations,
            "final_code": code,
            "success": success,
            "canonical_solution": sample.get("canonical_solution", "")
        }
        
        processed_samples.append(result)
        
    # Clean up
    reviewer.cleanup()
    
    return processed_samples


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Multi-agent code generation with review feedback")
    parser.add_argument("--language", type=str, default="rust", choices=["rust"],
                        help="Programming language (only Rust supported currently)")
    parser.add_argument("--gen_model_type", type=str, default="lambdalabs",
                        choices=["openai", "lambdalabs"],
                        help="Model type for code generation")
    parser.add_argument("--gen_model_name", type=str, default="llama3-8b-instruct",
                        help="Model name for code generation")
    parser.add_argument("--review_model_type", type=str, default="openai",
                        choices=["openai", "lambdalabs"],
                        help="Model type for code review")
    parser.add_argument("--review_model_name", type=str, default="gpt-4-turbo",
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
                        
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Override with environment variables if not set
    gen_model_type = os.getenv("GEN_MODEL_TYPE", args.gen_model_type)
    gen_model_name = os.getenv("GEN_MODEL_NAME", args.gen_model_name)
    review_model_type = os.getenv("REVIEW_MODEL_TYPE", args.review_model_type)
    review_model_name = os.getenv("REVIEW_MODEL_NAME", args.review_model_name)
    
    # Create the models
    gen_model = create_model(
        gen_model_type,
        gen_model_name,
        args.temperature,
        args.top_p
    )
    
    review_model = create_model(
        review_model_type,
        review_model_name,
        args.temperature,
        args.top_p
    )
    
    print(f"Using generator model: {gen_model.model_name}")
    print(f"Using reviewer model: {review_model.model_name}")
    
    # Load the dataset
    samples = [s for s in load_dataset("bigcode/humanevalpack", args.language)["test"]]
    print(f"Loaded {len(samples)} samples from HumanEvalPack {args.language} dataset")
    
    # Run multi-agent generation
    results = run_multi_agent_generation(
        language=args.language,
        gen_model=gen_model,
        review_model=review_model,
        samples=samples,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        limit=args.limit
    )
    
    # Save results
    output_file = args.output_file or f"multi_agent_results_{args.language}.jsonl"
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(results)
        
    print(f"Results saved to {output_file}")
    
    # Calculate success rate
    success_count = sum(1 for r in results if r["success"])
    success_rate = success_count / len(results) if results else 0
    print(f"Success rate: {success_rate:.2%} ({success_count}/{len(results)})")
