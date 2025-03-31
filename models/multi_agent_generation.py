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
import traceback

from models.code_generation_models import CodeGenerationModel, OpenAIChatModel, LambdaLabsModel
from models.rust_code_reviewer_agent import RustCodeReviewerAgent
from models.code_refinement_agent import CodeRefinementAgent

# Default timeout for subprocess calls (in seconds)
DEFAULT_TIMEOUT = 60  # 1 minute timeout for compilation and test execution


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
        timeout: int = 60,
        thread_id: Optional[int] = None,
        sample_idx: Optional[int] = None,
        rust_dir: Optional[str] = None,
        replace_generated_function_signature: bool = False
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
            sample_idx: Optional sample index to use for file naming
            rust_dir: Optional custom path to the Rust project directory
            replace_generated_function_signature: If True, use the entire implementation including function signatures.
                                      If False, extract only imports from declaration and use them with implementation
        """
        self._gen_model = gen_model
        self._review_model = review_model
        self._language = language
        self._max_iterations = max_iterations
        self._verbose = verbose
        self._timeout = timeout
        self._thread_id = thread_id
        self._sample_idx = sample_idx
        self._rust_dir = rust_dir
        self._replace_generated_function_signature = replace_generated_function_signature
        
        # Initialize agents
        self._generator = CodeRefinementAgent(gen_model, verbose=verbose)
        self._reviewer = RustCodeReviewerAgent(
            review_model, 
            timeout=timeout, 
            sample_idx=sample_idx,
            thread_id=thread_id,
            rust_dir=rust_dir,
            replace_generated_function_signature=replace_generated_function_signature
        )
    
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

            self._log("Extracted entry point and declaration from prompt:")
            self._log(f"Entry point: {entry_point}")
            self._log(f"Declaration: {declaration}")
            
        self._log("STARTING GENERATION WITH MULTI-AGENT APPROACH...\n", "cyan", always=True, attrs=["bold"])

        # Generate the initial code - raw output from the model
        raw_codes = self._generator.generate_code(prompt, n=1)
        raw_code = raw_codes[0] if raw_codes else ""
        
        # Refinement loop
        success = False
        current_raw_code = raw_code
        current_implementation = None  # Track the current parsed implementation
        exit_reason = self.EXIT_REASON_MAX_ITERATIONS  # Default exit reason
        iterations_data = []  # Track data for each iteration
        
        for i in range(self._max_iterations):
            self._log(f"STARTING REVIEW ITERATION {i+1}/{self._max_iterations}...\n", "cyan", always=True, attrs=["bold"])
                
            try:
                feedback_list, details, successes = self._reviewer.generate_feedback(
                    prompt,
                    "" if self._replace_generated_function_signature else declaration, 
                    current_raw_code, 
                    entry_point,
                    n=1
                )
                feedback = feedback_list[0] if feedback_list else ""
                success = successes[0] if successes else False
                review_details = details[0] if details else {}
                
                current_implementation = self._reviewer._parse_code(current_raw_code, prompt, entry_point)
                
                iteration_data = {
                    "iteration": i,
                    "raw_code": current_raw_code,
                    "feedback": feedback,
                    "review_details": review_details,
                    "success": success
                }
                iterations_data.append(iteration_data)
                
            except Exception as e:
                error_msg = f"Error during code review: {str(e)}"
                self._log(error_msg, "red", always=True, attrs=["bold"], separate_section=True)
                if self._verbose:
                    traceback.print_exc()
                feedback = f"There was an error during code review: {str(e)}. The code might have timing or resource issues."
                review_details = {"error": str(e)}
                success = False
                iteration_data = {
                    "iteration": i,
                    "raw_code": current_raw_code,
                    "feedback": feedback,
                    "review_details": review_details,
                    "success": False,
                    "error": str(e)
                }
                iterations_data.append(iteration_data)
                exit_reason = self.EXIT_REASON_ERROR
            
            # Check review results and log summary only
            if success:
                self._log("\nCODE PASSED ALL REVIEWS!", "green", always=True, attrs=["bold"], separate_section=True)
                exit_reason = self.EXIT_REASON_SUCCESS
                break
                
            if i == self._max_iterations - 1:
                self._log("\nREACHED MAXIMUM ITERATIONS", "yellow", always=True, attrs=["bold"])
                break
                
            # Only log the high-level action, details are in the agent logs
            self._log("\nSTARTING CODE REFINEMENT...", "cyan", always=True, attrs=["bold"], separate_section=True)
                
            # Refine the code based on feedback - get new raw code
            try:
                # If replace_generated_function_signature is True, don't combine with declaration 
                # since we'll use the function signatures from the implementation
                if self._replace_generated_function_signature:
                    # Use the implementation code directly with its own function signatures
                    combined_code = current_raw_code
                else:
                    # Extract imports from declaration (everything up to the first function declaration)
                    imports_match = re.search(r'^(.*?)(?=\bfn\b)', declaration, re.DOTALL)
                    imports = imports_match.group(1).strip() if imports_match else ""
                    combined_code = f"{imports}\n\n{current_raw_code}" if imports else current_raw_code
                
                refined_codes = self._generator.refine_code(prompt, code=combined_code, feedback=feedback, n=1)
                new_raw_code = refined_codes[0] if refined_codes else ""
                
                parsed_implementation = self._reviewer._parse_code(new_raw_code, prompt, entry_point)
                
                if parsed_implementation == current_implementation:
                    self._log("\nCODE DIDN'T CHANGE AFTER REFINEMENT. STOPPING ITERATIONS.", "red", always=True, attrs=["bold"])
                    exit_reason = self.EXIT_REASON_NO_CHANGE
                    break
                
                current_raw_code = new_raw_code
                current_implementation = parsed_implementation
            except Exception as e:
                error_msg = f"Error during code refinement: {str(e)}"
                self._log(error_msg, "red", always=True, attrs=["bold"], separate_section=True)
                if self._verbose:
                    traceback.print_exc()
                exit_reason = self.EXIT_REASON_ERROR
                break
        
        self._log("\nGETTING FINAL RESULT...", "cyan", always=True, attrs=["bold"])
        final_code = self._reviewer._parse_code(current_raw_code, prompt, entry_point)
        
        self._log(f"\nGENERATION COMPLETE - EXIT REASON: {exit_reason}", "green", always=True, attrs=["bold"], separate_section=True)
        
        generation_details = {
            "exit_reason": exit_reason,
            "success": success,
            "iterations": len(iterations_data),
            "iterations_data": iterations_data,
            "final_parsed_code": final_code
        }
        
        return [final_code] * n, generation_details
    
    def _parse_prompt(self, prompt: str) -> Tuple[str, str]:
        """
        Parse the prompt to extract declaration and entry point
        
        Args:
            prompt: The prompt text
            
        Returns:
            Tuple of (declaration, entry_point)
        """
        if self._language == "rust":
            fn_defs = re.findall(r'fn\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?\s*\(', prompt)
            fn_defs = [fn_name for fn_name in fn_defs if fn_name != "main"]
            if fn_defs:
                entry_point = fn_defs[0]
                main_match = re.search(r'fn\s+main\s*\(\s*\)\s*\{\s*\}(.*)', prompt, re.DOTALL)
                if main_match:
                    content_after_main = main_match.group(1).strip()
                    fn_signature_match = re.search(fr'fn\s+{entry_point}(?:<[^>]*>)?\s*\([^{{]*', content_after_main, re.DOTALL)
                    if fn_signature_match:
                        imports_match = re.search(fr'(.*?)fn\s+{entry_point}', content_after_main, re.DOTALL)
                        imports = imports_match.group(1).strip() if imports_match else ""
                        declaration = imports + ("\n" if imports else "") + fn_signature_match.group(0)
                        self._log(f"Parsed entry point: {entry_point}")
                        return declaration, entry_point
                declaration_match = re.search(fr'(fn\s+{entry_point}(?:<[^>]*>)?\s*\([^{{]*)', prompt, re.DOTALL)
                declaration = declaration_match.group(1) if declaration_match else ""
                self._log(f"Fallback parsed entry point: {entry_point}")
                return declaration, entry_point
        
        declaration_match = re.search(r'declaration:\s*(.+?)(?=\n\n|\Z)', prompt, re.DOTALL | re.IGNORECASE)
        if not declaration_match:
            declaration_match = re.search(r'according to this declaration:\s*(.+?)(?=\n\n|\Z)', prompt, re.DOTALL | re.IGNORECASE)
        declaration = declaration_match.group(1).strip() if declaration_match else ""
        entry_point_match = re.search(r'fn\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?\s*\(', declaration)
        if entry_point_match and entry_point_match.group(1) != "main":
            entry_point = entry_point_match.group(1)
        else:
            entry_point_match = re.search(r'fn\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?\s*\(', prompt)
            entry_point = entry_point_match.group(1) if entry_point_match and entry_point_match.group(1) != "main" else "solution"
        self._log(f"Generic parsed entry point: {entry_point}")
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
    sample_idx: Optional[int] = None,
    replace_generated_function_signature: bool = False
) -> MultiAgentModel:
    """
    Create a MultiAgentModel with specified generation and review models
    
    Args:
        gen_model_type: Model type for code generation
        gen_model_name: Model name for code generation
        review_model_type: Model type for code review
        review_model_name: Model name for code review
        temperature: Temperature parameter for generation
        top_p: Top-p parameter for generation
        language: Programming language
        max_iterations: Maximum iterations for refinement
        verbose: Print detailed logs
        skip_review: Skip the review process
        timeout: Timeout for subprocess calls in seconds
        sample_idx: Optional sample index for file naming
        replace_generated_function_signature: If True, use entire implementation with function signatures
    """
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
    
    cargo_available = True
    if language == "rust" and not skip_review:
        try:
            RustCodeReviewerAgent(review_model, language=language, timeout=timeout, sample_idx=sample_idx)
        except FileNotFoundError as e:
            cargo_available = False
            print(f"Warning: {str(e)}")
            print("Running in code generation only mode (no compilation or testing)")
            if not skip_review:
                print("Use --skip_review to suppress this warning")
                max_iterations = 1
    
    multi_agent = MultiAgentModel(
        gen_model=gen_model,
        review_model=review_model,
        language=language,
        max_iterations=max_iterations if cargo_available else 1,
        verbose=verbose,
        timeout=timeout,
        sample_idx=sample_idx,
        replace_generated_function_signature=replace_generated_function_signature
    )
    
    if verbose:
        print(f"Using multi-agent model: {multi_agent.model_name}")
    return multi_agent


if __name__ == "__main__":
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
    parser.add_argument("--replace_generated_function_signature", action="store_true", default=False,
                        help="Use entire implementation with function signatures (default: False, only use imports)")
                        
    args = parser.parse_args()
    
    load_dotenv()
    
    gen_model_type = os.getenv("GEN_MODEL_TYPE", args.gen_model_type)
    gen_model_name = os.getenv("GEN_MODEL_NAME", args.gen_model_name)
    review_model_type = os.getenv("REVIEW_MODEL_TYPE", args.review_model_type)
    review_model_name = os.getenv("REVIEW_MODEL_NAME", args.review_model_name)
    
    samples = [s for s in load_dataset("bigcode/humanevalpack", args.language)["test"]]
    print(f"Loaded {len(samples)} samples from HumanEvalPack {args.language} dataset")
    
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
        skip_review=args.skip_review,
        replace_generated_function_signature=args.replace_generated_function_signature
    )
    
    print(f"Using multi-agent model: {multi_agent.model_name}")
    
    results = []
    
    for idx, sample in enumerate(tqdm(samples)):
        if args.limit is not None and idx >= args.limit:
            break
            
        if args.verbose:
            print(f"\n{'-'*80}")
            print(f"Processing sample {idx+1}/{len(samples)}: {sample['task_id']}")
            print(f"{'-'*80}")
            
        prompt = get_prompt_from_sample(sample, args.language)
        declaration = sample["declaration"]
        entry_point = sample["entry_point"]
        
        final_code, generation_details = multi_agent.generate_code(
            prompt, 
            declaration=declaration, 
            entry_point=entry_point
        )
        final_code = final_code[0]
        
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
    
    output_file = args.output_file or f"multi_agent_results_{args.language}.jsonl"
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(results)
        
    print(f"Results saved to {output_file}")
    
    success_count = sum(1 for r in results if r["success"])
    success_rate = success_count / len(results) if results else 0
    print(f"Success rate: {success_rate:.2%} ({success_count}/{len(results)})")
    
    exit_reasons = {}
    for r in results:
        exit_reason = r.get("exit_reason", "unknown")
        exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
    
    print("\nExit reason statistics:")
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count} ({count/len(results):.2%})")
