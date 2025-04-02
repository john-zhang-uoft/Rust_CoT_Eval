#!/usr/bin/env python3
"""
Confidence-based multi-agent code generation system with planning, coding, and testing agents
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
from models.planner_agent import PlannerAgent
from models.confidence_checker import ConfidenceChecker
from models.tester_checker_agent import TesterCheckerAgent

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

class ConfidenceMultiAgentModel:
    """
    Confidence-based multi-agent code generation system that uses multiple agents with
    confidence checking to generate high-quality code solutions.
    """
    
    def __init__(
        self,
        planner_model: CodeGenerationModel,
        coder_model: CodeGenerationModel,
        tester_model: CodeGenerationModel,
        language: str = "rust",
        max_iterations: int = 3,
        max_planning_attempts: int = 2,
        confidence_threshold: int = 70,
        low_confidence_threshold: int = 30,
        verbose: bool = False
    ):
        """
        Initialize a confidence-based multi-agent code generation model
        
        Args:
            planner_model: Model for planning
            coder_model: Model for code generation
            tester_model: Model for testing
            language: Programming language
            max_iterations: Maximum number of refinement iterations
            max_planning_attempts: Maximum number of planning attempts
            confidence_threshold: Threshold for considering an agent confident (0-100)
            low_confidence_threshold: Threshold for considering an agent not confident (0-100)
            verbose: Whether to print detailed logs
        """
        self.language = language
        self.max_iterations = max_iterations
        self.max_planning_attempts = max_planning_attempts
        self.confidence_threshold = confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.verbose = verbose
        
        # Initialize agents
        self.planner = PlannerAgent(planner_model, verbose=verbose)
        self.coder = CodeRefinementAgent(coder_model, verbose=verbose)
        self.tester = RustCodeReviewerAgent(tester_model, verbose=verbose)
        self.test_checker = TesterCheckerAgent(tester_model, language=language, verbose=verbose)
        
        # Initialize confidence checkers
        self.planner_confidence_checker = ConfidenceChecker(planner_model, verbose=verbose)
        self.coder_confidence_checker = ConfidenceChecker(coder_model, verbose=verbose)
        self.tester_confidence_checker = ConfidenceChecker(tester_model, verbose=verbose)
        
        # Store models for reference
        self.planner_model = planner_model
        self.coder_model = coder_model
        self.tester_model = tester_model
        
        self._log(f"Initialized confidence multi-agent model with {language} language")
    
    def _log(self, message: str, color: str = None, always: bool = False):
        """
        Log a message if verbose is true or always is true
        
        Args:
            message: The message to log
            color: Optional color for the message (using termcolor)
            always: Whether to log regardless of verbose setting
        """
        if self.verbose or always:
            if color:
                print(termcolor.colored(message, color))
            else:
                print(message)
    
    def generate_code(
        self, 
        prompt: str, 
        declaration: str, 
        entry_point: str
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Generate code using the multi-agent system
        
        Args:
            prompt: The problem statement
            declaration: The function declaration
            entry_point: The function name
            
        Returns:
            Tuple containing:
            - List of generated code
            - Dictionary with generation details
        """
        planning_attempts = 0
        iterations = 0
        iterations_data = []
        final_confidence = {}
        success = False
        exit_reason = "max_iterations_reached"
        
        # Phase 1: Planning
        while planning_attempts < self.max_planning_attempts:
            self._log(f"\n{'='*80}\nPHASE 1: PLANNING (Attempt {planning_attempts+1}/{self.max_planning_attempts})\n{'='*80}", "blue", always=True)
            
            # Get plan from planner agent
            plan_data = self.planner.create_plan(prompt, declaration, entry_point)
            
            # Check planner confidence
            planner_system_prompt = self.planner.system_prompt
            planner_response = json.dumps(plan_data, indent=2)
            planner_confidence, planner_explanation = self.planner_confidence_checker.check_confidence(
                planner_system_prompt, 
                planner_response, 
                prompt
            )
            self._log(f"Planner confidence: {planner_confidence}/100", "cyan")
            self._log(f"Planner explanation: {planner_explanation}", "cyan")
            
            # If planner has very low confidence, try again
            if planner_confidence < self.low_confidence_threshold:
                self._log(f"Planner confidence too low ({planner_confidence}). Restarting planning...", "yellow", always=True)
                planning_attempts += 1
                if planning_attempts >= self.max_planning_attempts:
                    exit_reason = "planning_confidence_too_low"
                    break
                continue
            
            # Phase 2: Initial coding
            self._log(f"\n{'='*80}\nPHASE 2: INITIAL CODING\n{'='*80}", "blue", always=True)
            
            # Generate initial code based on the plan
            coding_prompt = f"""
You are implementing a solution for the following problem:
{prompt}

Here is a plan to solve this problem:
{plan_data['pseudocode']}

Implement the solution in Rust according to this declaration:
{declaration}
"""
            initial_code = self.coder.generate_code(coding_prompt)[0]
            
            # Check coder confidence
            coder_system_prompt = self.coder.system_prompt
            coder_confidence, coder_explanation = self.coder_confidence_checker.check_confidence(
                coder_system_prompt,
                initial_code,
                coding_prompt
            )
            
            self._log(f"Coder confidence: {coder_confidence}/100", "cyan")
            self._log(f"Coder explanation: {coder_explanation}", "cyan")
            
            current_code = initial_code
            
            # Phase 3: Iterative refinement
            while iterations < self.max_iterations:
                self._log(f"\n{'='*80}\nPHASE 3: ITERATION {iterations+1}/{self.max_iterations}\n{'='*80}", "blue", always=True)
                
                # Test the code
                self._log(f"\nTesting code...", "cyan")
                feedback, review_details, test_success = self.tester.generate_feedback(
                    prompt, declaration, current_code, entry_point
                )
                # Convert to single values since generate_feedback returns lists
                feedback = feedback[0]
                success = test_success[0]
                review_detail = review_details[0]
                
                # Check tester confidence
                tester_system_prompt = self.tester.system_prompt
                tester_confidence, tester_explanation = self.tester_confidence_checker.check_confidence(
                    tester_system_prompt,
                    feedback,
                    prompt
                )
                
                self._log(f"Tester confidence: {tester_confidence}/100", "cyan")
                self._log(f"Tester explanation: {tester_explanation}", "cyan")
                
                # If test code was generated, check its quality
                if 'test_generation' in review_detail and 'test_module' in review_detail['test_generation']:
                    tests = review_detail['test_generation']['test_module']
                    test_quality_feedback = self.test_checker.check_tests(
                        prompt, declaration, current_code, tests, entry_point
                    )
                    self._log(f"Test quality feedback: {test_quality_feedback}", "cyan")
                
                # Store iteration data
                iteration_data = {
                    "iteration": iterations,
                    "code": current_code,
                    "feedback": feedback,
                    "success": success,
                    "confidence": {
                        "planner": planner_confidence,
                        "coder": coder_confidence,
                        "tester": tester_confidence
                    }
                }
                iterations_data.append(iteration_data)
                
                # Exit early if tests pass
                if success:
                    self._log(f"Tests passed! Exiting early.", "green", always=True)
                    exit_reason = "tests_passed"
                    break
                
                # Exit if low confidence across all agents
                agents_confidence = {
                    "planner": planner_confidence,
                    "coder": coder_confidence,
                    "tester": tester_confidence
                }
                
                # If all agents have low confidence, try restarting with new plan
                restart_needed = self.planner.evaluate_confidence(agents_confidence)
                if restart_needed:
                    self._log(f"Low confidence detected across agents. Restarting planning...", "yellow", always=True)
                    planning_attempts += 1
                    break
                
                # Otherwise refine the code
                if iterations < self.max_iterations - 1:
                    self._log(f"\nRefining code...", "cyan")
                    refined_code = self.coder.refine_code(
                        prompt, current_code, feedback
                    )[0]
                    
                    # Check coder confidence in refined code
                    refinement_prompt = f"Refine code based on feedback: {feedback}"
                    coder_confidence, coder_explanation = self.coder_confidence_checker.check_confidence(
                        coder_system_prompt,
                        refined_code,
                        refinement_prompt
                    )
                    
                    self._log(f"Coder confidence in refined code: {coder_confidence}/100", "cyan")
                    self._log(f"Coder explanation: {coder_explanation}", "cyan")
                    
                    current_code = refined_code
                
                iterations += 1
            
            # If we've succeeded or exhausted iterations, exit the planning loop
            if success or iterations >= self.max_iterations:
                break
            
            planning_attempts += 1
        
        # Store final confidence scores
        final_confidence = {
            "planner": planner_confidence,
            "coder": coder_confidence,
            "tester": tester_confidence
        }
        
        # Prepare generation details
        generation_details = {
            "success": success,
            "exit_reason": exit_reason,
            "iterations_data": iterations_data,
            "planning_attempts": planning_attempts,
            "final_confidence": final_confidence
        }
        
        return [current_code], generation_details
    
    @property
    def model_name(self) -> str:
        """Get the model name"""
        return f"ConfidenceMultiAgent({self.planner_model.model_name},{self.coder_model.model_name},{self.tester_model.model_name})"


def create_confidence_multi_agent_model(
    planner_model_type: str,
    planner_model_name: str,
    coder_model_type: str,
    coder_model_name: str,
    tester_model_type: str,
    tester_model_name: str,
    temperature: float,
    top_p: float,
    language: str = "rust",
    max_iterations: int = 3,
    max_planning_attempts: int = 2,
    confidence_threshold: int = 70,
    low_confidence_threshold: int = 30,
    verbose: bool = False
) -> ConfidenceMultiAgentModel:
    """
    Create a confidence-based multi-agent model
    
    Args:
        planner_model_type: Model type for planner
        planner_model_name: Model name for planner
        coder_model_type: Model type for coder
        coder_model_name: Model name for coder
        tester_model_type: Model type for tester
        tester_model_name: Model name for tester
        temperature: Temperature for generation
        top_p: Top-p for generation
        language: Programming language
        max_iterations: Maximum number of refinement iterations
        max_planning_attempts: Maximum number of planning attempts
        confidence_threshold: Threshold for considering an agent confident (0-100)
        low_confidence_threshold: Threshold for considering an agent not confident (0-100)
        verbose: Whether to print detailed logs
        
    Returns:
        A confidence-based multi-agent model
    """
    # Create models
    planner_model = create_model(planner_model_type, planner_model_name, temperature, top_p)
    coder_model = create_model(coder_model_type, coder_model_name, temperature, top_p)
    tester_model = create_model(tester_model_type, tester_model_name, temperature, top_p)
    
    # Create multi-agent model
    multi_agent = ConfidenceMultiAgentModel(
        planner_model=planner_model,
        coder_model=coder_model,
        tester_model=tester_model,
        language=language,
        max_iterations=max_iterations,
        max_planning_attempts=max_planning_attempts,
        confidence_threshold=confidence_threshold,
        low_confidence_threshold=low_confidence_threshold,
        verbose=verbose
    )
    
    return multi_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence-based multi-agent code generation")
    parser.add_argument("--language", type=str, default="rust", choices=["rust"],
                        help="Programming language (only Rust supported currently)")
    parser.add_argument("--planner_model_type", type=str, default="lambdalabs",
                        choices=["openai", "lambdalabs"],
                        help="Model type for planning")
    parser.add_argument("--planner_model_name", type=str, default="llama3.3-70b-instruct-fp8",
                        help="Model name for planning")
    parser.add_argument("--coder_model_type", type=str, default="lambdalabs",
                        choices=["openai", "lambdalabs"],
                        help="Model type for coding")
    parser.add_argument("--coder_model_name", type=str, default="llama3.3-70b-instruct-fp8",
                        help="Model name for coding")
    parser.add_argument("--tester_model_type", type=str, default="lambdalabs",
                        choices=["openai", "lambdalabs"],
                        help="Model type for testing")
    parser.add_argument("--tester_model_name", type=str, default="llama3.3-70b-instruct-fp8",
                        help="Model name for testing")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature parameter for generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p parameter for generation")
    parser.add_argument("--max_iterations", type=int, default=3,
                        help="Maximum number of refinement iterations")
    parser.add_argument("--max_planning_attempts", type=int, default=2,
                        help="Maximum number of planning attempts")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file path (default: confidence_multi_agent_results_{language}.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of samples to process")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed logs")
                        
    args = parser.parse_args()
    
    load_dotenv()
    
    # Override with environment variables if available
    planner_model_type = os.getenv("PLANNER_MODEL_TYPE", args.planner_model_type)
    planner_model_name = os.getenv("PLANNER_MODEL_NAME", args.planner_model_name)
    coder_model_type = os.getenv("CODER_MODEL_TYPE", args.coder_model_type)
    coder_model_name = os.getenv("CODER_MODEL_NAME", args.coder_model_name)
    tester_model_type = os.getenv("TESTER_MODEL_TYPE", args.tester_model_type)
    tester_model_name = os.getenv("TESTER_MODEL_NAME", args.tester_model_name)
    
    samples = [s for s in load_dataset("bigcode/humanevalpack", args.language)["test"]]
    print(f"Loaded {len(samples)} samples from HumanEvalPack {args.language} dataset")
    
    multi_agent = create_confidence_multi_agent_model(
        planner_model_type=planner_model_type,
        planner_model_name=planner_model_name,
        coder_model_type=coder_model_type,
        coder_model_name=coder_model_name,
        tester_model_type=tester_model_type,
        tester_model_name=tester_model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        language=args.language,
        max_iterations=args.max_iterations,
        max_planning_attempts=args.max_planning_attempts,
        verbose=args.verbose
    )
    
    print(f"Using confidence multi-agent model: {multi_agent.model_name}")
    
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
            "final_confidence": generation_details.get("final_confidence", {}),
            "canonical_solution": sample.get("canonical_solution", "")
        }
        
        results.append(result)
    
    output_file = args.output_file or f"confidence_multi_agent_results_{args.language}.jsonl"
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
