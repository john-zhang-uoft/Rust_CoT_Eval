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
        keep_generated_function_signature: bool = True,
        tester_knows_cases: bool = False,
        disable_plan_restart: bool = False,
        disable_system_prompts: bool = False,
        no_planner: bool = False,
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
            tester_knows_cases: Whether to use the test cases from the dataset
            disable_plan_restart: Whether to disable restarting from plan when confidence is low
            disable_system_prompts: Whether to disable all system prompts
            no_planner: Whether to skip the planning phase entirely
            verbose: Whether to print detailed logs
        """
        self.language = language
        self.max_iterations = max_iterations
        self.max_planning_attempts = max_planning_attempts
        self.confidence_threshold = confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.keep_generated_function_signature = keep_generated_function_signature
        self.tester_knows_cases = tester_knows_cases
        self.disable_plan_restart = disable_plan_restart
        self.disable_system_prompts = disable_system_prompts
        self.no_planner = no_planner
        self.verbose = verbose
        
        # Initialize agents
        self.planner = PlannerAgent(planner_model, verbose=verbose, disable_system_prompts=disable_system_prompts)
        self.coder = CodeRefinementAgent(coder_model, verbose=verbose, disable_system_prompts=disable_system_prompts)
        self.tester = RustCodeReviewerAgent(tester_model, verbose=verbose, keep_generated_function_signature=keep_generated_function_signature, disable_system_prompts=disable_system_prompts)
        self.test_checker = TesterCheckerAgent(tester_model, language=language, verbose=verbose, disable_system_prompts=disable_system_prompts)
        
        # Initialize a single confidence checker instance for all agents
        self.confidence_checker = ConfidenceChecker(tester_model, verbose=verbose, disable_system_prompts=disable_system_prompts)
        
        # Store models for reference
        self.planner_model = planner_model
        self.coder_model = coder_model
        self.tester_model = tester_model
        
        self._log(f"Initialized confidence multi-agent model with {language} language")
        self._log(f"Confidence threshold: {confidence_threshold}, Low confidence threshold: {low_confidence_threshold}")
        self._log(f"Disable plan restart: {disable_plan_restart}, Disable system prompts: {disable_system_prompts}")
        if no_planner:
            self._log(f"Planning phase disabled, starting directly with coding", "yellow")
    
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
        entry_point: str,
        test_code: Optional[str] = None
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Generate code using the multi-agent system
        
        Args:
            prompt: The problem statement
            declaration: The function declaration
            entry_point: The function name
            test_code: The provided test code (if tester_knows_cases is True)
            
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
        
        # Initialize full_code with empty string as default
        full_code = ""
        code_that_compiles = ""
        
        planner_confidence, coder_confidence, tester_confidence = -1, -1, -1
        plan_data = {"pseudocode": ""}
        
        # Phase 1: Planning (unless no_planner is True)
        if not self.no_planner:
            while planning_attempts < self.max_planning_attempts:
                self._log(f"\n{'='*80}\nPHASE 1: PLANNING (Attempt {planning_attempts+1}/{self.max_planning_attempts})\n{'='*80}", "blue", always=True)
                
                # Get plan from planner agent - use refine_plan if not first attempt
                if planning_attempts == 0:
                    plan_data = self.planner.create_plan(prompt, declaration, entry_point)
                else:
                    # Get feedback from the latest iteration
                    latest_iteration = iterations_data[-1] if iterations_data else None
                    feedback = latest_iteration.get("feedback", "") if latest_iteration else ""
                    self._log(f"Using feedback from latest iteration to refine plan: {feedback}", "cyan")
                    plan_data = self.planner.refine_plan(prompt, declaration, entry_point, feedback)
                
                # Check planner confidence
                planner_system_prompt = self.planner.system_prompt
                planner_response = json.dumps(plan_data, indent=2)
                planner_confidence, planner_explanation = self.confidence_checker.check_confidence(
                    planner_system_prompt, 
                    planner_response, 
                    prompt
                )
                self._log(f"Planner confidence: {planner_confidence}/100", "cyan")
                self._log(f"Planner explanation: {planner_explanation}", "cyan")
                
                # Phase 2: Initial coding
                self._log(f"\n{'='*80}\nPHASE 2: INITIAL CODING\n{'='*80}", "blue", always=True)
                
                # Generate initial code based on the plan
                coding_prompt = f"""
You are implementing a solution for the following problem:
{prompt}

Here is a plan to solve this problem:
{plan_data['pseudocode']}

Implement the solution in Rust according to this function signature:
{declaration}
"""
                
                initial_coder_output = self.coder.generate_code(coding_prompt)[0]
                
                # Check coder confidence
                coder_system_prompt = self.coder.system_prompt
                coder_confidence, coder_explanation = self.confidence_checker.check_confidence(
                    coder_system_prompt,
                    initial_coder_output,
                    coding_prompt
                )
                
                self._log(f"Coder confidence: {coder_confidence}/100", "cyan")
                self._log(f"Coder explanation: {coder_explanation}", "cyan")
                
                current_coder_output = initial_coder_output

                compilation_black_board = ""
                testing_blackboard = {}

                original_declaration = declaration

                if self.keep_generated_function_signature:
                    # When we want to keep function signatures from the implementation
                    self._log("Using implementation with its own function signatures instead of replacing it.", "cyan")
                    # Remove the function signature from the declaration
                    declaration = declaration.split("fn " + entry_point)[0]

                # Phase 3: Iterative refinement
                while iterations < self.max_iterations:
                    self._log(f"\n{'='*80}\nPHASE 3: ITERATION {iterations+1}/{self.max_iterations}\n{'='*80}", "blue", always=True)
                    
                    current_extracted_code = self.tester.parse_code(current_coder_output, prompt, entry_point)

                    if self.keep_generated_function_signature:
                        # When we want to keep function signatures from the implementation
                        full_code = current_extracted_code

                        self._log(f"\nSTEP 0: CHECKING SIGNATURE MATCHES...", "cyan")
                        signature_matches = self.tester.check_signature_matches(original_declaration, full_code, entry_point)
                        if not signature_matches:
                            self._log("Signature does not match. Refining code...", "yellow")
                            # Refine the code
                            fixed_output = self.coder.refine_code(prompt, full_code, "The function signature does not match the signature in the problem statement. Please fix it.")[0]
                            full_code = self.tester.parse_code(fixed_output, prompt, entry_point)
                    else:
                        # Normal case: combine declaration and implementation
                        self._log("Using declaration + implementation", "cyan")
                        full_code = f"{declaration}\n{current_extracted_code}"

                    # Step 1: Check if the code compiles
                    self._log(f"\nSTEP 1: CHECKING COMPILATION...", "cyan")
                    compiles, compile_feedback, compile_details = self.tester.check_compilation(declaration, full_code)
                    compilation_black_board += "Code:\n\n" + full_code + "\n\nCompilation feedback:\n\n" + compile_feedback
                    if not compiles:
                        self._log(f"Compilation failed: {compile_feedback}", "red")
                        
                        # Store iteration data
                        iteration_data = {
                            "iteration": iterations,
                            "code": full_code,
                            "feedback": compile_feedback,
                            "success": False,
                            "compilation": compile_details,
                            "confidence": {
                                "planner": planner_confidence,
                                "coder": coder_confidence,
                                "tester": 0  # No tester confidence if compilation fails
                            }
                        }
                        iterations_data.append(iteration_data)
                        
                        # Refine the code
                        if iterations < self.max_iterations - 1:
                            self._log(f"\nRefining code after compilation error...", "cyan")
                            refined_code = self.coder.refine_code(prompt, full_code, compile_feedback)[0]
                            
                            # Check coder confidence in refined code
                            refinement_prompt = f"Refine code based on compilation feedback: {compilation_black_board}"
                            coder_confidence, coder_explanation = self.confidence_checker.check_confidence(
                                coder_system_prompt,
                                refined_code,
                                refinement_prompt
                            )
                            
                            self._log(f"Coder confidence in refined code: {coder_confidence}/100", "cyan")
                            self._log(f"Coder explanation: {coder_explanation}", "cyan")
                            
                            current_coder_output = refined_code
                            iterations += 1
                            continue
                        else:
                            success = False
                            exit_reason = "compilation_failed"
                            full_code = code_that_compiles
                            break
                    
                    code_that_compiles = full_code
                    # Step 2: Generate tests
                    self._log(f"\nSTEP 2: GENERATING TESTS...", "cyan")
                    
                    if self.tester_knows_cases and test_code:
                        # Use the prebuilt test cases if the flag is enabled and test_code is provided
                        self._log(f"Using prebuilt test cases from the dataset", "cyan")
                        tests_generated = True
                        test_gen_details = {"source": "dataset"}
                        # Set tester confidence high when using dataset tests
                        tester_confidence = 100
                        tester_explanation = "Using provided test cases from the dataset"
                        # Initialize variables that would normally be set during test generation/refinement
                        test_quality_feedback = "Using dataset test cases"
                        refined_test_details = {"source": "dataset"}
                    else:
                        # Check if coder has low confidence
                        test_quality_feedback = ""
                        refined_test_details = {}
                        refined_tests = ""
                        low_confidence_message = ""
                        if coder_confidence < self.low_confidence_threshold:
                            low_confidence_message = "I'm not really sure if this solution is correct..."
                            self._log(f"Adding low confidence message to test generator: {low_confidence_message}", "yellow")
                        
                        if "current_tests" not in testing_blackboard:
                            tests_generated, test_code, test_gen_details = self.tester.generate_tests(
                                prompt, declaration, full_code, entry_point, extra_instructions=low_confidence_message
                            )
                            if not tests_generated:
                                self._log(f"Could not generate tests: {test_code}", "red")
                                
                                # Store iteration data
                                iteration_data = {
                                    "iteration": iterations,
                                    "code": full_code,
                                    "feedback": f"Test generation failed: {test_code}",
                                    "success": False,
                                    "compilation": compile_details,
                                    "test_generation": test_gen_details,
                                    "confidence": {
                                        "planner": planner_confidence,
                                        "coder": coder_confidence,
                                        "tester": 0  # No tester confidence if test generation fails
                                    }
                                }
                                iterations_data.append(iteration_data)
                                
                                # Exit with failure
                                success = False
                                exit_reason = "test_generation_failed"
                                break
                        else:
                            test_code = testing_blackboard["current_tests"]
                            
                        # Either we have tests from a previous iteration or we just generated them
                        # Check tester confidence in the test code
                        tester_system_prompt = self.tester.system_prompt
                        tester_confidence, tester_explanation = self.confidence_checker.check_confidence(
                            tester_system_prompt,
                            test_code,
                            "Write tests for the following code:\n" + full_code,
                            "" if "test_feedback" not in testing_blackboard else testing_blackboard["test_feedback"]
                        )
                        self._log(f"Tester confidence in tests: {tester_confidence}/100", "cyan")
                        self._log(f"Tester explanation of confidence: {tester_explanation}", "cyan")
                        # Check if tester has low confidence
                        if tester_confidence < self.low_confidence_threshold:
                            # Step 3: Check test quality (Skip this step if using prebuilt tests)
                            self._log(f"\nSTEP 3: CHECKING TEST QUALITY...", "cyan")
                            test_low_confidence_message = "I'm not really sure if these tests are correct..."
                            self._log(f"Adding low confidence message to test checker: {test_low_confidence_message}", "yellow")
                            self._log("Checking and refining tests...", "cyan")
                            test_quality_feedback = self.test_checker.check_tests(
                                prompt, declaration, full_code, test_code, entry_point, extra_instructions=test_low_confidence_message
                            )
                        
                            self._log(f"Test quality feedback: {test_quality_feedback}", "cyan")
                            
                            # Step 4: Refine tests based on feedback
                            self._log(f"\nSTEP 4: REFINING TESTS...", "cyan")
                            extra_instructions = f"""
    Based on the review, improve your tests to better match the requirements.
    Feedback on your tests:
    {test_quality_feedback}
    """
                            tests_refined, refined_test_code, refined_test_details = self.tester.generate_tests(
                                prompt, declaration, full_code, entry_point, extra_instructions=extra_instructions
                            )      
                            if not tests_refined:
                                self._log(f"Could not refine tests: {refined_test_code}", "red")
                                self._log(f"Falling back to original tests", "yellow")
                            else:
                                test_code = refined_test_code
                
                    # Step 5: Run tests
                    self._log(f"\nSTEP 5: RUNNING TESTS...", "cyan")
                    tests_pass, test_output, test_details = self.tester.run_tests(
                        declaration, full_code, test_code, entry_point
                    )
                    # Store overall feedback
                    if tests_pass:
                        feedback = "Code looks good. All tests passed."
                    else:
                        # Generate detailed feedback for test failures
                        detailed_test_feedback, analysis_details = self.tester.generate_detailed_feedback(
                            prompt, declaration, full_code, test_code, test_output
                        )
                        feedback = detailed_test_feedback
                        test_details["analysis"] = analysis_details

                    testing_blackboard["test_feedback"] = feedback
                    testing_blackboard["current_tests"] = test_code

                    # Store iteration data
                    iteration_data = {
                        "iteration": iterations,
                        "code": full_code,
                        "feedback": feedback,
                        "success": tests_pass,
                        "compilation": compile_details,
                        "test_generation": test_gen_details,
                        "test_quality_feedback": test_quality_feedback,
                        "refined_tests": refined_test_details,
                        "test_execution": test_details,
                        "confidence": {
                            "planner": planner_confidence,
                            "coder": coder_confidence,
                            "tester": tester_confidence
                        }
                    }
                    iterations_data.append(iteration_data)
                    
                    # Exit early if tests pass
                    if tests_pass:
                        self._log(f"Tests passed! Exiting early.", "green", always=True)
                        success = True
                        exit_reason = "tests_passed"
                        break
                    
                    # Exit if low confidence across all agents
                    agents_confidence = {
                        "planner": planner_confidence,
                        "coder": coder_confidence,
                        "tester": tester_confidence
                    }
                    
                    # If all agents have low confidence and plan restart is not disabled, try restarting with new plan
                    restart_needed = False
                    if not self.disable_plan_restart:
                        restart_needed = self.planner.evaluate_confidence(agents_confidence)
                    
                    if restart_needed:
                        self._log(f"Low confidence detected across agents. Restarting planning...", "yellow", always=True)
                        planning_attempts += 1
                        break
                    
                    # Otherwise refine the code
                    if iterations < self.max_iterations - 2:
                        self._log(f"\nRefining code based on test failures...", "cyan")
                        refined_code = self.coder.refine_code(
                            prompt, full_code, feedback
                        )[0]
                        
                        # Check coder confidence in refined code
                        refinement_prompt = f"Refine code based on feedback: {feedback}"
                        coder_confidence, coder_explanation = self.confidence_checker.check_confidence(
                            coder_system_prompt,
                            refined_code,
                            refinement_prompt
                        )
                        
                        self._log(f"Coder confidence in refined code: {coder_confidence}/100", "cyan")
                        self._log(f"Coder explanation: {coder_explanation}", "cyan")
                        
                        current_coder_output = refined_code
                    
                    iterations += 1
                
                # If we've succeeded or exhausted iterations, exit the planning loop
                if success or iterations >= self.max_iterations:
                    break
                
                planning_attempts += 1
        else:
            # Skip planning when no_planner is True
            self._log(f"\n{'='*80}\nPHASE 1: PLANNING SKIPPED (no_planner=True)\n{'='*80}", "blue", always=True)
            planner_confidence = 100  # Assume high confidence
            
            # Phase 2: Initial coding without plan
            self._log(f"\n{'='*80}\nPHASE 2: INITIAL CODING\n{'='*80}", "blue", always=True)
            
            # Direct coding without plan
            coding_prompt = f"""
You are implementing a solution for the following problem:
{prompt}

Implement the solution in Rust according to this function signature:
{declaration}
"""
            
            initial_coder_output = self.coder.generate_code(coding_prompt)[0]
            
            # Check coder confidence
            coder_system_prompt = self.coder.system_prompt
            coder_confidence, coder_explanation = self.confidence_checker.check_confidence(
                coder_system_prompt,
                initial_coder_output,
                coding_prompt
            )
            
            self._log(f"Coder confidence: {coder_confidence}/100", "cyan")
            self._log(f"Coder explanation: {coder_explanation}", "cyan")
            
            current_coder_output = initial_coder_output

            compilation_black_board = ""
            testing_blackboard = {}

            original_declaration = declaration

            if self.keep_generated_function_signature:
                # When we want to keep function signatures from the implementation
                self._log("Using implementation with its own function signatures instead of replacing it.", "cyan")
                # Remove the function signature from the declaration
                declaration = declaration.split("fn " + entry_point)[0]

            # Phase 3: Iterative refinement
            while iterations < self.max_iterations:
                self._log(f"\n{'='*80}\nPHASE 3: ITERATION {iterations+1}/{self.max_iterations}\n{'='*80}", "blue", always=True)
                
                current_extracted_code = self.tester.parse_code(current_coder_output, prompt, entry_point)

                if self.keep_generated_function_signature:
                    # When we want to keep function signatures from the implementation
                    full_code = current_extracted_code

                    self._log(f"\nSTEP 0: CHECKING SIGNATURE MATCHES...", "cyan")
                    signature_matches = self.tester.check_signature_matches(original_declaration, full_code, entry_point)
                    if not signature_matches:
                        self._log("Signature does not match. Refining code...", "yellow")
                        # Refine the code
                        fixed_output = self.coder.refine_code(prompt, full_code, "The function signature does not match the signature in the problem statement. Please fix it.")[0]
                        full_code = self.tester.parse_code(fixed_output, prompt, entry_point)
                else:
                    # Normal case: combine declaration and implementation
                    self._log("Using declaration + implementation", "cyan")
                    full_code = f"{declaration}\n{current_extracted_code}"

                # Step 1: Check if the code compiles
                self._log(f"\nSTEP 1: CHECKING COMPILATION...", "cyan")
                compiles, compile_feedback, compile_details = self.tester.check_compilation(declaration, full_code)
                compilation_black_board += "Code:\n\n" + full_code + "\n\nCompilation feedback:\n\n" + compile_feedback
                if not compiles:
                    self._log(f"Compilation failed: {compile_feedback}", "red")
                    
                    # Store iteration data
                    iteration_data = {
                        "iteration": iterations,
                        "code": full_code,
                        "feedback": compile_feedback,
                        "success": False,
                        "compilation": compile_details,
                        "confidence": {
                            "planner": planner_confidence,
                            "coder": coder_confidence,
                            "tester": 0  # No tester confidence if compilation fails
                        }
                    }
                    iterations_data.append(iteration_data)
                    
                    # Refine the code
                    if iterations < self.max_iterations - 1:
                        self._log(f"\nRefining code after compilation error...", "cyan")
                        refined_code = self.coder.refine_code(prompt, full_code, compile_feedback)[0]
                        
                        # Check coder confidence in refined code
                        refinement_prompt = f"Refine code based on compilation feedback: {compilation_black_board}"
                        coder_confidence, coder_explanation = self.confidence_checker.check_confidence(
                            coder_system_prompt,
                            refined_code,
                            refinement_prompt
                        )
                        
                        self._log(f"Coder confidence in refined code: {coder_confidence}/100", "cyan")
                        self._log(f"Coder explanation: {coder_explanation}", "cyan")
                        
                        current_coder_output = refined_code
                        iterations += 1
                        continue
                    else:
                        success = False
                        exit_reason = "compilation_failed"
                        full_code = code_that_compiles
                        break
                
                code_that_compiles = full_code
                # Step 2: Generate tests
                self._log(f"\nSTEP 2: GENERATING TESTS...", "cyan")
                
                if self.tester_knows_cases and test_code:
                    # Use the prebuilt test cases if the flag is enabled and test_code is provided
                    self._log(f"Using prebuilt test cases from the dataset", "cyan")
                    tests_generated = True
                    test_gen_details = {"source": "dataset"}
                    # Set tester confidence high when using dataset tests
                    tester_confidence = 100
                    tester_explanation = "Using provided test cases from the dataset"
                    # Initialize variables that would normally be set during test generation/refinement
                    test_quality_feedback = "Using dataset test cases"
                    refined_test_details = {"source": "dataset"}
                else:
                    # Check if coder has low confidence
                    test_quality_feedback = ""
                    refined_test_details = {}
                    refined_tests = ""
                    low_confidence_message = ""
                    if coder_confidence < self.low_confidence_threshold:
                        low_confidence_message = "I'm not really sure if this solution is correct..."
                        self._log(f"Adding low confidence message to test generator: {low_confidence_message}", "yellow")
                    
                    if "current_tests" not in testing_blackboard:
                        tests_generated, test_code, test_gen_details = self.tester.generate_tests(
                            prompt, declaration, full_code, entry_point, extra_instructions=low_confidence_message
                        )
                        if not tests_generated:
                            self._log(f"Could not generate tests: {test_code}", "red")
                            
                            # Store iteration data
                            iteration_data = {
                                "iteration": iterations,
                                "code": full_code,
                                "feedback": f"Test generation failed: {test_code}",
                                "success": False,
                                "compilation": compile_details,
                                "test_generation": test_gen_details,
                                "confidence": {
                                    "planner": planner_confidence,
                                    "coder": coder_confidence,
                                    "tester": 0  # No tester confidence if test generation fails
                                }
                            }
                            iterations_data.append(iteration_data)
                            
                            # Exit with failure
                            success = False
                            exit_reason = "test_generation_failed"
                            break
                    else:
                        test_code = testing_blackboard["current_tests"]
                        
                    # Either we have tests from a previous iteration or we just generated them
                    # Check tester confidence in the test code
                    tester_system_prompt = self.tester.system_prompt
                    tester_confidence, tester_explanation = self.confidence_checker.check_confidence(
                        tester_system_prompt,
                        test_code,
                        "Write tests for the following code:\n" + full_code,
                        "" if "test_feedback" not in testing_blackboard else testing_blackboard["test_feedback"]
                    )
                    self._log(f"Tester confidence in tests: {tester_confidence}/100", "cyan")
                    self._log(f"Tester explanation of confidence: {tester_explanation}", "cyan")
                    # Check if tester has low confidence
                    if tester_confidence < self.low_confidence_threshold:
                        # Step 3: Check test quality (Skip this step if using prebuilt tests)
                        self._log(f"\nSTEP 3: CHECKING TEST QUALITY...", "cyan")
                        test_low_confidence_message = "I'm not really sure if these tests are correct..."
                        self._log(f"Adding low confidence message to test checker: {test_low_confidence_message}", "yellow")
                        self._log("Checking and refining tests...", "cyan")
                        test_quality_feedback = self.test_checker.check_tests(
                            prompt, declaration, full_code, test_code, entry_point, extra_instructions=test_low_confidence_message
                        )
                    
                        self._log(f"Test quality feedback: {test_quality_feedback}", "cyan")
                        
                        # Step 4: Refine tests based on feedback
                        self._log(f"\nSTEP 4: REFINING TESTS...", "cyan")
                        extra_instructions = f"""
    Based on the review, improve your tests to better match the requirements.
    Feedback on your tests:
    {test_quality_feedback}
    """
                        tests_refined, refined_test_code, refined_test_details = self.tester.generate_tests(
                            prompt, declaration, full_code, entry_point, extra_instructions=extra_instructions
                        )      
                        if not tests_refined:
                            self._log(f"Could not refine tests: {refined_test_code}", "red")
                            self._log(f"Falling back to original tests", "yellow")
                        else:
                            test_code = refined_test_code
                
                # Step 5: Run tests
                self._log(f"\nSTEP 5: RUNNING TESTS...", "cyan")
                tests_pass, test_output, test_details = self.tester.run_tests(
                    declaration, full_code, test_code, entry_point
                )
                # Store overall feedback
                if tests_pass:
                    feedback = "Code looks good. All tests passed."
                else:
                    # Generate detailed feedback for test failures
                    detailed_test_feedback, analysis_details = self.tester.generate_detailed_feedback(
                        prompt, declaration, full_code, test_code, test_output
                    )
                    feedback = detailed_test_feedback
                    test_details["analysis"] = analysis_details

                testing_blackboard["test_feedback"] = feedback
                testing_blackboard["current_tests"] = test_code

                # Store iteration data
                iteration_data = {
                    "iteration": iterations,
                    "code": full_code,
                    "feedback": feedback,
                    "success": tests_pass,
                    "compilation": compile_details,
                    "test_generation": test_gen_details,
                    "test_quality_feedback": test_quality_feedback,
                    "refined_tests": refined_test_details,
                    "test_execution": test_details,
                    "confidence": {
                        "planner": planner_confidence,
                        "coder": coder_confidence,
                        "tester": tester_confidence
                    }
                }
                iterations_data.append(iteration_data)
                
                # Exit early if tests pass
                if tests_pass:
                    self._log(f"Tests passed! Exiting early.", "green", always=True)
                    success = True
                    exit_reason = "tests_passed"
                    break
                
                # Exit if low confidence across all agents
                agents_confidence = {
                    "planner": planner_confidence,
                    "coder": coder_confidence,
                    "tester": tester_confidence
                }
                
                # If all agents have low confidence and plan restart is not disabled, try restarting with new plan
                restart_needed = False
                if not self.disable_plan_restart:
                    restart_needed = self.planner.evaluate_confidence(agents_confidence)
                
                if restart_needed:
                    self._log(f"Low confidence detected across agents. Restarting planning...", "yellow", always=True)
                    planning_attempts += 1
                    break
                
                # Otherwise refine the code
                if iterations < self.max_iterations - 2:
                    self._log(f"\nRefining code based on test failures...", "cyan")
                    refined_code = self.coder.refine_code(
                        prompt, full_code, feedback
                    )[0]
                    
                    # Check coder confidence in refined code
                    refinement_prompt = f"Refine code based on feedback: {feedback}"
                    coder_confidence, coder_explanation = self.confidence_checker.check_confidence(
                        coder_system_prompt,
                        refined_code,
                        refinement_prompt
                    )
                    
                    self._log(f"Coder confidence in refined code: {coder_confidence}/100", "cyan")
                    self._log(f"Coder explanation: {coder_explanation}", "cyan")
                    
                    current_coder_output = refined_code
                
                iterations += 1
            
            # When plan-based iteration completes, either from success or max iterations
            # No need for break here since we're not in a loop
        
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
        
        # If no successful iterations, use the last attempted code or empty string
        if not full_code and iterations_data:
            last_iteration = iterations_data[-1]
            full_code = last_iteration.get("code", "")
        
        return [full_code], generation_details
    
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
    keep_generated_function_signature: bool = True,
    tester_knows_cases: bool = False,
    disable_plan_restart: bool = False,
    disable_system_prompts: bool = False,
    no_planner: bool = False,
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
        tester_knows_cases: Whether to use the test cases from the dataset
        disable_plan_restart: Whether to disable restarting from plan when confidence is low
        disable_system_prompts: Whether to disable all system prompts
        no_planner: Whether to skip the planning phase entirely
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
        keep_generated_function_signature=keep_generated_function_signature,
        tester_knows_cases=tester_knows_cases,
        disable_plan_restart=disable_plan_restart,
        disable_system_prompts=disable_system_prompts,
        no_planner=no_planner,
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
                        help="Output file path (default: confidence_multi_agent_results_{language}_{planner_model_name}_{coder_model_name}_{tester_model_name}.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of samples to process")
    parser.add_argument("--keep_generated_function_signature", type=bool, default=True,
                        help="Keep the generated function signature")
    parser.add_argument("--tester_knows_cases", action="store_true",
                        help="Use test cases from the dataset instead of generating tests")
    parser.add_argument("--confidence_threshold", type=int, default=70,
                        help="Threshold for considering an agent confident (0-100)")
    parser.add_argument("--low_confidence_threshold", type=int, default=30,
                        help="Threshold for considering an agent not confident (0-100)")
    parser.add_argument("--max_confidence", action="store_true",
                        help="Set confidence to maximum (100) for all models")
    parser.add_argument("--min_confidence", action="store_true",
                        help="Set confidence to minimum (0) for all models")
    parser.add_argument("--disable_plan_restart", action="store_true",
                        help="Disable restarting from plan when confidence is low")
    parser.add_argument("--disable_system_prompts", action="store_true",
                        help="Disable all system prompts for the models")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed logs")
    parser.add_argument("--no_planner", action="store_true",
                        help="Skip the planning phase entirely")
                        
    args = parser.parse_args()
    
    load_dotenv()
    
    # Override with environment variables if available
    planner_model_type = os.getenv("PLANNER_MODEL_TYPE", args.planner_model_type)
    planner_model_name = os.getenv("PLANNER_MODEL_NAME", args.planner_model_name)
    coder_model_type = os.getenv("CODER_MODEL_TYPE", args.coder_model_type)
    coder_model_name = os.getenv("CODER_MODEL_NAME", args.coder_model_name)
    tester_model_type = os.getenv("TESTER_MODEL_TYPE", args.tester_model_type)
    tester_model_name = os.getenv("TESTER_MODEL_NAME", args.tester_model_name)
    
    # Set max confidence if requested
    if args.max_confidence:
        args.confidence_threshold = 100
        args.low_confidence_threshold = 100
        print("Using maximum confidence settings (confidence=100, low_confidence=100)")
    
    # Set min confidence if requested
    if args.min_confidence:
        args.confidence_threshold = 0
        args.low_confidence_threshold = 0
        args.disable_plan_restart = True
        print("Using minimum confidence settings (confidence=0, low_confidence=0, plan restart disabled)")
    
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
        keep_generated_function_signature=args.keep_generated_function_signature,
        tester_knows_cases=args.tester_knows_cases,
        confidence_threshold=args.confidence_threshold,
        low_confidence_threshold=args.low_confidence_threshold,
        disable_plan_restart=args.disable_plan_restart,
        disable_system_prompts=args.disable_system_prompts,
        no_planner=args.no_planner,
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
        
        # Get the test code from the sample if tester_knows_cases is enabled
        test_code = sample.get("test") if args.tester_knows_cases else None
        
        final_code, generation_details = multi_agent.generate_code(
            prompt, 
            declaration=declaration, 
            entry_point=entry_point,
            test_code=test_code
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
    
    output_file = args.output_file or f"confidence_multi_agent_results_{args.language}_{args.planner_model_name}_{args.coder_model_name}_{args.tester_model_name}.jsonl"
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
