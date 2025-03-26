#!/usr/bin/env python3
"""
Run multi-agent generation with generate_completions on the HumanEval dataset.
This script combines the multi-agent approach for code generation with the
generate_completions function to evaluate on the HumanEval benchmark.
"""

import os
import argparse
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import jsonlines

from multi_agent_generation import (
    MultiAgentModel,
    create_model,
    get_prompt_from_sample,
    CodeReviewerAgent
)
from generate_completions import (
    generate_completions,
    LANGUAGE_TO_NAME,
    get_prompt_synthesize,
    get_prompt_fix,
    get_prompt_explain_desc,
    get_prompt_explain_syn
)
from code_generation_models import CodeGenerationModel


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
    timeout: int = 60
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
        timeout: Timeout in seconds for subprocess calls (compilation/test execution)
        
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
    
    print(f"Using generator model: {gen_model.model_name}")
    print(f"Using reviewer model: {review_model.model_name}")
    
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
            if not skip_review:
                print("Use --skip_review to suppress this warning")
                max_iterations = 1  # No iterations if cargo not available
    
    # Create the multi-agent model
    multi_agent = MultiAgentModel(
        gen_model=gen_model,
        review_model=review_model,
        language=language,
        max_iterations=max_iterations if cargo_available else 1,
        verbose=verbose
    )
    
    print(f"Using multi-agent model: {multi_agent.model_name}")
    return multi_agent


def run_multi_agent_completions(
    task: str,
    language: str,
    gen_model_type: str,
    gen_model_name: str,
    review_model_type: str,
    review_model_name: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_iterations: int = 3,
    samples_per_problem: int = 1,
    limit: Optional[int] = None,
    verbose: bool = False,
    output_file: Optional[str] = None,
    skip_review: bool = False,
    timeout: int = 60
) -> None:
    """
    Run multi-agent code generation on HumanEval tasks
    
    Args:
        task: Task type ('humanevalfix', 'humanevalsynthesize', etc.)
        language: Programming language
        gen_model_type: Type of model for code generation
        gen_model_name: Name of the model for code generation
        review_model_type: Type of model for code review
        review_model_name: Name of the model for code review
        temperature: Temperature parameter for generation
        top_p: Top-p parameter for generation
        max_iterations: Maximum number of refinement iterations
        samples_per_problem: Number of samples to generate per problem
        limit: Optional limit on number of problems to process
        verbose: Whether to print detailed logs
        output_file: Custom output file path (if None, a default is used)
        skip_review: Whether to skip code review when cargo/compiler is not available
        timeout: Timeout in seconds for subprocess calls (compilation/test execution)
    """
    # Create the multi-agent model
    multi_agent = create_multi_agent_model(
        gen_model_type=gen_model_type,
        gen_model_name=gen_model_name,
        review_model_type=review_model_type,
        review_model_name=review_model_name,
        temperature=temperature,
        top_p=top_p,
        language=language,
        max_iterations=max_iterations,
        verbose=verbose,
        skip_review=skip_review,
        timeout=timeout
    )
    
    # Set default output file if not provided
    if output_file is None:
        output_file = f"multiagent_completions_{language}_{task}.jsonl"
    
    # Run generate_completions with the multi-agent model
    print(f"Running multi-agent generation on {task} for {language}")
    print(f"Settings: samples={samples_per_problem}, temperature={temperature}, top_p={top_p}, max_iterations={max_iterations}, timeout={timeout}s")
    
    # Custom function to run generate_completions that passes declaration and entry_point
    from datasets import load_dataset
    from content_parser import ContentParser, ParseError
    from tqdm import tqdm
    
    # Load the dataset
    samples = [s for s in load_dataset("bigcode/humanevalpack", language)["test"]]
    print(f"Loaded {len(samples)} samples from HumanEvalPack {language} dataset")
    
    # Initialize parser
    parser = ContentParser()
    parse_errors = 0
    
    # Process each sample with explicit declaration and entry_point
    for idx, sample in enumerate(tqdm(samples)):
        if limit is not None and idx >= limit:
            break
            
        # Get the appropriate prompt based on the task
        if task == "humanevalfix":
            prompt = get_prompt_fix(sample, language=language, mode="tests")
        elif task == "humanevalsynthesize":
            prompt = get_prompt_synthesize(sample, language=language)
        elif task == "humanevalexplaindescribe":
            prompt, docstring_len = get_prompt_explain_desc(sample, language=language)
            gen = multi_agent.generate_code(
                prompt, 
                samples_per_problem,
                declaration=sample["declaration"],
                entry_point=sample["entry_point"]
            )
            sample["raw_generation"] = gen
            sample["generation"] = [gen_item[:docstring_len] for gen_item in gen]
            continue  # Skip the normal parsing for this task
        elif task == "humanevalexplainsynthesize":
            with jsonlines.open(f"completions_{language}_humanevalexplaindescribe.jsonl", "r") as f:
                descriptions = [line["raw_generation"][0] for line in f]
            desc = descriptions[idx]
            prompt = get_prompt_explain_syn(sample, desc, language=language)
        else:
            raise ValueError(f"Unknown task: {task}")
            
        if verbose:
            print(f"Processing {sample['task_id']} ({idx + 1}/{len(samples)})...")
            
        # Generate solutions with explicit declaration and entry_point
        sample["raw_generation"] = multi_agent.generate_code(
            prompt, 
            samples_per_problem,
            declaration=sample["declaration"],
            entry_point=sample["entry_point"]
        )
        
        # Parse the generated solutions
        try:
            sample["generation"] = [
                parser(prompt, generation_item, sample["entry_point"]) 
                for generation_item in sample["raw_generation"]
            ]
        except ParseError as e:
            parse_errors += 1
            print(f"PARSE EXCEPTION for {sample['task_id']}: {e}")
            sample["generation"] = [""] * samples_per_problem
            
        # Print details if verbose
        if verbose:
            for i in range(samples_per_problem):
                print(f"\nTask: {sample['task_id']}")
                print(f"Entry point: {sample['entry_point']}")
                print("-" * 40)
                print("Raw generation:")
                print(sample["raw_generation"][i])
                print("-" * 40)
                print("Parsed generation:")
                print(sample["generation"][i])
                print("-" * 40)
    
    # Print error rate
    if verbose:
        print(f"Parse error rate: {parse_errors / len(samples):.2%}")
    
    # Save results to file  
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(samples)
        
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run multi-agent code generation on HumanEval tasks")
    parser.add_argument("--task", type=str, default="humanevalsynthesize", 
                        choices=["humanevalfix", "humanevalsynthesize", 
                                 "humanevalexplaindescribe", "humanevalexplainsynthesize"],
                        help="Task to generate completions for")
    parser.add_argument("--language", type=str, default="rust",
                        choices=list(LANGUAGE_TO_NAME.keys()),
                        help="Programming language to generate completions for")
    parser.add_argument("--gen_model_type", type=str, default="lambdalabs",
                        choices=["openai", "lambdalabs"],
                        help="Type of model for code generation")
    parser.add_argument("--gen_model_name", type=str, default="llama3.3-70b-instruct-fp8",
                        help="Name of the model for code generation")
    parser.add_argument("--review_model_type", type=str, default="lambdalabs",
                        choices=["openai", "lambdalabs"],
                        help="Type of model for code review")
    parser.add_argument("--review_model_name", type=str, default="llama3.3-70b-instruct-fp8",
                        help="Name of the model for code review")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature parameter for generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p parameter for generation")
    parser.add_argument("--max_iterations", type=int, default=3,
                        help="Maximum number of refinement iterations")
    parser.add_argument("--samples", type=int, default=1,
                        help="Number of samples to generate per problem")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of problems to process (defaults to all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Custom output file path")
    parser.add_argument("--skip_review", action="store_true",
                        help="Skip code review when cargo/compiler is not available")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout in seconds for compilation and test execution")
                        
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    
    # Override from environment variables if not specified
    gen_model_type = os.getenv("GEN_MODEL_TYPE", args.gen_model_type)
    gen_model_name = os.getenv("GEN_MODEL_NAME", args.gen_model_name)
    review_model_type = os.getenv("REVIEW_MODEL_TYPE", args.review_model_type)
    review_model_name = os.getenv("REVIEW_MODEL_NAME", args.review_model_name)
    temperature = float(os.getenv("TEMPERATURE", args.temperature))
    top_p = float(os.getenv("TOP_P", args.top_p))
    samples = int(os.getenv("TIMES", args.samples))
    verbose = os.getenv("VERBOSE", str(args.verbose)).lower() == "true"
    task = os.getenv("TASK", args.task)
    language = os.getenv("LANGUAGE", args.language)
    limit = int(os.getenv("LIMIT", args.limit)) if os.getenv("LIMIT") or args.limit else None
    
    # Run multi-agent completions
    run_multi_agent_completions(
        task=task,
        language=language,
        gen_model_type=gen_model_type,
        gen_model_name=gen_model_name,
        review_model_type=review_model_type,
        review_model_name=review_model_name,
        temperature=temperature,
        top_p=top_p,
        max_iterations=args.max_iterations,
        samples_per_problem=samples,
        limit=limit,
        verbose=verbose,
        output_file=args.output_file,
        skip_review=args.skip_review,
        timeout=int(os.getenv("TIMEOUT", args.timeout))
    )
