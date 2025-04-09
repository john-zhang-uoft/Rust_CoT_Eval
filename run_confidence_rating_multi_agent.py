#!/usr/bin/env python3
"""
Parallel runner for confidence-based multi-agent code generation system
"""

import os
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List
import multiprocessing
from tqdm import tqdm
import jsonlines
from dotenv import load_dotenv
from datasets import load_dataset

from models.confidence_rating_multi_agent_generation import create_confidence_multi_agent_model, get_prompt_from_sample


def process_sample(args, sample_with_idx, is_verbose=False):
    """
    Process a single sample using the confidence multi-agent system
    
    Args:
        args: Command line arguments
        sample_with_idx: Tuple containing (index, sample)
        is_verbose: Whether to print verbose output
        
    Returns:
        Result dictionary with all generation data
    """
    idx, sample = sample_with_idx
    
    try:
        # Create a new multi-agent model for this process
        multi_agent = create_confidence_multi_agent_model(
            planner_model_type=args.planner_model_type,
            planner_model_name=args.planner_model_name,
            coder_model_type=args.coder_model_type,
            coder_model_name=args.coder_model_name,
            tester_model_type=args.tester_model_type,
            tester_model_name=args.tester_model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            language=args.language,
            max_iterations=args.max_iterations,
            max_planning_attempts=args.max_planning_attempts,
            keep_generated_function_signature=args.keep_generated_function_signature,
            tester_knows_cases=args.tester_knows_cases,
            verbose=is_verbose
        )
        
        # Process sample
        prompt = get_prompt_from_sample(sample, args.language)
        declaration = sample["declaration"]
        entry_point = sample["entry_point"]
        
        # Add process ID to log
        process_id = multiprocessing.current_process().name
        if is_verbose:
            print(f"\n{'-'*80}")
            print(f"[Process {process_id}] Processing sample {idx+1}: {sample['task_id']}")
            print(f"{'-'*80}")
        
        # Generate code
        start_time = time.time()
        final_code, generation_details = multi_agent.generate_code(
            prompt, 
            declaration=declaration, 
            entry_point=entry_point,
            test_code=sample.get("test") if args.tester_knows_cases else None
        )
        final_code = final_code[0]
        duration = time.time() - start_time
        
        # Create result
        result = {
            "task_id": sample["task_id"],
            "entry_point": entry_point,
            "declaration": declaration,
            "prompt": prompt,
            "test": sample["test"],
            "final_code": final_code,
            "success": generation_details.get("success", False),
            "exit_reason": generation_details.get("exit_reason", "unknown"),
            "iterations": generation_details.get("iterations_data", []),
            "final_confidence": generation_details.get("final_confidence", {}),
            "canonical_solution": sample.get("canonical_solution", ""),
            "process_id": process_id,
            "duration": duration
        }
        
        if is_verbose:
            print(f"[Process {process_id}] Completed sample {idx+1}: {sample['task_id']} - Success: {result['success']}")
        
        return result
    
    except Exception as e:
        process_id = multiprocessing.current_process().name
        error_result = {
            "task_id": sample.get("task_id", f"error-{idx}"),
            "entry_point": sample.get("entry_point", ""),
            "error": str(e),
            "success": False,
            "exit_reason": "error",
            "process_id": process_id
        }
        if is_verbose:
            print(f"[Process {process_id}] Error processing sample {idx+1}: {error_result['task_id']} - {str(e)}")
        return error_result


def run_parallel(args):
    """
    Run the confidence multi-agent system in parallel
    
    Args:
        args: Command line arguments
        
    Returns:
        List of results
    """
    # Load dataset
    if args.dataset_path:
        print(f"Loading custom dataset from {args.dataset_path}")
        with jsonlines.open(args.dataset_path) as reader:
            samples = list(reader)
    else:
        print(f"Loading HumanEvalPack {args.language} dataset")
        samples = [s for s in load_dataset("bigcode/humanevalpack", args.language)["test"]]
    
    print(f"Loaded {len(samples)} samples")
    
    # Limit samples if requested
    if args.limit is not None:
        samples = samples[:args.limit]
        print(f"Limited to {len(samples)} samples")
    
    # Prepare samples with indices
    samples_with_idx = list(enumerate(samples))
    
    # Determine number of workers
    max_workers = min(args.max_workers, len(samples))
    print(f"Using {max_workers} workers to process {len(samples)} samples")
    
    # Process samples in parallel
    results = []
    
    # Create progress bar
    progress_bar = tqdm(total=len(samples), desc="Processing samples")
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_sample, args, sample_with_idx, args.verbose): 
            sample_with_idx[0] for sample_with_idx in samples_with_idx
        }
        
        # Process as they complete
        for future in as_completed(futures):
            sample_idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                # Log success status
                success_str = "✓" if result.get("success", False) else "✗"
                exit_reason = result.get("exit_reason", "unknown")
                task_id = result.get("task_id", f"unknown-{sample_idx}")
                
                # Update progress bar with descriptive message
                progress_bar.set_description(
                    f"Processing: {success_str} {task_id} ({exit_reason})"
                )
                progress_bar.update(1)
                
            except Exception as e:
                print(f"Error processing sample {sample_idx}: {str(e)}")
                results.append({
                    "task_id": f"error-{sample_idx}",
                    "error": str(e),
                    "success": False,
                    "exit_reason": "error"
                })
                progress_bar.update(1)
    
    progress_bar.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel confidence-based multi-agent code generation")
    parser.add_argument("--language", type=str, default="rust", choices=["rust"],
                        help="Programming language (only Rust supported currently)")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to custom dataset JSONL file (if not provided, uses HumanEval)")
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
                        help="Output file path (default: parallel_confidence_multi_agent_results_{language}_{planner_model_name}_{coder_model_name}_{tester_model_name}.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of samples to process")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed logs")
    parser.add_argument("--max_workers", type=int, default=16,
                        help="Maximum number of parallel workers")
    parser.add_argument("--keep_generated_function_signature", action="store_true", default=True,
                        help="Keep the generated function signature")
    parser.add_argument("--tester_knows_cases", action="store_true",
                        help="Use test cases from the dataset instead of generating tests")
                        
    args = parser.parse_args()
    
    load_dotenv()
    
    # Override with environment variables if available
    args.planner_model_type = os.getenv("PLANNER_MODEL_TYPE", args.planner_model_type)
    args.planner_model_name = os.getenv("PLANNER_MODEL_NAME", args.planner_model_name)
    args.coder_model_type = os.getenv("CODER_MODEL_TYPE", args.coder_model_type)
    args.coder_model_name = os.getenv("CODER_MODEL_NAME", args.coder_model_name)
    args.tester_model_type = os.getenv("TESTER_MODEL_TYPE", args.tester_model_type)
    args.tester_model_name = os.getenv("TESTER_MODEL_NAME", args.tester_model_name)
    
    # Record start time
    start_time = time.time()
    
    # Run in parallel
    results = run_parallel(args)
    
    # Calculate total duration
    total_duration = time.time() - start_time
    
    # Save results
    output_file = args.output_file or f"parallel_confidence_multi_agent_results_{args.language}_{args.planner_model_name}_{args.coder_model_name}_{args.tester_model_name}.jsonl"
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(results)
        
    print(f"Results saved to {output_file}")
    print(f"Total execution time: {total_duration:.2f} seconds")
    
    # Print statistics
    success_count = sum(1 for r in results if r.get("success", False))
    success_rate = success_count / len(results) if results else 0
    print(f"Success rate: {success_rate:.2%} ({success_count}/{len(results)})")
    
    exit_reasons = {}
    for r in results:
        exit_reason = r.get("exit_reason", "unknown")
        exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
    
    print("\nExit reason statistics:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({count/len(results):.2%})")
    
    # Calculate average duration per sample
    durations = [r.get("duration", 0) for r in results if "duration" in r]
    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"\nAverage duration per sample: {avg_duration:.2f} seconds")
        print(f"Total computation time: {sum(durations):.2f} seconds")
        print(f"Parallelization efficiency: {sum(durations)/total_duration:.2f}x faster")
