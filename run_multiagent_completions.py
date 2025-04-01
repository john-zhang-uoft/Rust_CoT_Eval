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

from models.multi_agent_generation import (
    MultiAgentModel,
    create_model,
)

from models.rust_code_reviewer_agent import RustCodeReviewerAgent

from generate_completions import (
    LANGUAGE_TO_NAME,
    get_prompt_synthesize,
    get_prompt_fix,
    get_prompt_explain_desc,
    get_prompt_explain_syn
)

from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, as_completed
import threading
import time
import gc
import math
    

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
    timeout: int = 60,
    thread_id: Optional[int] = None,
    sample_idx: Optional[int] = None,
    rust_dir: Optional[str] = None,
    keep_generated_function_signature: bool = False
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
        thread_id: Unique identifier for the thread
        sample_idx: Index of the sample for file naming
        rust_dir: Optional custom path to the Rust project directory
        keep_generated_function_signature: If True, use the entire implementation including function signatures
        
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
            RustCodeReviewerAgent(review_model, rust_dir=rust_dir)
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
        rust_dir=rust_dir,
        thread_id=thread_id,
        sample_idx=sample_idx,
        keep_generated_function_signature=keep_generated_function_signature
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
    task_id: Optional[str] = None,
    verbose: bool = False,
    output_file: Optional[str] = None,
    skip_review: bool = False,
    timeout: int = 60,
    max_workers: int = 16,
    custom_dataset: Optional[List[Dict[str, Any]]] = None,
    rust_dir: Optional[str] = None,
    keep_generated_function_signature: bool = False
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
        task_id: Optional specific task_id to run (e.g., "HumanEval/42")
        verbose: Whether to print detailed logs
        output_file: Custom output file path (if None, a default is used)
        skip_review: Whether to skip code review when cargo/compiler is not available
        timeout: Timeout in seconds for subprocess calls (compilation/test execution)
        max_workers: Maximum number of concurrent workers
        custom_dataset: Optional custom dataset to use instead of HumanEvalPack
        rust_dir: Optional custom path to the Rust project directory
        keep_generated_function_signature: If True, use the entire implementation including function signatures
    """
    
    # Set default output file if not provided
    if output_file is None:
        output_file = f"multiagent_completions_{language}_{task}.jsonl"
        if task_id:
            # If running a single task, add it to the filename
            task_id_suffix = task_id.replace("/", "_").replace("\\", "_")
            output_file = f"multiagent_completions_{language}_{task}_{task_id_suffix}.jsonl"
    
    # Get checkpoint file path
    checkpoint_file = f"{output_file}.checkpoint"
    
    # Run generate_completions with the multi-agent model
    print(f"Running multi-agent generation on {task} for {language}")
    print(f"Settings: samples={samples_per_problem}, temperature={temperature}, top_p={top_p}, max_iterations={max_iterations}, timeout={timeout}s, max_workers={max_workers}")
    print(f"Replace generated function signature: {keep_generated_function_signature}")
    print(f"Using concurrent processing with {max_workers} workers")
    
    # Load the dataset
    if custom_dataset is not None:
        # Use the provided custom dataset
        samples = custom_dataset
        print(f"Using custom dataset with {len(samples)} samples")
    else:
        # Load from HumanEvalPack
        samples = [s for s in load_dataset("bigcode/humanevalpack", language)["test"]]
        print(f"Loaded {len(samples)} samples from HumanEvalPack {language} dataset")
    
    # Filter by task_id if specified
    if task_id:
        original_count = len(samples)
        samples = [s for s in samples if s["task_id"] == task_id]
        if not samples:
            raise ValueError(f"Task ID '{task_id}' not found in the dataset")
        print(f"Filtered to 1 sample with task_id={task_id} (from {original_count} total samples)")
    # Limit samples if specified and no specific task_id
    elif limit is not None:
        samples = samples[:limit]
    
    # Create a thread-safe counter for parse errors
    parse_errors = 0
    
    # Create a thread-safe list for processed samples
    processed_samples = []
    completed_indices = set()
    samples_lock = threading.Lock()
    
    # Create a thread-safe tqdm progress bar
    progress_bar = tqdm(total=len(samples))
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        try:
            with jsonlines.open(checkpoint_file, "r") as reader:
                for item in reader:
                    processed_samples.append(item)
                    completed_indices.add(samples.index(item))
                    progress_bar.update(1)
            print(f"Loaded {len(processed_samples)} samples from checkpoint")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            # Continue without the checkpoint
            processed_samples = []
    
    # Helper function to save checkpoint
    def save_checkpoint(samples_to_save):
        if not samples_to_save:
            return
            
        try:
            with jsonlines.open(checkpoint_file, "w") as writer:
                writer.write_all(samples_to_save)
            print(f"Checkpoint saved with {len(samples_to_save)} samples")
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
    
    # Function to process a single sample
    def process_sample(idx, sample):
        nonlocal parse_errors
        
        if idx in completed_indices:
            return None  # Skip already processed samples
            
        # Create a thread ID based on the sample index for unique file paths
        # Adding task_id to make thread_id unique across different samples
        thread_id = idx + 1000 + hash(sample["task_id"]) % 10000  # Offset to avoid potential conflicts with process IDs
        local_multi_agent = None
        
        try:
            # Create a thread-local MultiAgentModel instance
            local_multi_agent = create_multi_agent_model(
                gen_model_type=gen_model_type,
                gen_model_name=gen_model_name,
                review_model_type=review_model_type,
                review_model_name=review_model_name,
                temperature=temperature,
                top_p=top_p,
                language=language,
                max_iterations=max_iterations,
                verbose=verbose and idx % max_workers == 0,  # Only enable verbose for some threads to avoid output mess
                skip_review=skip_review,
                timeout=timeout,
                thread_id=thread_id,
                sample_idx=idx,  # Pass the sample index for file naming
                rust_dir=rust_dir,
                keep_generated_function_signature=keep_generated_function_signature
            )
            
            # Get the appropriate prompt based on the task
            if task == "humanevalfix":
                prompt = get_prompt_fix(sample, language=language, mode="tests")
            elif task == "humanevalsynthesize":
                prompt = get_prompt_synthesize(sample, language=language)
            elif task == "humanevalexplaindescribe":
                prompt, docstring_len = get_prompt_explain_desc(sample, language=language)
                gen = local_multi_agent.generate_code(
                    prompt, 
                    samples_per_problem,
                    declaration=sample["declaration"],
                    entry_point=sample["entry_point"]
                )
                sample["raw_generation"] = gen
                sample["generation"] = [gen_item[:docstring_len] for gen_item in gen]
                return sample  # Return early for this special case
            elif task == "humanevalexplainsynthesize":
                with jsonlines.open(f"completions_{language}_humanevalexplaindescribe.jsonl", "r") as f:
                    descriptions = [line["raw_generation"][0] for line in f]
                desc = descriptions[idx]
                prompt = get_prompt_explain_syn(sample, desc, language=language)
            elif task == "custom":
                # For custom tasks, use the prompt directly from the sample
                prompt = sample.get("prompt", "")
                if not prompt:
                    raise ValueError(f"Sample {idx} does not have a prompt field")
            else:
                raise ValueError(f"Unknown task: {task}")
                
            if verbose:
                print(f"Processing {sample['task_id']} ({idx + 1}/{len(samples)})...")
                
            # Generate solutions with explicit declaration and entry_point
            sample["raw_generation"] = local_multi_agent.generate_code(
                prompt, 
                samples_per_problem,
                declaration=sample["declaration"],
                entry_point=sample["entry_point"]
            )
            
            # Print details if verbose
            if verbose:
                for i in range(samples_per_problem):
                    print(f"\nTask: {sample['task_id']}")
                    print(f"Entry point: {sample['entry_point']}")
                    print("-" * 40)
                    print("Raw generation:")
                    print(sample["raw_generation"][i])
                    print("-" * 40)
            
            return sample
        except Exception as e:
            import traceback
            print(f"Error processing sample {sample['task_id']}: {str(e)}")
            traceback.print_exc()
            # Return the sample with an error field
            sample["error"] = str(e)
            return sample
        finally:
            # Clean up thread-specific resources
            if local_multi_agent and hasattr(local_multi_agent, '_reviewer'):
                try:
                    local_multi_agent._reviewer.cleanup()
                except Exception as e:
                    print(f"Warning: Failed to clean up resources for sample {sample['task_id']}: {str(e)}")
            
            # Update progress bar
            progress_bar.update(1)
    
    # Calculate number of batches (process in smaller groups to manage resources better)
    remaining_samples = [s for i, s in enumerate(samples) if i not in completed_indices]
    batch_size = min(max(1, len(remaining_samples) // 4), max_workers * 2)  # Reasonable batch size
    batches = [remaining_samples[i:i + batch_size] for i in range(0, len(remaining_samples), batch_size)]
    
    print(f"Processing {len(remaining_samples)} remaining samples in {len(batches)} batches of ~{batch_size} samples each")
    
    # Process batches with ThreadPoolExecutor
    for batch_idx, batch in enumerate(batches):
        print(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch)} samples")
        batch_results = []
        
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as executor:
            # Submit only this batch to the executor
            future_to_idx = {executor.submit(process_sample, samples.index(sample), sample): samples.index(sample) 
                            for sample in batch}

            try:
                # Process results as they complete
                for future in as_completed(future_to_idx):  # Use the futures (keys) here
                    try:
                        result = future.result(timeout=timeout * max_iterations * 2)  # Longer timeout based on iterations
                        if result is not None:  # Skip None results (already processed)
                            batch_results.append(result)
                    except Exception as e:
                        idx = future_to_idx.get(future)
                        print(f"Sample at index {idx} generated an exception: {str(e)}")
                
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received, canceling all pending tasks...")
                # Cancel all pending futures on keyboard interrupt
                for future in future_to_idx.keys():
                    if not future.done():
                        future.cancel()
                # Save checkpoint before exiting
                all_processed = processed_samples + batch_results
                all_processed.sort(key=lambda s: samples.index(s))
                save_checkpoint(all_processed)
                raise
        
        # Add batch results to processed samples
        with samples_lock:
            processed_samples.extend(batch_results)
            # Save checkpoint after each batch
            save_checkpoint(processed_samples)
        
        # Force garbage collection between batches
        gc.collect()
        
        # Small delay between batches to allow system resources to stabilize
        time.sleep(1)
    
    # Close progress bar
    progress_bar.close()
    
    # Sort the processed samples by their original index (to maintain order)
    processed_samples.sort(key=lambda s: samples.index(s))
    
    # Print error rate
    if verbose:
        print(f"Parse error rate: {parse_errors / len(samples):.2%}")
    
    # Save results to file
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(processed_samples)
    
    # Remove checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print(f"Removed checkpoint file: {checkpoint_file}")
        except Exception as e:
            print(f"Warning: Failed to remove checkpoint file: {str(e)}")
        
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run multi-agent code generation on HumanEval tasks")
    parser.add_argument("--task", type=str, default="humanevalsynthesize", 
                        choices=["humanevalfix", "humanevalsynthesize", 
                                 "humanevalexplaindescribe", "humanevalexplainsynthesize",
                                 "custom"],
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
    parser.add_argument("--task_id", type=str, default=None,
                        help="Run on a specific task ID only (e.g., 'HumanEval/42')")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Custom output file path")
    parser.add_argument("--skip_review", action="store_true",
                        help="Skip code review when cargo/compiler is not available")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout in seconds for compilation and test execution")
    parser.add_argument("--max_workers", type=int, default=16,
                       help="Maximum number of concurrent workers for processing samples")
    parser.add_argument("--rust_dir", type=str, default=None,
                        help="Custom path to the Rust project directory")
    parser.add_argument("--keep_generated_function_signature", action="store_true",
                        help="Keeps generated function signature")
                        
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
    task_id = os.getenv("TASK_ID", args.task_id)
    
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
        task_id=task_id,
        verbose=verbose,
        output_file=args.output_file,
        skip_review=args.skip_review,
        timeout=int(os.getenv("TIMEOUT", args.timeout)),
        max_workers=int(os.getenv("MAX_WORKERS", args.max_workers)),
        rust_dir=args.rust_dir,
        keep_generated_function_signature=args.keep_generated_function_signature
    )
