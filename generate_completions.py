"""Testing
from datasets import load_dataset

ds = load_dataset("bigcode/humaneval-x-bugs", "python")["test"]
idx = 0

def get_prompt_base(doc, language="python"):
    # See 
    # https://github.com/roG0d/CodeGeeX/blob/f66205b5f615a4eead9c26d7ec297e14738ea18d/codegeex/benchmark/evaluate_humaneval_x.py#L78
    # https://github.com/THUDM/CodeGeeX/pull/76#issuecomment-1500653190
    if language == "rust":
        main = "fn main(){}\n"
        prompt_base = main + doc["declaration"] + doc["prompt"]
    else:
        prompt_base = doc["prompt"]
    return prompt_base

prompt_base = get_prompt_base(ds[idx], language="python")
    
messages = [
    {
        "role": "user",
        "content": ds[idx]["instruction"],
    },
    {
        "role": "assistant",
        "content": prompt_base,
    },
]

gpt-4-0613
response = openai.ChatCompletion.create(
model="gpt-4-0613",
messages=messages
)
"""

import os
import argparse
import jsonlines
import termcolor
from dotenv import load_dotenv
from datasets import load_dataset
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import threading
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.code_generation_models import CodeGenerationModel, OpenAIChatModel, LambdaLabsModel
from parsers.content_parser import ContentParser, ParseError

_CITATION = """
@article{muennighoff2023octopack,
      title={OctoPack: Instruction Tuning Code Large Language Models}, 
      author={Niklas Muennighoff and Qian Liu and Armel Zebaze and Qinkai Zheng and Binyuan Hui and Terry Yue Zhuo and Swayam Singh and Xiangru Tang and Leandro von Werra and Shayne Longpre},
      journal={arXiv preprint arXiv:2308.07124},
      year={2023}
}
"""

LANGUAGE_TO_NAME = {
    "python": "Python",
    "cpp": "C++",
    "js": "JavaScript",
    "java": "Java",
    "go": "Go",
    "rust": "Rust",
}

def get_prompt_base(doc, language):
    # See 
    # https://github.com/roG0d/CodeGeeX/blob/f66205b5f615a4eead9c26d7ec297e14738ea18d/codegeex/benchmark/evaluate_humaneval_x.py#L78
    # https://github.com/THUDM/CodeGeeX/pull/76#issuecomment-1500653190
    if language == "rust":
        main = "fn main(){}\n"
        prompt_base = main + doc["declaration"]
    else:
        prompt_base = doc["declaration"]
    return prompt_base


def get_prompt_synthesize(doc, language="python"):
    # addon = f"Start your code with:\n{get_prompt_base(sample, language)}"
    addon = f"Do not include the ` character. Start your code with:\n{get_prompt_base(doc, language)}"
    return doc["instruction"] + "\n" + addon # Results in worse performance for GPT4

    return doc["instruction"] # Problem: Difficult for problems that have helper functions


def get_base_prompt_fix(doc, language="python", mode="tests"):
    if language == "rust":
        if mode == "tests":
            return "fn main(){}\n" + doc["declaration"]
        elif mode == "docs":
            return "fn main(){}\n" + doc["declaration"] + doc["prompt"]
        else:
            raise ValueError
    else:
        if mode == "tests":
            return doc["declaration"]
        elif mode == "docs":
            return doc["prompt"]
        else:
            raise ValueError

def get_prompt_fix(doc, language="python", mode="tests"):
    prompt_base = get_base_prompt_fix(doc, language, mode)
    func = prompt_base + doc["buggy_solution"]
    instruction = f'Fix bugs in {doc["entry_point"]}.'
    return func + "\n" + instruction

def get_prompt_explain_desc(doc, language="python"):
    if language == "rust":
        main = "fn main(){}\n"
        prompt_base = main + doc["declaration"]
    else:
        prompt_base = doc["declaration"]
    docstring_len = len(doc["docstring"])

    instruction = f"Provide a concise natural language description of the code using at most {docstring_len} characters."
    func = prompt_base + doc["canonical_solution"]

    return instruction + "\n" + func, docstring_len

def get_prompt_explain_syn(sample, desc, language="python"):
    instruction = f"Write functional code in {LANGUAGE_TO_NAME[language]} according to the description."
    addon = f"Start your code with:\n{get_prompt_base(sample, language)}"
    return desc + "\n" + instruction + "\n" + addon


def create_model(model_type: str, model_name: str, temperature: float, top_p: float) -> CodeGenerationModel:
    """
    Create a code generation model based on type
    
    Args:
        model_type: Type of model to create ('openai', 'lambdalabs', etc.)
        model_name: Name of the model
        temperature: Temperature parameter
        top_p: Top-p parameter
        
    Returns:
        An instance of a CodeGenerationModel
    """
    if model_type.lower() == 'openai':
        return OpenAIChatModel(model=model_name, temperature=temperature, top_p=top_p)
    elif model_type.lower() == 'lambdalabs':
        return LambdaLabsModel(model=model_name, temperature=temperature, top_p=top_p)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_completions(
    task: str,
    language: str,
    model: CodeGenerationModel,
    samples_per_problem: int = 1,
    limit: Optional[int] = None,
    verbose: bool = False,
    output_file: Optional[str] = None,
    max_workers: int = 16
):
    """
    Generate code completions for the given task and language
    
    Args:
        task: Task type ('humanevalfix', 'humanevalsynthesize', etc.)
        language: Programming language
        model: CodeGenerationModel instance to use
        samples_per_problem: Number of samples to generate per problem
        limit: Optional limit on number of problems to process
        verbose: Whether to print detailed information
        output_file: Custom output file path (if None, a default is used)
        max_workers: Maximum number of concurrent workers
    """
    # Set default output file if not provided
    if output_file is None:
        output_file = f"completions_{language}_{task}.jsonl"
    
    # Get checkpoint file path
    checkpoint_file = f"{output_file}.checkpoint"
        
    # Load the dataset
    samples = [s for s in load_dataset("bigcode/humanevalpack", language)["test"]]
    print(f"Loaded {len(samples)} samples from HumanEvalPack {language} dataset")
    
    # Limit samples if specified
    if limit is not None:
        samples = samples[:limit]
    
    # Handle specific tasks
    descriptions = None
    if task == "humanevalexplainsynthesize":
        with jsonlines.open(f"completions_{language}_humanevalexplaindescribe.jsonl", "r") as f:
            descriptions = [line["raw_generation"][0] for line in f]
    
    # Create thread-safe structures
    processed_samples = []
    completed_indices = set()
    samples_lock = threading.Lock()
    
    # Thread-safe parser errors counter
    parse_errors = 0
    errors_lock = threading.Lock()
    
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
        
        try:
            # Get the appropriate prompt based on the task
            if task == "humanevalfix":
                prompt = get_prompt_fix(sample, language=language, mode="tests")
            elif task == "humanevalsynthesize":
                prompt = get_prompt_synthesize(sample, language=language)
            elif task == "humanevalexplaindescribe":
                prompt, docstring_len = get_prompt_explain_desc(sample, language=language)
                gen = model.generate_code(prompt, samples_per_problem)
                sample["raw_generation"] = gen
                sample["generation"] = [gen_item[:docstring_len] for gen_item in gen]
                return sample  # Return early for this special case
            elif task == "humanevalexplainsynthesize":
                desc = descriptions[idx]
                prompt = get_prompt_explain_syn(sample, desc, language=language)
            else:
                raise ValueError(f"Unknown task: {task}")
                
            if verbose:
                print(f"Processing {sample['task_id']} ({idx + 1}/{len(samples)})...")
                
            # Generate solutions
            sample["raw_generation"] = model.generate_code(prompt, None, samples_per_problem)
            
            # Initialize parser
            parser = ContentParser()
            
            # Parse the generated solutions
            try:
                sample["generation"] = [
                    parser(prompt, generation_item, sample["entry_point"], extract_all=True) 
                    for generation_item in sample["raw_generation"]
                ]
            except ParseError as e:
                with errors_lock:
                    parse_errors += 1
                print(f"PARSE EXCEPTION for {sample['task_id']}: {e}")
                sample["generation"] = [""] * samples_per_problem
                sample["parse_error"] = str(e)
                
            # Print details if verbose
            if verbose:
                for i in range(samples_per_problem):
                    print(termcolor.colored(sample["entry_point"], "yellow", attrs=["bold"]))
                    print(termcolor.colored(prompt, "yellow"))
                    print(termcolor.colored(sample["canonical_solution"], "red"))
                    print(termcolor.colored(sample["generation"][i], "green")+"\n\n")
                    
            return sample
        except Exception as e:
            import traceback
            print(f"Error processing sample {sample['task_id']}: {str(e)}")
            traceback.print_exc()
            # Return the sample with an error field
            sample["error"] = str(e)
            return sample
        finally:
            # Update progress bar
            progress_bar.update(1)
    
    # Calculate number of batches
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
                for future in as_completed(future_to_idx.keys()):
                    try:
                        result = future.result(timeout=60)  # 1-minute timeout for generation
                        if result is not None:  # Skip None results (already processed)
                            batch_results.append(result)
                    except Exception as e:
                        idx = future_to_idx[future]
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


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate code completions for HumanEval tasks")
    parser.add_argument("--task", type=str, default="humanevalsynthesize", 
                        choices=["humanevalfix", "humanevalsynthesize", 
                                 "humanevalexplaindescribe", "humanevalexplainsynthesize"],
                        help="Task to generate completions for")
    parser.add_argument("--language", type=str, default="rust",
                        choices=list(LANGUAGE_TO_NAME.keys()),
                        help="Programming language to generate completions for")
    parser.add_argument("--model_type", type=str, default="lambdalabs",
                        choices=["openai", "lambdalabs"],
                        help="Type of model to use")
    parser.add_argument("--model_name", type=str, default="llama3.2-3b-instruct",
                        help="Name of the model to use")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature parameter for generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p parameter for generation")
    parser.add_argument("--samples", type=int, default=1,
                        help="Number of samples to generate per problem")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of problems to process (defaults to all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Custom output file path")
    parser.add_argument("--max_workers", type=int, default=16,
                       help="Maximum number of concurrent workers for processing samples")
                        
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    
    # Override from environment variables if not specified
    model_type = os.getenv("MODEL_TYPE", args.model_type)
    model_name = os.getenv("MODEL", args.model_name)
    temperature = float(os.getenv("TEMPERATURE", args.temperature))
    top_p = float(os.getenv("TOP_P", args.top_p))
    samples = int(os.getenv("TIMES", args.samples))
    verbose = os.getenv("VERBOSE", str(args.verbose)).lower() == "true"
    task = os.getenv("TASK", args.task)
    language = os.getenv("LANGUAGE", args.language)
    limit = int(os.getenv("LIMIT", args.limit)) if os.getenv("LIMIT") or args.limit else None
    max_workers = int(os.getenv("MAX_WORKERS", args.max_workers))
    
    # Create the model
    model = create_model(model_type, model_name, temperature, top_p)
    
    print(f"Evaluating on {task} for {language}")
    print(f"Using model: {model.model_name}")
    print(f"Settings: samples={samples}, temperature={temperature}, top_p={top_p}, max_workers={max_workers}")
    
    # Generate completions
    generate_completions(
        task=task,
        language=language,
        model=model,
        samples_per_problem=samples,
        limit=limit,
        verbose=verbose,
        output_file=args.output_file,
        max_workers=max_workers
    ) 