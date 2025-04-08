import os
import argparse
import tempfile
import jsonlines
from tqdm import tqdm
import json
import shutil
import subprocess
import signal
import time
import threading

# Timeout class for handling subprocess timeouts
class TimeoutError(Exception):
    pass

def timeout_handler(proc, timeout_seconds):
    """
    Handler function for timing out subprocesses.
    Kills the process after timeout_seconds.
    """
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        # Check if process has finished
        if proc.poll() is not None:
            return proc.returncode
        time.sleep(0.1)
    
    # Process still running after timeout, terminate it
    try:
        # Send SIGTERM and wait for a moment to allow clean termination
        proc.terminate()
        time.sleep(0.5)
        
        # If process is still running, force kill it
        if proc.poll() is None:
            proc.kill()
    except:
        pass
    
    raise TimeoutError("Process timed out")

def run_process_with_timeout(cmd, timeout_seconds=30):
    """
    Run a subprocess with a timeout.
    
    Args:
        cmd: Shell command to execute
        timeout_seconds: Maximum time to wait (in seconds)
        
    Returns:
        Return code of the process if successful, -1 for timeout
    """
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        return timeout_handler(process, timeout_seconds)
    except TimeoutError:
        print(f"Process timed out after {timeout_seconds} seconds: {cmd}")
        return -1

def evaluate_completions(input_file, k=1, language="rust", verbose=False, timeout=30):
    """
    Evaluate completions from input_file against unit tests
    and calculate top-k accuracy
    """
    # Define directories
    WD = os.path.dirname(os.path.abspath(__file__))
    RUST_DIR = os.path.join(WD, "rust")
    RUST_SRC = os.path.join(RUST_DIR, "src")
    RUST_BIN = os.path.join(RUST_SRC, "bin")
    RUST_TMP_DIR = os.path.join(RUST_DIR, "tmp")
    RUST_LOGS = os.path.join(RUST_TMP_DIR, "logs")
    RUST_EXT = ".rs"
    
    # Create mandatory tmp directories
    os.makedirs(RUST_TMP_DIR, exist_ok=True)
    os.makedirs(RUST_LOGS, exist_ok=True)
    os.makedirs(RUST_SRC, exist_ok=True)
    os.makedirs(RUST_BIN, exist_ok=True)
    
    # Load samples from input file
    with jsonlines.open(input_file, "r") as reader:
        samples = list(reader)
    
    print(f"Loaded {len(samples)} samples from {input_file}")
    
    # Results tracking
    results = []
    successful_samples = 0
    timeout_samples = 0
    
    # Process each sample
    for idx, sample in enumerate(tqdm(samples)):
        if "generation" not in sample or not sample["generation"]:
            if verbose:
                print(f"Warning: No generation field in sample {idx}")
            results.append({"task_id": sample.get("task_id", f"sample_{idx}"), "results": ["skipped"] * k})
            continue
        
        # Get test code
        if "test" not in sample:
            if verbose:
                print(f"Warning: No test field in sample {idx}")
            results.append({"task_id": sample.get("task_id", f"sample_{idx}"), "results": ["skipped"] * k})
            continue
        
        sample_results = []
        success = False
        
        # Test up to k generations (but not more than available)
        generations_to_test = min(k, len(sample["generation"]))
        for gen_idx in range(generations_to_test):
            generation = sample["generation"][gen_idx]
            if not generation:  # Skip empty generations
                sample_results.append("skipped")
                continue
            
            # Create temporary file and write the code
            with tempfile.NamedTemporaryFile(dir=RUST_BIN, delete=False) as f:
                # Temporal file name
                file_prefix = sample["task_id"].lower().replace("/", "_")
                file_name = file_prefix + RUST_EXT
                full_path = os.path.join(RUST_BIN, file_name)
                
                # Rename the temp file
                os.rename(f.name, full_path)
                
                # Get the test code and the solution
                rust_test_code = sample["test"]
                
                # Prepare the full code (solution + test code)
                # Different languages might need different handling here
                if language == "rust":
                    # For Rust, we need to prepend the function declaration
                    if "declaration" in sample:
                        if "fn main(){}" not in sample["declaration"]:  # Remove the main function if it exists
                            declaration = "\nfn main(){}" + sample["declaration"]
                        else:
                            declaration = sample["declaration"]
                    else:
                        declaration = ""
                    
                    # Combine declaration, implementation and test code
                    rust_code = declaration + generation + "\n" + rust_test_code
                
                # Log the code being tested
                if verbose:
                    print(f"\nTesting {sample['task_id']} (Generation {gen_idx+1}/{generations_to_test}):")
                    print("-----CODE BEGIN-----")
                    print(rust_code)
                    print("-----CODE END-----")
                
                # Write the combined code to the file
                with open(full_path, "w", encoding="utf-8") as code_file:
                    code_file.write(rust_code)
            
            # Proceed towards Rust binaries compilation. Therefore move to Rust module root dir.
            original_dir = os.getcwd()
            os.chdir(RUST_DIR)
            
            # Two possible outcomes: Pass OR Fail compilation
            log_filename = f"{file_prefix}_gen{gen_idx}.jsonl"
            log_path = os.path.join(RUST_LOGS, log_filename)
            cargo_check = f"cargo check --bin {file_prefix} --message-format json > {log_path} 2>&1"
            
            # Overwrite file content if it exists
            if os.path.exists(log_path):
                os.remove(log_path)
            
            # Run cargo check with timeout
            returned_val_compilation = run_process_with_timeout(cargo_check, timeout)
            
            if returned_val_compilation == -1:
                # Timeout occurred
                error_message = "failed: compilation timeout"
                sample_results.append(error_message)
                if verbose:
                    print(f"Generation {gen_idx+1}: {error_message}")
                
                # Return to original directory
                os.chdir(original_dir)
                continue
            
            # 0 means success
            if returned_val_compilation == 0:
                # Execution pipeline
                cargo_test = f"cargo test --bin {file_prefix} --message-format json >> {log_path} 2>&1"
                
                # Run with timeout
                returned_val_execution = run_process_with_timeout(cargo_test, timeout)
                
                if returned_val_execution == -1:
                    # Timeout occurred
                    error_message = "failed: execution timeout"
                    sample_results.append(error_message)
                    if verbose:
                        print(f"Generation {gen_idx+1}: {error_message}")
                elif returned_val_execution == 0:
                    sample_results.append("passed")
                    success = True
                    if verbose:
                        print(f"Generation {gen_idx+1}: PASSED")
                else:
                    error_message = "failed: execution error"
                    sample_results.append(error_message)
                    if verbose:
                        print(f"Generation {gen_idx+1}: {error_message}")
                        print("See logs at:", log_path)
            else:
                error_message = "failed: compilation error"
                sample_results.append(error_message)
                if verbose:
                    print(f"Generation {gen_idx+1}: {error_message}")
                    print("See logs at:", log_path)
            
            # Return to original directory
            os.chdir(original_dir)
            
            # If we've found a successful solution, we can stop testing additional generations
            # for this sample (if we're doing top-1 accuracy)
            if success and k == 1:
                break
        
        # Count timeout samples
        if any(result.endswith("timeout") for result in sample_results):
            timeout_samples += 1
        
        # Fill in remaining results if we didn't test all k generations
        while len(sample_results) < k:
            sample_results.append("skipped")
        
        # Record results for this sample
        results.append({
            "task_id": sample.get("task_id", f"sample_{idx}"), 
            "results": sample_results
        })
        
        # Update successful samples count
        if success:
            successful_samples += 1
    
    # Calculate top-k accuracy
    top_k_accuracy = successful_samples / len(samples)
    print(f"Top-{k} Accuracy: {top_k_accuracy:.2%} ({successful_samples}/{len(samples)})")
    print(f"Timeout Rate: {timeout_samples / len(samples):.2%} ({timeout_samples}/{len(samples)})")
    
    return results, top_k_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate code completions against test cases")
    parser.add_argument("input_file", help="Input JSONL file containing parsed completions")
    parser.add_argument("--k", type=int, default=1, help="Evaluate top-k accuracy (default: 1)")
    parser.add_argument("--language", default="rust", choices=["rust"], help="Programming language (default: rust)")
    parser.add_argument("--output_file", help="Output JSONL file for results (default: input_file with '_eval' suffix)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds for compiling/running each sample (default: 30)")
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output_file:
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = f"{base_name}_eval.jsonl"
    
    # Run evaluation
    results, accuracy = evaluate_completions(
        args.input_file, 
        k=args.k, 
        language=args.language, 
        verbose=args.verbose,
        timeout=args.timeout
    )
    
    # Save results to file
    with jsonlines.open(args.output_file, "w") as writer:
        writer.write_all(results)
    
    print(f"Evaluation results saved to {args.output_file}")