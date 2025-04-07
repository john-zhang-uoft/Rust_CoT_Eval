#!/usr/bin/env python3
"""
Script to run confidence multi-agent system with different model combinations
"""

import os
import subprocess
import time
from datetime import datetime
import argparse

def read_models(file_path):
    """Read models from the specified file"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def create_results_dir():
    """Create a results directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def run_confidence_multi_agent(model, results_dir, language="rust", max_workers=16, limit=None):
    """Run the confidence multi-agent system with the specified model"""
    output_file = os.path.join(results_dir, f"results_{model.replace('/', '_')}.jsonl")
    
    cmd = [
        "python", "run_confidence_multi_agent.py",
        "--language", language,
        "--planner_model_name", model,
        "--coder_model_name", model,
        "--tester_model_name", model,
        "--output_file", output_file,
        "--max_workers", str(max_workers)
    ]
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    print(f"\nRunning with model: {model}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    duration = time.time() - start_time
    print(f"Completed in {duration:.2f} seconds")
    
    if process.returncode != 0:
        print(f"Error running with model {model}:")
        print(stderr.decode())
    else:
        print(f"Results saved to: {output_file}")
    
    return process.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Run confidence multi-agent system with different models")
    parser.add_argument("--models_file", type=str, default="models_to_test.txt",
                        help="Path to file containing models to test")
    parser.add_argument("--language", type=str, default="rust",
                        help="Programming language to use")
    parser.add_argument("--max_workers", type=int, default=16,
                        help="Maximum number of parallel workers")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of samples to process per model")
    
    args = parser.parse_args()
    
    # Read models
    models = read_models(args.models_file)
    print(f"Found {len(models)} models to test")
    
    # Create results directory
    results_dir = create_results_dir()
    print(f"Results will be saved to: {results_dir}")
    
    # Run tests
    successful_runs = 0
    for model in models:
        success = run_confidence_multi_agent(
            model=model,
            results_dir=results_dir,
            language=args.language,
            max_workers=args.max_workers,
            limit=args.limit
        )
        if success:
            successful_runs += 1
    
    print(f"\nSummary:")
    print(f"Total models tested: {len(models)}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {len(models) - successful_runs}")
    print(f"Results directory: {results_dir}")

if __name__ == "__main__":
    main() 