#!/usr/bin/env python3
"""
Script to run all combinations of models as generator and reviewer,
with and without keeping generated function signatures.
"""

import os
import subprocess
import itertools
from datetime import datetime
import time

def read_models(filename):
    """Read model names from file, stripping whitespace."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def run_command(gen_model, review_model, keep_signature, output_file):
    """Run the multiagent completions command with given parameters."""
    cmd = [
        "python3", "run_multiagent_completions.py",
        "--max_workers=16",
        "--rust_dir=/home/john/Thesis/Rust_CoT_Eval/rust",
        "--verbose",
        f"--gen_model_name={gen_model}",
        f"--review_model_name={review_model}",
        f"--output_file={output_file}"
    ]
    
    if keep_signature:
        cmd.append("--keep_generated_function_signature")
    
    print(f"\nRunning command: {' '.join(cmd)}")
    print(f"Output will be saved to: {output_file}")
    
    try:
        with open("latest_trial.txt", "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(f"Successfully completed run for {gen_model} + {review_model}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False

def main():
    # Read models from file
    models = read_models("models_to_test.txt")
    print(f"Loaded {len(models)} models from models_to_test.txt")
    
    # Create timestamp for unique output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate all possible pairs of models
    model_pairs = list(itertools.permutations(models, 2))
    print(f"Will run {len(model_pairs)} model combinations")
    
    # Run each combination with both signature options
    for gen_model, review_model in model_pairs:
        # Create unique output filename
        base_filename = f"multiagent_completions_rust_humanevalsynthesize_{gen_model}_{review_model}_{timestamp}"
        
        # Run with keep_signature=True
        output_file = f"{base_filename}_with_signature.jsonl"
        print(f"\nRunning with keep_signature=True")
        success = run_command(gen_model, review_model, True, output_file)
        
        if success:
            # Wait a bit between runs to let system resources stabilize
            time.sleep(10)
            
            # Run with keep_signature=False
            output_file = f"{base_filename}_without_signature.jsonl"
            print(f"\nRunning with keep_signature=False")
            run_command(gen_model, review_model, False, output_file)
        
        # Wait between model combinations
        time.sleep(30)

if __name__ == "__main__":
    main() 