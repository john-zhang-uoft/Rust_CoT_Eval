#!/usr/bin/env python3

import os
import subprocess
import glob

def main():
    # Create the evaluations directory if it doesn't exist
    eval_dir = "./results/evaluations"
    os.makedirs(eval_dir, exist_ok=True)
    
    # Get all parsed jsonl files in the results directory
    parsed_files = glob.glob("./results/*_parsed.jsonl")
    
    if not parsed_files:
        print("No parsed files found in ./results directory")
        return
    
    print(f"Found {len(parsed_files)} parsed files to evaluate")
    
    # Process each file
    for file_path in parsed_files:
        print(f"Evaluating {file_path}...")
        
        # Extract the base filename without path and extension
        base_filename = os.path.basename(file_path)
        base_name_without_ext = os.path.splitext(base_filename)[0]
        
        # Create output file path in the evaluations directory
        output_file = os.path.join(eval_dir, f"{base_name_without_ext}_eval.jsonl")
        
        # Construct the command
        command = [
            "python3", 
            "evaluate_completions.py",
            file_path,
            "--output_file", output_file,
            "--language", "rust",
            "--timeout", "30"
        ]
        
        # Run the command
        try:
            subprocess.run(command, check=True)
            print(f"Successfully evaluated {file_path}")
            print(f"Results saved to {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {file_path}: {e}")
    
    print("All files evaluated")

if __name__ == "__main__":
    main() 