#!/usr/bin/env python3

import os
import subprocess
import glob

def main():
    # Get all .jsonl files in the results directory
    result_files = glob.glob("./results/*.jsonl")
    
    if not result_files:
        print("No result files found in ./results directory")
        return
    
    print(f"Found {len(result_files)} files to process")
    
    # Process each file
    for file_path in result_files:
        print(f"Processing {file_path}...")
        
        # Construct the command
        base_name = os.path.splitext(file_path)[0]
        output_file = f"{base_name}_parsed.jsonl"
        
        command = [
            "python3", 
            "parse_completions_with_helper_functions.py",
            file_path,
            "--output_file", output_file,
            "--generation_field_name", "final_code"
        ]
        
        # Run the command
        try:
            subprocess.run(command, check=True)
            print(f"Successfully processed {file_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {file_path}: {e}")
    
    print("All files processed")

if __name__ == "__main__":
    main() 