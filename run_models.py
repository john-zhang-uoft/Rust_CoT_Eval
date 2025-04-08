#!/usr/bin/env python3
"""
Script to run confidence multi-agent system with different models in different roles
"""

import os
import subprocess
import argparse
from datetime import datetime
import time
def load_models(filename):
    """Load model names from a file"""
    with open(filename, 'r') as f:
        models = [line.strip() for line in f if line.strip()]
    return models

def run_model_combination(planner_model, coder_model, tester_model, max_workers=None):
    """Run the confidence multi agent with the specified model combination"""
    # Create a unique output filename based on the model combination
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results_{planner_model.split('/')[-1]}_{coder_model.split('/')[-1]}_{tester_model.split('/')[-1]}_{timestamp}.jsonl"
    log_file = f"log_{planner_model.split('/')[-1]}_{coder_model.split('/')[-1]}_{tester_model.split('/')[-1]}_{timestamp}.txt"
    
    # Set environment variables for the models
    env = os.environ.copy()
    env["PLANNER_MODEL_TYPE"] = "lambdalabs"  # Assuming lambdalabs as default
    env["PLANNER_MODEL_NAME"] = planner_model
    env["CODER_MODEL_TYPE"] = "lambdalabs"
    env["CODER_MODEL_NAME"] = coder_model
    env["TESTER_MODEL_TYPE"] = "lambdalabs"
    env["TESTER_MODEL_NAME"] = tester_model
    
    # Construct and run the command
    command = f"python3 run_confidence_rating_multi_agent.py --output_file {results_file}"
    if max_workers:
        command += f" --max_workers {max_workers}"
    
    print(f"Running: {command} with models:")
    print(f"  Planner: {planner_model}")
    print(f"  Coder: {coder_model}")
    print(f"  Tester: {tester_model}")
    print(f"Results will be saved to: {results_file}")
    print(f"Log will be saved to: {log_file}")
    
    with open(log_file, 'w') as f:
        subprocess.run(command, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT)
    
    print(f"Completed run, results saved to {results_file}")
    print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Run confidence multi-agent system with different models")
    parser.add_argument("--models-file", default="models_to_test.txt", help="File containing model names")
    parser.add_argument("--same-model", action="store_true", help="Use the same model for all roles")
    parser.add_argument("--planner-only", action="store_true", help="Vary only the planner model")
    parser.add_argument("--coder-only", action="store_true", help="Vary only the coder model")
    parser.add_argument("--tester-only", action="store_true", help="Vary only the tester model")
    parser.add_argument("--default-model", default="llama3.3-70b-instruct-fp8", help="Default model to use for non-varying roles")
    parser.add_argument("--planner", help="Specific model to use as planner")
    parser.add_argument("--coder", help="Specific model to use as coder")
    parser.add_argument("--tester", help="Specific model to use as tester")
    parser.add_argument("--limit", type=int, help="Limit runs to a specific number")
    parser.add_argument("--max-workers", type=int, default=None, help="Max workers to use for parallel processing")
    args = parser.parse_args()
    
    # Load models from file
    models = load_models(args.models_file)
    print(f"Loaded {len(models)} models to test")
    
    # If specific models are provided, use them directly
    if args.planner and args.coder and args.tester:
        run_model_combination(args.planner, args.coder, args.tester, args.max_workers)
        return
    
    if args.same_model:
        # Run the same model for all three roles
        for model in models:
            run_model_combination(model, model, model, args.max_workers)
            # Sleep for 5 minutes
            time.sleep(600)
    elif args.planner_only:
        # Vary only the planner model
        for planner_model in models:
            run_model_combination(planner_model, args.default_model, args.default_model, args.max_workers)
            # Sleep for 5 minutes
            time.sleep(600)
    elif args.coder_only:
        # Vary only the coder model
        for coder_model in models:
            run_model_combination(args.default_model, coder_model, args.default_model, args.max_workers)
            # Sleep for 5 minutes
            time.sleep(600)
    elif args.tester_only:
        # Vary only the tester model
        for tester_model in models:
            run_model_combination(args.default_model, args.default_model, tester_model, args.max_workers)
            # Sleep for 5 minutes
            time.sleep(600)
    else:
        # Run all combinations (warning: can be a lot!)
        total_runs = len(models)**3
        if args.limit:
            total_runs = min(total_runs, args.limit)
        
        print(f"Warning: Running combinations will result in up to {total_runs} runs.")
        confirm = input("Are you sure you want to continue? (y/n): ")
        if confirm.lower() == 'y':
            run_count = 0
            for planner_model in models:
                for coder_model in models:
                    for tester_model in models:
                        run_model_combination(planner_model, coder_model, tester_model, args.max_workers)
                        run_count += 1
                        if args.limit and run_count >= args.limit:
                            print(f"Reached limit of {args.limit} runs. Stopping.")
                            return
        else:
            print("Aborted. Use --same-model, --planner-only, --coder-only, or --tester-only for a more targeted approach.")

if __name__ == "__main__":
    main() 