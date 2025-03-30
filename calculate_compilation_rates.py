#!/usr/bin/env python3
"""
Calculate compilation success rates for different difficulty levels of Rust problems
"""

import os
import json
import jsonlines
from tqdm import tqdm
import time
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
# Try to find the .env file in different locations
env_path = find_dotenv(usecwd=True)
if env_path:
    print(f"Loading environment variables from: {env_path}")
    load_dotenv(env_path)
else:
    print("Warning: No .env file found")

# Check if API keys are set
LAMBDALABS_API_KEY = os.getenv("LAMBDALABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not LAMBDALABS_API_KEY and not OPENAI_API_KEY:
    print("Error: No API keys found in environment. Please set LAMBDALABS_API_KEY or OPENAI_API_KEY in your .env file.")
    print("You can also set them directly in the environment before running this script.")
    print("Example: export OPENAI_API_KEY=your_key_here")
    sys.exit(1)

print(f"API keys status: LambdaLabs: {'Set' if LAMBDALABS_API_KEY else 'Not set'}, OpenAI: {'Set' if OPENAI_API_KEY else 'Not set'}")

# Import functions from run_multiagent_completions.py instead
from run_code.run_multiagent_completions import run_multi_agent_completions

def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file"""
    data = []
    with jsonlines.open(file_path) as reader:
        for item in reader:
            data.append(item)
    return data

def format_rust_problem(problem_description: str, difficulty: str, idx: int) -> Dict[str, Any]:
    """Format a problem description into a format suitable for MultiAgentModel"""
    
    # Check if this is a general Rust coding problem vs a function-specific one
    is_general_question = any(general_keyword in problem_description.lower() for general_keyword in 
                             ["implement a program", "create a rust application", "write a program", 
                              "develop a rust", "build a", "write code", "implement a rust program",
                              "create a simulation", "implement a simulation"])
    
    if is_general_question:
        # For general programming tasks, we don't need to extract function details
        return {
            "task_id": f"{difficulty}_general_{idx}",
            "prompt": problem_description,
            "is_general": True,
            "canonical_solution": "",
            "test": ""
        }
    
    # For function-specific problems, extract function details
    import re
    fn_match = re.search(r'fn\s+([a-zA-Z0-9_]+)', problem_description)
    entry_point = fn_match.group(1) if fn_match else "solution"
    
    # For Rust problems, create a minimal declaration
    declaration = ""
    if fn_match:
        signature_match = re.search(fr'fn\s+{entry_point}[^\{{]*', problem_description)
        if signature_match:
            declaration = signature_match.group(0)
    
    # If we couldn't extract a declaration, create a simple one
    if not declaration:
        declaration = f"fn {entry_point}() -> ()"
    
    # Format sample in the style expected by HumanEvalPack
    return {
        "task_id": f"{difficulty}_{entry_point}_{idx}",
        "entry_point": entry_point,
        "declaration": declaration,
        "prompt": problem_description,
        "canonical_solution": "",
        "test": "",
        "is_general": False
    }

def analyze_results(output_file: str) -> Dict[str, Dict[str, Any]]:
    """Analyze the results from the output file"""
    results = []
    with jsonlines.open(output_file) as reader:
        for item in reader:
            results.append(item)
    
    # Extract difficulty from task_id
    for result in results:
        if "task_id" in result:
            parts = result["task_id"].split("_")
            difficulty = parts[0]
            # Check if this is a general problem
            if len(parts) > 1 and parts[1] == "general":
                result["is_general"] = True
            result["difficulty"] = difficulty
    
    # Group by difficulty
    difficulty_stats = {}
    
    for result in results:
        difficulty = result.get("difficulty", "unknown")
        is_general = result.get("is_general", False)
        
        # Create separate stats for general vs function-specific problems
        stat_key = f"{difficulty}_general" if is_general else difficulty
        
        if stat_key not in difficulty_stats:
            difficulty_stats[stat_key] = {
                "total": 0,
                "compilation_success": 0,
                "test_success": 0,
                "exit_reasons": {},
                "is_general": is_general
            }
        
        stats = difficulty_stats[stat_key]
        stats["total"] += 1
        
        # Check if compilation was successful in any iteration
        # This is stored in the raw_generation field which contains a tuple (code, details)
        compilation_success = False
        
        if "raw_generation" in result:
            # The raw_generation field should contain a list of (code, details) tuples
            generations = result["raw_generation"]
            if isinstance(generations, list) and len(generations) > 0:
                for generation in generations:
                    # Each generation is a tuple (code, details)
                    if isinstance(generation, tuple) and len(generation) > 1:
                        details = generation[1]
                        # Check for successful compilation in iterations_data
                        if "iterations_data" in details:
                            for iteration_data in details["iterations_data"]:
                                review_details = iteration_data.get("review_details", {})
                                if "compilation" in review_details:
                                    compilation = review_details["compilation"]
                                    if compilation.get("return_code", 1) == 0:
                                        compilation_success = True
                                        break
                                # For general mode, also check execution
                                elif is_general and "execution" in review_details:
                                    execution = review_details["execution"]
                                    if execution.get("return_code", 1) == 0:
                                        compilation_success = True
                                        break
                        
                        # If there were no iterations but we have success info directly
                        if not details.get("iterations_data") and details.get("success") is not None:
                            compilation_success = details["success"]
                        
                        # Also track exit reasons
                        exit_reason = details.get("exit_reason", "unknown")
                        if exit_reason not in stats["exit_reasons"]:
                            stats["exit_reasons"][exit_reason] = 0
                        stats["exit_reasons"][exit_reason] += 1
        
        if compilation_success:
            stats["compilation_success"] += 1
            
        # Track test success (if available)
        if result.get("success", False):
            stats["test_success"] += 1
    
    # Calculate rates
    for difficulty, stats in difficulty_stats.items():
        total = stats["total"]
        if total > 0:
            stats["compilation_rate"] = stats["compilation_success"] / total
            stats["test_success_rate"] = stats["test_success"] / total
        else:
            stats["compilation_rate"] = 0
            stats["test_success_rate"] = 0
    
    return difficulty_stats

def main():
    # Load data from files
    data_dir = "data"
    files = {
        "easy": os.path.join(data_dir, "ideas_easy.jsonl"),
        "medium": os.path.join(data_dir, "ideas_medium.jsonl"),
        "hard": os.path.join(data_dir, "ideas_hard.jsonl"),
        "creative": os.path.join(data_dir, "ideas_creative.jsonl")
    }
    
    all_problems = []
    
    # Track problem counts by difficulty
    problem_counts = {}
    
    for difficulty, file_path in files.items():
        print(f"Loading {difficulty} problems from {file_path}")
        problems = load_jsonl_file(file_path)
        problem_counts[difficulty] = len(problems)
        
        # Format each problem to match HumanEvalPack format
        formatted_problems = []
        for idx, problem in enumerate(problems):
            description = problem["description"]
            formatted = format_rust_problem(description, difficulty, idx)
            formatted_problems.append(formatted)
        
        print(f"Loaded {len(formatted_problems)} {difficulty} problems")
        all_problems.extend(formatted_problems)
    
    # Limit the number of problems for testing
    max_problems_per_difficulty = 2
    if max_problems_per_difficulty:
        # Group by difficulty and take a subset
        by_difficulty = {}
        for problem in all_problems:
            difficulty = problem["task_id"].split("_")[0]
            if difficulty not in by_difficulty:
                by_difficulty[difficulty] = []
            if len(by_difficulty[difficulty]) < max_problems_per_difficulty:
                by_difficulty[difficulty].append(problem)
        
        # Flatten
        all_problems = []
        for difficulty, problems in by_difficulty.items():
            all_problems.extend(problems)
        
        print(f"Limited to {len(all_problems)} problems total")
    
    # Save the problems to a temporary file that run_multi_agent_completions can read
    temp_dataset_file = "temp_rust_problems.jsonl"
    with jsonlines.open(temp_dataset_file, "w") as writer:
        writer.write_all(all_problems)
    
    print(f"Saved {len(all_problems)} problems to {temp_dataset_file}")
    
    # Environment variables or default values for models
    gen_model_type = "openai" if OPENAI_API_KEY else "lambdalabs"
    gen_model_name = os.getenv("GEN_MODEL_NAME", "gpt-4" if gen_model_type == "openai" else "llama3.3-70b-instruct-fp8")
    review_model_type = "openai" if OPENAI_API_KEY else "lambdalabs"
    review_model_name = os.getenv("REVIEW_MODEL_NAME", "gpt-4" if review_model_type == "openai" else "llama3.3-70b-instruct-fp8")
    
    print(f"Using generator model: {gen_model_type}/{gen_model_name}")
    print(f"Using reviewer model: {review_model_type}/{review_model_name}")
    
    # Find the main Rust project directory (outside of the models directory)
    main_rust_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rust")
    if not os.path.exists(main_rust_dir):
        print(f"Warning: Main Rust directory not found at {main_rust_dir}")
        print("Creating it...")
        os.makedirs(main_rust_dir, exist_ok=True)
        
        # Create a basic Cargo.toml file
        cargo_toml_path = os.path.join(main_rust_dir, "Cargo.toml")
        with open(cargo_toml_path, "w") as f:
            f.write("""[package]
name = "rust_eval"
version = "0.1.0"
edition = "2021"

[dependencies]
""")
        # Create src directory
        os.makedirs(os.path.join(main_rust_dir, "src"), exist_ok=True)
        # Create a basic lib.rs file
        with open(os.path.join(main_rust_dir, "src", "lib.rs"), "w") as f:
            f.write("""// Empty lib.rs file for the Rust project
""")
    
    print(f"Using Rust directory: {main_rust_dir}")
    
    # Output file for the results
    output_file = "compilation_results_concurrent.jsonl"
    
    # Run the multi-agent completions on the custom dataset
    run_multi_agent_completions(
        task="custom",
        language="rust",
        gen_model_type=gen_model_type,
        gen_model_name=gen_model_name,
        review_model_type=review_model_type,
        review_model_name=review_model_name,
        temperature=0.2,
        top_p=0.95,
        max_iterations=3,
        samples_per_problem=1,
        limit=None,
        verbose=True,
        output_file=output_file,
        skip_review=False,
        timeout=60,
        max_workers=16,
        custom_dataset=all_problems,  # Pass our custom problems directly
        rust_dir=main_rust_dir,  # Pass the main Rust directory
        general_mode=True  # Enable general mode to handle both function-specific and general problems
    )
    
    # Analyze the results
    print("\nAnalyzing results...")
    difficulty_stats = analyze_results(output_file)
    
    # Display results
    print("\nCompilation Success Rates:")
    print("==========================")
    
    # Header row
    print(f"{'Difficulty':<15} | {'Type':<12} | {'Total':<6} | {'Compiled':<10} | {'Compile %':<10} | {'Tests Pass':<10} | {'Test %':<10}")
    print("-" * 90)
    
    # Calculate overall stats
    total_problems = 0
    total_compiled = 0
    total_tests_passed = 0
    
    # Separate general and function-specific problems
    general_stats = {}
    function_stats = {}
    
    for key, stats in difficulty_stats.items():
        if stats.get("is_general", False):
            general_stats[key] = stats
        else:
            function_stats[key] = stats
    
    # Print stats for function-specific problems
    for difficulty, stats in sorted(function_stats.items()):
        total = stats["total"]
        compilation_success = stats["compilation_success"]
        test_success = stats["test_success"]
        compilation_rate = stats["compilation_rate"]
        test_success_rate = stats["test_success_rate"]
        
        print(f"{difficulty:<15} | {'Function':<12} | {total:<6} | {compilation_success:<10} | {compilation_rate:>9.2%} | {test_success:<10} | {test_success_rate:>9.2%}")
        
        total_problems += total
        total_compiled += compilation_success
        total_tests_passed += test_success
    
    # Print stats for general programming problems
    for difficulty, stats in sorted(general_stats.items()):
        base_difficulty = difficulty.replace("_general", "")
        total = stats["total"]
        compilation_success = stats["compilation_success"]
        test_success = stats["test_success"]
        compilation_rate = stats["compilation_rate"]
        test_success_rate = stats["test_success_rate"]
        
        print(f"{base_difficulty:<15} | {'General':<12} | {total:<6} | {compilation_success:<10} | {compilation_rate:>9.2%} | {test_success:<10} | {test_success_rate:>9.2%}")
        
        total_problems += total
        total_compiled += compilation_success
        total_tests_passed += test_success
    
    # Add overall stats
    print("-" * 90)
    overall_compile_rate = total_compiled / total_problems if total_problems > 0 else 0
    overall_test_rate = total_tests_passed / total_problems if total_problems > 0 else 0
    print(f"{'OVERALL':<15} | {'All':<12} | {total_problems:<6} | {total_compiled:<10} | {overall_compile_rate:>9.2%} | {total_tests_passed:<10} | {overall_test_rate:>9.2%}")
    
    # Save summary stats
    summary = {
        "total_problems_available": problem_counts,
        "total_problems_tested": total_problems,
        "total_compiled": total_compiled,
        "total_tests_passed": total_tests_passed,
        "overall_compile_rate": overall_compile_rate,
        "overall_test_rate": overall_test_rate,
        "by_difficulty": difficulty_stats
    }
    
    with open("compilation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to compilation_summary.json")

if __name__ == "__main__":
    main() 