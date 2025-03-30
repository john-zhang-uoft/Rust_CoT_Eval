#!/usr/bin/env python3
import json
import argparse
import sys
from pathlib import Path


def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def print_sample_details(sample):
    """Print relevant details from a sample."""
    print("\n" + "="*80)
    print(f"TASK ID: {sample.get('task_id', 'N/A')}")
    print("="*80)
    
    # Print prompt
    if 'prompt' in sample:
        print("\nPROMPT:")
        print("-"*80)
        print(sample['prompt'])
        print("-"*80)
    
    # Print declaration if available
    if 'declaration' in sample and sample['declaration']:
        print("\nDECLARATION:")
        print("-"*80)
        print(sample['declaration'])
        print("-"*80)
    
    # Print docstring if available
    if 'docstring' in sample and sample['docstring']:
        print("\nDOCSTRING:")
        print("-"*80)
        print(sample['docstring'])
        print("-"*80)
    
    # Print bug information
    if 'bug_type' in sample and 'failure_symptoms' in sample:
        print("\nBUG INFORMATION:")
        print(f"Bug type: {sample['bug_type']}")
        print(f"Failure symptoms: {sample['failure_symptoms']}")
        print("-"*80)
    
    # Print canonical solution
    if 'canonical_solution' in sample:
        print("\nCANONICAL SOLUTION:")
        print("-"*80)
        print(sample['canonical_solution'])
        print("-"*80)
    
    # Print buggy solution
    if 'buggy_solution' in sample:
        print("\nBUGGY SOLUTION:")
        print("-"*80)
        print(sample['buggy_solution'])
        print("-"*80)
    
    # Print final generated code
    if 'raw_generation' in sample and isinstance(sample['raw_generation'], list) and sample['raw_generation']:
        if isinstance(sample['raw_generation'][0], list) and sample['raw_generation'][0]:
            print("\nGENERATED SOLUTION:")
            print("-"*80)
            print(sample['raw_generation'][0][0])
            print("-"*80)
    
    # Extract and print final parsed code if available
    if isinstance(sample.get('success'), bool):
        success_status = "SUCCESS" if sample['success'] else "FAILED"
        print(f"\nCOMPLETION STATUS: {success_status}")
        
        if 'iterations_data' in sample and 'final_parsed_code' in sample:
            print("\nFINAL PARSED CODE:")
            print("-"*80)
            print(sample['final_parsed_code'])
            print("-"*80)
            
            # Print iteration information
            if 'iterations' in sample:
                print(f"\nITERATIONS: {sample['iterations']}")
                
                for i, iteration_data in enumerate(sample['iterations_data']):
                    print(f"\nITERATION {i+1}:")
                    if 'failing_tests' in iteration_data.get('analysis', {}):
                        print(f"Failing tests: {', '.join(iteration_data['analysis']['failing_tests'])}")
                    print(f"Success: {iteration_data['success']}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Parse and display code completion sample from JSONL file')
    parser.add_argument('file', help='Path to the JSONL file')
    parser.add_argument('-i', '--index', type=int, help='Sample index to display (0-based)')
    parser.add_argument('-t', '--task-id', help='Task ID to search for')
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found.")
        return 1
    
    try:
        data = load_jsonl(file_path)
        print(f"Loaded {len(data)} samples from {file_path}")
        
        if args.index is not None:
            if 0 <= args.index < len(data):
                print_sample_details(data[args.index])
            else:
                print(f"Error: Index {args.index} is out of range. Valid range: 0-{len(data)-1}")
                return 1
        elif args.task_id:
            found = False
            for sample in data:
                if sample.get('task_id') == args.task_id:
                    print_sample_details(sample)
                    found = True
                    break
            if not found:
                print(f"Error: No sample found with task ID '{args.task_id}'")
                return 1
        else:
            print("Please specify either a sample index (-i/--index) or task ID (-t/--task-id)")
            return 1
        
        return 0
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 