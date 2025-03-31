import os
import argparse
import jsonlines
import re
from tqdm import tqdm
from parsers.content_parser import ContentParser, ParseError

def process_code(code: str, entry_point: str) -> str:
    """
    Process the code by:
    1. Removing any test modules
    2. Moving the entry point function to the front
    3. Keeping all other functions in their original order
    
    Args:
        code: The input code string
        entry_point: The name of the function to process
        
    Returns:
        Processed code string
    """
    # Remove test module if present
    test_module_start = code.find("#[cfg(test)]")
    if test_module_start != -1:
        code = code[:test_module_start]
    
    # Find all function definitions
    function_pattern = re.compile(r'\bfn\s+([a-zA-Z0-9_]+).*?(?=\bfn\s+|\Z)', re.DOTALL)
    functions = list(function_pattern.finditer(code))
    
    if not functions:
        return code
        
    # Find the entry point function and other functions
    entry_point_fn = None
    other_functions = []
    
    for match in functions:
        fn_name = match.group(1)
        if fn_name == entry_point:
            entry_point_fn = match
        elif fn_name != "main":  # Skip main function
            other_functions.append(match)
            
    if not entry_point_fn:
        return code
        
    # Combine functions with entry point first
    parser = ContentParser()

    result = parser("", entry_point_fn.group(0), entry_point)
    if other_functions:
        result += "\n\n" + "\n\n".join(f.group(0) for f in other_functions)
    
    return result

def parse_completions(input_file, output_file, verbose=False):
    """
    Parse raw completions from input_file and save the parsed results to output_file
    """
    parse_errors = 0
    
    # Load samples from input file
    with jsonlines.open(input_file, "r") as reader:
        samples = list(reader)
    
    print(f"Loaded {len(samples)} samples from {input_file}")
    
    # Process each sample
    for idx, sample in enumerate(tqdm(samples)):
        if verbose:
            print(f"Processing sample {idx}")
            
        if "raw_generation" not in sample:
            print(f"Warning: No raw_generation field in sample {idx}")
            continue
            
        if "entry_point" not in sample:
            print(f"Warning: No entry_point field in sample {idx}")
            continue
            
        # Parse the generated solutions
        try:
            # Handle raw_generation which might be a list of lists or just a list
            raw_gen = sample["raw_generation"]
            if isinstance(raw_gen, list) and len(raw_gen) > 0:
                if isinstance(raw_gen[0], list):
                    # If it's a list of lists, take the first list
                    raw_gen = raw_gen[0]
                
                # Process each generation item
                processed_generations = []
                for gen_item in raw_gen:
                    try:
                        # Skip ContentParser and directly use process_code
                        # This will only rearrange functions without extracting implementations
                        processed_code = process_code(gen_item, sample["entry_point"])
                        if verbose:
                            print(f"Processed code:\n{processed_code}...")
                        processed_generations.append(processed_code)
                    except Exception as e:
                        if verbose:
                            print(f"ERROR for {sample.get('task_id', f'sample {idx}')}:")
                            print(f"Raw generation: {gen_item[:200]}...")  # Print first 200 chars
                            print(f"Error: {e}")
                        processed_generations.append("")
                
                sample["generation"] = processed_generations
                
            else:
                if verbose:
                    print(f"Warning: Invalid raw_generation format for sample {idx}")
                sample["generation"] = [""]
                
        except Exception as e:
            parse_errors += 1
            if verbose:
                print(f"TOP LEVEL ERROR for {sample.get('task_id', f'sample {idx}')}:")
                print(f"Error: {e}")
                if "raw_generation" in sample:
                    print(f"Raw generation type: {type(sample['raw_generation'])}")
                    if isinstance(sample["raw_generation"], list):
                        print(f"Raw generation length: {len(sample['raw_generation'])}")
                        if len(sample["raw_generation"]) > 0:
                            print(f"First item type: {type(sample['raw_generation'][0])}")
            sample["generation"] = [""]
    
    # Print error rate
    print(f"Parse error rate: {parse_errors / len(samples):.2%}")
    
    # Save results to file
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(samples)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse completions from a JSONL file")
    parser.add_argument("input_file", help="Input JSONL file containing raw completions")
    parser.add_argument("--output_file", help="Output JSONL file (defaults to input_file with '_parsed' suffix)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed error messages")
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output_file:
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = f"{base_name}_parsed.jsonl"
    
    parse_completions(args.input_file, args.output_file, args.verbose)
