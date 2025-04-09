import os
import argparse
import jsonlines
import re
from tqdm import tqdm
from parsers.content_parser import ContentParser, ParseError

def find_function_bounds(code: str, entry_point: str) -> tuple[int, int]:
    """
    Find the start and end positions of a function by counting braces.
    First finds the line containing 'fn entry_point', then counts braces until balanced.
    
    Args:
        code: The source code to search
        entry_point: The name of the function to find
        
    Returns:
        Tuple of (start_pos, end_pos) or (-1, -1) if not found
    """
    # Find the line containing the function declaration
    lines = code.split('\n')
    fn_line = -1
    
    # Look for the function declaration, including possible attributes
    for i, line in enumerate(lines):
        if f"fn {entry_point}" in line:
            # Look backwards for attributes
            fn_line = i
            while fn_line > 0 and lines[fn_line - 1].strip().startswith('#['):
                fn_line -= 1
            break
    
    if fn_line == -1:
        return (-1, -1)
        
    # Get the position in the original string
    start_pos = sum(len(l) + 1 for l in lines[:fn_line])
    pos = start_pos
    
    # Find position after the first line (which contains the opening brace)
    while pos < len(code) and code[pos] != '\n':
        pos += 1
    pos += 1
    
    # Now count braces until we find the matching end
    brace_count = 1  # We already found the opening brace
    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escaped = False
    
    while pos < len(code):
        char = code[pos]
        next_char = code[pos + 1] if pos + 1 < len(code) else ''
        
        # Handle escaping
        if char == '\\' and not escaped:
            escaped = True
            pos += 1
            continue
            
        # Handle string literals
        if not escaped and char in ('"', "'") and not in_line_comment and not in_block_comment:
            if not in_string and not in_char:
                # Starting a string/char
                in_string = char == '"'
                in_char = char == "'"
            elif (in_string and char == '"') or (in_char and char == "'"):
                in_string = False
                in_char = False
                    
        # Handle comments
        elif not escaped and char == '/' and next_char == '/' and not in_string and not in_char and not in_block_comment:
            in_line_comment = True
            pos += 1
        elif not escaped and char == '/' and next_char == '*' and not in_string and not in_char and not in_line_comment:
            in_block_comment = True
            pos += 1
        elif not escaped and char == '*' and next_char == '/' and in_block_comment:
            in_block_comment = False
            pos += 1
        elif char == '\n' and in_line_comment:
            in_line_comment = False
            
        # Count braces if we're not in a string/comment
        elif not in_string and not in_char and not in_line_comment and not in_block_comment:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return (start_pos, pos + 1)
                    
        escaped = False
        pos += 1
        
    return (-1, -1)  # No matching end brace found

def process_code(code: str, entry_point: str) -> str:
    """
    Process the code by:
    1. Removing any test modules
    2. Moving the specified function to the front
    3. Keeping all other code (structs, enums, other functions) intact
    4. Removing the signature of the target function
    
    Args:
        code: The input code string
        entry_point: The name of the function to process
        
    Returns:
        Processed code string
    """
    # Remove test module if present
    test_module_start = code.find("#[cfg(test)]")
    if test_module_start != -1:
        code = code[:test_module_start].strip()
        
    # Find the target function
    start_pos, end_pos = find_function_bounds(code, entry_point)
    if start_pos == -1:  # Function not found
        return code
        
    # Extract the function and remove its signature
    target_fn = code[start_pos:end_pos]
    # Find the opening brace
    brace_pos = target_fn.find('{')
    if brace_pos != -1:
        # Keep only what's after the opening brace
        target_fn = target_fn[brace_pos+1:].strip()
        # Find the last closing brace that matches the function's opening brace
        brace_count = 1
        pos = 0
        while pos < len(target_fn):
            if target_fn[pos] == '{':
                brace_count += 1
            elif target_fn[pos] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    target_fn = target_fn[:pos].strip()
                    break
            pos += 1
    
    # Split the code into sections
    lines = code.split('\n')
    
    # Find where the actual code starts (after imports/types/etc)
    code_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not (
            stripped.startswith('use ') or
            stripped.startswith('type ') or
            stripped.startswith('const ') or
            stripped.startswith('static ') or
            stripped.startswith('extern ') or
            stripped.startswith('//') or
            stripped.startswith('/*') or
            stripped.startswith('#[') or  # Skip attributes
            not stripped  # Skip empty lines
        ):
            code_start = i
            break
            
    # Split into prefix and code body
    prefix = '\n'.join(lines[:code_start]).strip()
    code_body = '\n'.join(lines[code_start:]).strip()
    
    # Remove the target function from the code body
    # Find the line containing the function
    body_lines = code_body.split('\n')
    fn_start = -1
    for i, line in enumerate(body_lines):
        if f"fn {entry_point}" in line:
            fn_start = i
            break
            
    if fn_start != -1:
        # Remove the function and any preceding attributes
        while fn_start > 0 and body_lines[fn_start - 1].strip().startswith('#['):
            fn_start -= 1
            
        # Find the end of the function
        fn_end = fn_start
        brace_count = 0
        while fn_end < len(body_lines):
            line = body_lines[fn_end]
            brace_count += line.count('{')
            brace_count -= line.count('}')
            if brace_count == 0 and fn_end > fn_start:
                break
            fn_end += 1
            
        # Remove the function
        body_lines = body_lines[:fn_start] + body_lines[fn_end+1:]
        code_body = '\n'.join(body_lines).strip()
    
    # Combine with target function at the start of the code section
    result_parts = []
    if prefix:
        result_parts.append(prefix + '\n')
    result_parts.append(target_fn + '\n}\n')
    if code_body:
        result_parts.append(code_body)
        
    return "\n\n".join(part for part in result_parts if part)

def parse_completions(input_file, output_file, verbose=False, generation_field_name="raw_generation"):
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
            
        if generation_field_name not in sample:
            print(f"Warning: No {generation_field_name} field in sample {idx}")
            continue
            
        if "entry_point" not in sample:
            print(f"Warning: No entry_point field in sample {idx}")
            continue
            
        # Parse the generated solutions
        try:
            # Handle raw_generation which might be a list of lists or just a list
            raw_gen = sample[generation_field_name]
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
            
            elif isinstance(raw_gen, str):
                processed_generations = []
                try:
                    # Skip ContentParser and directly use process_code
                    # This will only rearrange functions without extracting implementations
                    processed_code = process_code(raw_gen, sample["entry_point"])
                    if verbose:
                        print(f"Processed code:\n{processed_code}...")
                    processed_generations.append(processed_code)
                except Exception as e:
                    if verbose:
                        print(f"ERROR for {sample.get('task_id', f'sample {idx}')}:")
                        print(f"Raw generation: {raw_gen[:200]}...")  # Print first 200 chars
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
                if generation_field_name in sample:
                    print(f"Raw generation type: {type(sample['raw_generation'])}")
                    if isinstance(sample[generation_field_name], list):
                        print(f"Raw generation length: {len(sample['raw_generation'])}")
                        if len(sample[generation_field_name]) > 0:
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
    parser.add_argument("--generation_field_name", default="raw_generation", help="Field name for the generation in the input file")
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output_file:
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = f"{base_name}_parsed.jsonl"
    
    parse_completions(args.input_file, args.output_file, args.verbose, args.generation_field_name)
