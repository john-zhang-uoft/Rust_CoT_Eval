import os
import argparse
import jsonlines
from tqdm import tqdm
from parsers.content_parser import ContentParser, ParseError

def parse_completions(input_file, output_file, verbose=False):
    """
    Parse raw completions from input_file and save the parsed results to output_file
    """
    # Initialize parser
    parser = ContentParser()
    parse_errors = 0
    
    # Load samples from input file
    with jsonlines.open(input_file, "r") as reader:
        samples = list(reader)
    
    print(f"Loaded {len(samples)} samples from {input_file}")
    
    # Process each sample
    for idx, sample in enumerate(tqdm(samples)):
        if "raw_generation" not in sample:
            print(f"Warning: No raw_generation field in sample {idx}")
            continue
            
        if "prompt" not in sample:
            # Try to reconstruct prompt if it's missing
            if "task_id" in sample:
                print(f"Warning: No prompt field in sample {idx} (task_id: {sample['task_id']})")
            else:
                print(f"Warning: No prompt field in sample {idx}")
            continue
            
        # Parse the generated solutions
        try:
            sample["generation"] = [
                parser(sample["prompt"], generation_item, sample["entry_point"]) 
                for generation_item in sample["raw_generation"]
            ]
        except ParseError as e:
            parse_errors += 1
            if verbose:
                print(f"PARSE EXCEPTION for {sample.get('task_id', f'sample {idx}')}: {e}")
            sample["generation"] = [""] * len(sample["raw_generation"])
        except Exception as e:
            parse_errors += 1
            if verbose:
                print(f"UNEXPECTED EXCEPTION for {sample.get('task_id', f'sample {idx}')}: {e}")
            sample["generation"] = [""] * len(sample["raw_generation"])
    
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
