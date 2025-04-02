#!/usr/bin/env python3
"""
Confidence-based multi-agent code generation system with planning, coding, and testing agents
"""

import os
import tempfile
import subprocess
import re
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm
import jsonlines
import time
import termcolor
import traceback

from models.code_generation_models import CodeGenerationModel, OpenAIChatModel, LambdaLabsModel
from models.rust_code_reviewer_agent import RustCodeReviewerAgent
from models.code_refinement_agent import CodeRefinementAgent

# Default timeout for subprocess calls (in seconds)
DEFAULT_TIMEOUT = 60  # 1 minute timeout for compilation and test execution


def create_model(model_type: str, model_name: str, temperature: float, top_p: float) -> CodeGenerationModel:
    """Create a code generation model"""
    if model_type.lower() == 'openai':
        return OpenAIChatModel(model=model_name, temperature=temperature, top_p=top_p)
    elif model_type.lower() == 'lambdalabs':
        return LambdaLabsModel(model=model_name, temperature=temperature, top_p=top_p)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_prompt_from_sample(sample: Dict[str, Any], language: str = "rust") -> str:
    """Get the prompt from a sample"""
    if language == "rust":
        # Add the main function for Rust
        main = "fn main(){}\n" if "fn main()" not in sample["declaration"] else ""
        declaration = sample["declaration"]
        
        # Return the instruction with the declaration
        return f"{sample['instruction']}\n\nYou should implement the function according to this declaration:\n{main}{declaration}"
    else:
        return sample['instruction']



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence-based multi-agent code generation")
    parser.add_argument("--language", type=str, default="rust", choices=["rust"],
                        help="Programming language (only Rust supported currently)")
    parser.add_argument("--planner_model_type", type=str, default="lambdalabs",
                        choices=["openai", "lambdalabs"],
                        help="Model type for planning")
    parser.add_argument("--planner_model_name", type=str, default="llama3.3-70b-instruct-fp8",
                        help="Model name for planning")
    parser.add_argument("--coder_model_type", type=str, default="lambdalabs",
                        choices=["openai", "lambdalabs"],
                        help="Model type for coding")
    parser.add_argument("--coder_model_name", type=str, default="llama3.3-70b-instruct-fp8",
                        help="Model name for coding")
    parser.add_argument("--tester_model_type", type=str, default="lambdalabs",
                        choices=["openai", "lambdalabs"],
                        help="Model type for testing")
    parser.add_argument("--tester_model_name", type=str, default="llama3.3-70b-instruct-fp8",
                        help="Model name for testing")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature parameter for generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p parameter for generation")
    parser.add_argument("--max_iterations", type=int, default=3,
                        help="Maximum number of refinement iterations")
    parser.add_argument("--max_planning_attempts", type=int, default=2,
                        help="Maximum number of planning attempts")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file path (default: confidence_multi_agent_results_{language}.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of samples to process")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed logs")
                        
    args = parser.parse_args()
    
    load_dotenv()
    
    # Override with environment variables if available
    planner_model_type = os.getenv("PLANNER_MODEL_TYPE", args.planner_model_type)
    planner_model_name = os.getenv("PLANNER_MODEL_NAME", args.planner_model_name)
    coder_model_type = os.getenv("CODER_MODEL_TYPE", args.coder_model_type)
    coder_model_name = os.getenv("CODER_MODEL_NAME", args.coder_model_name)
    tester_model_type = os.getenv("TESTER_MODEL_TYPE", args.tester_model_type)
    tester_model_name = os.getenv("TESTER_MODEL_NAME", args.tester_model_name)
    
    samples = [s for s in load_dataset("bigcode/humanevalpack", args.language)["test"]]
    print(f"Loaded {len(samples)} samples from HumanEvalPack {args.language} dataset")
    
    multi_agent = create_confidence_multi_agent_model(
        planner_model_type=planner_model_type,
        planner_model_name=planner_model_name,
        coder_model_type=coder_model_type,
        coder_model_name=coder_model_name,
        tester_model_type=tester_model_type,
        tester_model_name=tester_model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        language=args.language,
        max_iterations=args.max_iterations,
        max_planning_attempts=args.max_planning_attempts,
        verbose=args.verbose
    )
    
    print(f"Using confidence multi-agent model: {multi_agent.model_name}")
    
    results = []
    
    for idx, sample in enumerate(tqdm(samples)):
        if args.limit is not None and idx >= args.limit:
            break
            
        if args.verbose:
            print(f"\n{'-'*80}")
            print(f"Processing sample {idx+1}/{len(samples)}: {sample['task_id']}")
            print(f"{'-'*80}")
            
        prompt = get_prompt_from_sample(sample, args.language)
        declaration = sample["declaration"]
        entry_point = sample["entry_point"]
        
        final_code, generation_details = multi_agent.generate_code(
            prompt, 
            declaration=declaration, 
            entry_point=entry_point
        )
        final_code = final_code[0]
        
        result = {
            "task_id": sample["task_id"],
            "entry_point": entry_point,
            "declaration": declaration,
            "prompt": prompt,
            "final_code": final_code,
            "success": generation_details.get("success", False),
            "exit_reason": generation_details.get("exit_reason", "unknown"),
            "iterations": generation_details.get("iterations_data", []),
            "final_confidence": generation_details.get("final_confidence", {}),
            "canonical_solution": sample.get("canonical_solution", "")
        }
        
        results.append(result)
    
    output_file = args.output_file or f"confidence_multi_agent_results_{args.language}.jsonl"
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(results)
        
    print(f"Results saved to {output_file}")
    
    success_count = sum(1 for r in results if r["success"])
    success_rate = success_count / len(results) if results else 0
    print(f"Success rate: {success_rate:.2%} ({success_count}/{len(results)})")
    
    exit_reasons = {}
    for r in results:
        exit_reason = r.get("exit_reason", "unknown")
        exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
    
    print("\nExit reason statistics:")
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count} ({count/len(results):.2%})")
