import os
import argparse
from dotenv import load_dotenv
from datasets import load_dataset
import jsonlines

from multi_agent_generation import (
    create_model,
    run_multi_agent_generation
)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run multi-agent generation on a single example")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of the sample to process (default: 0)")
    parser.add_argument("--language", type=str, default="rust", choices=["rust"],
                        help="Programming language (only Rust supported currently)")
    parser.add_argument("--gen_model_type", type=str, default="lambdalabs",
                        choices=["openai", "lambdalabs"],
                        help="Model type for code generation")
    parser.add_argument("--gen_model_name", type=str, default="llama3.3-70b-instruct-fp8",
                        help="Model name for code generation")
    parser.add_argument("--review_model_type", type=str, default="lambdalabs",
                        choices=["openai", "lambdalabs"],
                        help="Model type for code review")
    parser.add_argument("--review_model_name", type=str, default="llama3.3-70b-instruct-fp8",
                        help="Model name for code review")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature parameter for generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p parameter for generation")
    parser.add_argument("--max_iterations", type=int, default=3,
                        help="Maximum number of refinement iterations")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file path (default: multi_agent_single_sample_{sample_idx}.jsonl)")

    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Override with environment variables if not set
    gen_model_type = os.getenv("GEN_MODEL_TYPE", args.gen_model_type)
    gen_model_name = os.getenv("GEN_MODEL_NAME", args.gen_model_name)
    review_model_type = os.getenv("REVIEW_MODEL_TYPE", args.review_model_type)
    review_model_name = os.getenv("REVIEW_MODEL_NAME", args.review_model_name)
    
    # Load dataset and select a single sample
    samples = [s for s in load_dataset("bigcode/humanevalpack", args.language)["test"]]
    if args.sample_idx >= len(samples):
        print(f"Error: Sample index {args.sample_idx} out of range (0-{len(samples)-1})")
        exit(1)
    
    sample = samples[args.sample_idx]
    print(f"Selected sample {args.sample_idx}: {sample['task_id']}")
    print(f"Instruction: {sample['instruction']}...")
    
    # Create the models
    gen_model = create_model(
        gen_model_type,
        gen_model_name,
        args.temperature,
        args.top_p
    )
    
    review_model = create_model(
        review_model_type,
        review_model_name,
        args.temperature,
        args.top_p
    )
    
    print(f"Using generator model: {gen_model.model_name}")
    print(f"Using reviewer model: {review_model.model_name}")
    
    # Run multi-agent generation on a single sample
    results = run_multi_agent_generation(
        language=args.language,
        gen_model=gen_model,
        review_model=review_model,
        samples=[sample],
        max_iterations=args.max_iterations,
        verbose=True,  # Always verbose for single sample
        limit=None
    )
    
    # Save results
    output_file = args.output_file or f"multi_agent_single_sample_{args.sample_idx}.jsonl"
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(results)
        
    print(f"Results saved to {output_file}")
    
    # Print final status
    result = results[0]
    if result["success"]:
        print(f"SUCCESS! Code passed all reviews after {len(result['iterations'])-1} refinements.")
    else:
        print(f"FAILED! Code did not pass all reviews after {len(result['iterations'])-1} refinements.")
    
    # Print the final code
    print("\nFinal code:")
    print("=" * 80)
    print(result["final_code"])
    print("=" * 80)
    
    # Print iterations summary
    print("\nIterations summary:")
    for i, iteration in enumerate(result["iterations"]):
        status = "SUCCESS" if iteration["success"] else "FAILED"
        print(f"Iteration {i}: {status}")
        if iteration["feedback"]:
            # Print just the first few lines of feedback
            feedback_preview = "\n".join(iteration["feedback"].split("\n")[:3])
            print(f"  Feedback: {feedback_preview}...")
    
    print("\nDone!") 