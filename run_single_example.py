import os
import sys
import argparse
from dotenv import load_dotenv
from datasets import load_dataset
import jsonlines

from multi_agent_generation import (
    create_model,
    run_multi_agent_generation,
    MultiAgentModel,
    CodeReviewerAgent
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
    parser.add_argument("--use_legacy", action="store_true",
                        help="Use legacy run_multi_agent_generation instead of MultiAgentModel")
    parser.add_argument("--skip_review", action="store_true",
                        help="Skip code review when cargo/compiler is not available")

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
    
    # Check if Cargo is available for Rust
    cargo_available = True
    if args.language == "rust" and not args.skip_review:
        try:
            # Try to create a reviewer to check if cargo is available
            CodeReviewerAgent(review_model, language=args.language)
        except FileNotFoundError as e:
            cargo_available = False
            print(f"Warning: {str(e)}")
            print("Running in code generation only mode (no compilation or testing)")
            if not args.skip_review:
                print("Use --skip_review to continue without code review")
                sys.exit(1)
    
    try:
        if args.use_legacy:
            # Run legacy multi-agent generation
            results = run_multi_agent_generation(
                language=args.language,
                gen_model=gen_model,
                review_model=review_model,
                samples=[sample],
                max_iterations=args.max_iterations,
                verbose=True,  # Always verbose for single sample
                limit=None,
                skip_review=args.skip_review
            )
        else:
            # Use the new MultiAgentModel directly
            multi_agent = MultiAgentModel(
                gen_model=gen_model,
                review_model=review_model,
                language=args.language,
                max_iterations=args.max_iterations,
                verbose=True  # Always verbose for single sample
            )
            
            print(f"Using multi-agent model: {multi_agent.model_name}")
            
            # Get the prompt from sample
            from multi_agent_generation import get_prompt_from_sample
            prompt = get_prompt_from_sample(sample, args.language)
            
            # Get declaration and entry_point directly from the sample
            declaration = sample["declaration"]
            entry_point = sample["entry_point"]
            
            # Generate code using multi-agent model with explicit declaration and entry_point
            code = multi_agent.generate_code(
                prompt, 
                declaration=declaration, 
                entry_point=entry_point
            )[0]
            
            # Determine success status
            success = None
            if cargo_available:
                reviewer = CodeReviewerAgent(review_model, language=args.language)
                success, feedback, _ = reviewer.review(prompt, sample["declaration"], code, sample["entry_point"])
            
            # Create a result structure similar to run_multi_agent_generation
            results = [{
                "task_id": sample["task_id"],
                "entry_point": sample["entry_point"],
                "declaration": sample["declaration"],
                "prompt": prompt,
                "final_code": code,
                "success": success,
                "canonical_solution": sample.get("canonical_solution", ""),
                "iterations": [{"iteration": 0, "success": success}]  # Minimal iteration info
            }]
            
        # Save results
        output_file = args.output_file or f"multi_agent_single_sample_{args.sample_idx}.jsonl"
        with jsonlines.open(output_file, "w") as writer:
            writer.write_all(results)
            
        print(f"Results saved to {output_file}")
        
        # Print final status
        result = results[0]
        if result.get("success"):
            print(f"SUCCESS! Code passed all reviews.")
        elif result.get("success") is None:
            print("CODE GENERATED! Success status unknown (no code review available).")
        else:
            print(f"FAILED! Code did not pass all reviews.")
        
        # Print the final code
        print("\nFinal code:")
        print("=" * 80)
        print(result["final_code"])
        print("=" * 80)
        
        # Print iterations summary if available
        if "iterations" in result and result["iterations"] and args.use_legacy:
            print("\nIterations summary:")
            for i, iteration in enumerate(result["iterations"]):
                status = "SUCCESS" if iteration.get("success") else "FAILED"
                print(f"Iteration {i}: {status}")
                if iteration.get("feedback"):
                    # Print just the first few lines of feedback
                    feedback_preview = "\n".join(iteration["feedback"].split("\n")[:3])
                    print(f"  Feedback: {feedback_preview}...")
        
        print("\nDone!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 