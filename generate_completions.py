"""Testing
from datasets import load_dataset

ds = load_dataset("bigcode/humaneval-x-bugs", "python")["test"]
idx = 0

def get_prompt_base(doc, language="python"):
    # See 
    # https://github.com/roG0d/CodeGeeX/blob/f66205b5f615a4eead9c26d7ec297e14738ea18d/codegeex/benchmark/evaluate_humaneval_x.py#L78
    # https://github.com/THUDM/CodeGeeX/pull/76#issuecomment-1500653190
    if language == "rust":
        main = "fn main(){}\n"
        prompt_base = main + doc["declaration"] + doc["prompt"]
    else:
        prompt_base = doc["prompt"]
    return prompt_base

prompt_base = get_prompt_base(ds[idx], language="python")
    
messages = [
    {
        "role": "user",
        "content": ds[idx]["instruction"],
    },
    {
        "role": "assistant",
        "content": prompt_base,
    },
]

gpt-4-0613
response = openai.ChatCompletion.create(
model="gpt-4-0613",
messages=messages
)
"""

import os
import argparse
import jsonlines
import termcolor
from dotenv import load_dotenv
from datasets import load_dataset
from typing import List, Optional
from tqdm import tqdm

from code_generation_models import CodeGenerationModel, OpenAIChatModel, LambdaLabsModel
from content_parser import ContentParser, ParseError

_CITATION = """
@article{muennighoff2023octopack,
      title={OctoPack: Instruction Tuning Code Large Language Models}, 
      author={Niklas Muennighoff and Qian Liu and Armel Zebaze and Qinkai Zheng and Binyuan Hui and Terry Yue Zhuo and Swayam Singh and Xiangru Tang and Leandro von Werra and Shayne Longpre},
      journal={arXiv preprint arXiv:2308.07124},
      year={2023}
}
"""

LANGUAGE_TO_NAME = {
    "python": "Python",
    "cpp": "C++",
    "js": "JavaScript",
    "java": "Java",
    "go": "Go",
    "rust": "Rust",
}

def get_prompt_base(doc, language):
    # See 
    # https://github.com/roG0d/CodeGeeX/blob/f66205b5f615a4eead9c26d7ec297e14738ea18d/codegeex/benchmark/evaluate_humaneval_x.py#L78
    # https://github.com/THUDM/CodeGeeX/pull/76#issuecomment-1500653190
    if language == "rust":
        main = "fn main(){}\n"
        prompt_base = main + doc["declaration"]
    else:
        prompt_base = doc["declaration"]
    return prompt_base


def get_prompt_synthesize(doc, language="python"):
    # addon = f"Start your code with:\n{get_prompt_base(sample, language)}"
    addon = f"Do not include the ` character in your code. End your code with {get_prompt_base(doc, language)}"
    return doc["instruction"] + "\n" + addon # Results in worse performance for GPT4

    return doc["instruction"] # Problem: Difficult for problems that have helper functions


def get_base_prompt_fix(doc, language="python", mode="tests"):
    if language == "rust":
        if mode == "tests":
            return "fn main(){}\n" + doc["declaration"]
        elif mode == "docs":
            return "fn main(){}\n" + doc["declaration"] + doc["prompt"]
        else:
            raise ValueError
    else:
        if mode == "tests":
            return doc["declaration"]
        elif mode == "docs":
            return doc["prompt"]
        else:
            raise ValueError

def get_prompt_fix(doc, language="python", mode="tests"):
    prompt_base = get_base_prompt_fix(doc, language, mode)
    func = prompt_base + doc["buggy_solution"]
    instruction = f'Fix bugs in {doc["entry_point"]}.'
    return func + "\n" + instruction

def get_prompt_explain_desc(doc, language="python"):
    if language == "rust":
        main = "fn main(){}\n"
        prompt_base = main + doc["declaration"]
    else:
        prompt_base = doc["declaration"]
    docstring_len = len(doc["docstring"])

    instruction = f"Provide a concise natural language description of the code using at most {docstring_len} characters."
    func = prompt_base + doc["canonical_solution"]

    return instruction + "\n" + func, docstring_len

def get_prompt_explain_syn(sample, desc, language="python"):
    instruction = f"Write functional code in {LANGUAGE_TO_NAME[language]} according to the description."
    addon = f"Start your code with:\n{get_prompt_base(sample, language)}"
    return desc + "\n" + instruction + "\n" + addon


def create_model(model_type: str, model_name: str, temperature: float, top_p: float) -> CodeGenerationModel:
    """
    Create a code generation model based on type
    
    Args:
        model_type: Type of model to create ('openai', 'lambdalabs', etc.)
        model_name: Name of the model
        temperature: Temperature parameter
        top_p: Top-p parameter
        
    Returns:
        An instance of a CodeGenerationModel
    """
    if model_type.lower() == 'openai':
        return OpenAIChatModel(model=model_name, temperature=temperature, top_p=top_p)
    elif model_type.lower() == 'lambdalabs':
        return LambdaLabsModel(model=model_name, temperature=temperature, top_p=top_p)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_completions(
    task: str,
    language: str,
    model: CodeGenerationModel,
    samples_per_problem: int = 1,
    limit: Optional[int] = None,
    verbose: bool = False,
    output_file: Optional[str] = None
):
    """
    Generate code completions for the given task and language
    
    Args:
        task: Task type ('humanevalfix', 'humanevalsynthesize', etc.)
        language: Programming language
        model: CodeGenerationModel instance to use
        samples_per_problem: Number of samples to generate per problem
        limit: Optional limit on number of problems to process
        verbose: Whether to print detailed information
        output_file: Custom output file path (if None, a default is used)
    """
    # Load the dataset
    samples = [s for s in load_dataset("bigcode/humanevalpack", language)["test"]]
    print(f"Loaded {len(samples)} samples from HumanEvalPack {language} dataset")
    
    # Handle specific tasks
    descriptions = None
    if task == "humanevalexplainsynthesize":
        with jsonlines.open(f"completions_{language}_humanevalexplaindescribe.jsonl", "r") as f:
            descriptions = [line["raw_generation"][0] for line in f]
    
    # Initialize parser
    parser = ContentParser()
    parse_errors = 0
    
    # Process each sample
    for idx, sample in enumerate(tqdm(samples)):
        if limit is not None and idx >= limit:
            break
            
        # Get the appropriate prompt based on the task
        if task == "humanevalfix":
            prompt = get_prompt_fix(sample, language=language, mode="tests")
        elif task == "humanevalsynthesize":
            prompt = get_prompt_synthesize(sample, language=language)
        elif task == "humanevalexplaindescribe":
            prompt, docstring_len = get_prompt_explain_desc(sample, language=language)
            gen = model.generate_code(prompt, samples_per_problem)
            sample["raw_generation"] = gen
            sample["generation"] = [gen_item[:docstring_len] for gen_item in gen]
            continue  # Skip the normal parsing for this task
        elif task == "humanevalexplainsynthesize":
            desc = descriptions[idx]
            prompt = get_prompt_explain_syn(sample, desc, language=language)
        else:
            raise ValueError(f"Unknown task: {task}")
            
        if verbose:
            print(f"Processing {sample['task_id']} ({idx + 1}/{len(samples)})...")
            
        # Generate solutions
        sample["raw_generation"] = model.generate_code(prompt, samples_per_problem)
        
        # Parse the generated solutions
        try:
            sample["generation"] = [
                parser(prompt, generation_item, sample["entry_point"]) 
                for generation_item in sample["raw_generation"]
            ]
        except ParseError as e:
            parse_errors += 1
            print(f"PARSE EXCEPTION for {sample['task_id']}: {e}")
            sample["generation"] = [""] * samples_per_problem
            
        # Print details if verbose
        if verbose:
            for i in range(samples_per_problem):
                print(termcolor.colored(sample["entry_point"], "yellow", attrs=["bold"]))
                print(termcolor.colored(prompt, "yellow"))
                print(termcolor.colored(sample["canonical_solution"], "red"))
                print(termcolor.colored(sample["generation"][i], "green")+"\n\n")
                
    # Print error rate
    if verbose:
        print(f"Parse error rate: {parse_errors / len(samples):.2%}")
        
    # Save results to file
    if output_file is None:
        output_file = f"completions_{language}_{task}.jsonl"
        
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(samples)
        
    print(f"Results saved to {output_file}")
    

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate code completions for HumanEval tasks")
    parser.add_argument("--task", type=str, default="humanevalsynthesize", 
                        choices=["humanevalfix", "humanevalsynthesize", 
                                 "humanevalexplaindescribe", "humanevalexplainsynthesize"],
                        help="Task to generate completions for")
    parser.add_argument("--language", type=str, default="rust",
                        choices=list(LANGUAGE_TO_NAME.keys()),
                        help="Programming language to generate completions for")
    parser.add_argument("--model_type", type=str, default="lambdalabs",
                        choices=["openai", "lambdalabs"],
                        help="Type of model to use")
    parser.add_argument("--model_name", type=str, default="llama3.2-3b-instruct",
                        help="Name of the model to use")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature parameter for generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p parameter for generation")
    parser.add_argument("--samples", type=int, default=1,
                        help="Number of samples to generate per problem")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of problems to process (defaults to all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Custom output file path")
                        
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    
    # Override from environment variables if not specified
    model_type = os.getenv("MODEL_TYPE", args.model_type)
    model_name = os.getenv("MODEL", args.model_name)
    temperature = float(os.getenv("TEMPERATURE", args.temperature))
    top_p = float(os.getenv("TOP_P", args.top_p))
    samples = int(os.getenv("TIMES", args.samples))
    verbose = os.getenv("VERBOSE", str(args.verbose)).lower() == "true"
    task = os.getenv("TASK", args.task)
    language = os.getenv("LANGUAGE", args.language)
    limit = int(os.getenv("LIMIT", args.limit)) if os.getenv("LIMIT") or args.limit else None
    
    # Create the model
    model = create_model(model_type, model_name, temperature, top_p)
    
    print(f"Evaluating on {task} for {language}")
    print(f"Using model: {model.model_name}")
    print(f"Settings: samples={samples}, temperature={temperature}, top_p={top_p}")
    
    # Generate completions
    generate_completions(
        task=task,
        language=language,
        model=model,
        samples_per_problem=samples,
        limit=limit,
        verbose=verbose,
        output_file=args.output_file
    ) 