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
import openai
from openai import OpenAI
from dotenv import load_dotenv
import jsonlines
import termcolor

from cdifflib import CSequenceMatcher
from camel_converter import to_snake
from datasets import load_dataset
from typing import List
from tqdm import tqdm


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
    addon = "Do not include the ` character in your code."
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

class ParseError(Exception):
    pass

class ContentParser:

    @staticmethod
    def _entry_point_variations(entry_point: str) -> List[str]:
        # NOTE: workaround dataset's bug with entry point naming
        return [
            entry_point,
            to_snake(entry_point),
            entry_point[0].lower() + entry_point[1:],
        ]

    def __call__(self, prompt: str, content: str, entry_point: str):
        # NOTE: Model doesn't follow instructions directly:
        # adds description of change and sometimes fixes
        # typos, or other "bugs" in description.
        if "```" in content:
            content = content.split("```")[1]
        # first parse with assumption that content has description
        matcher = CSequenceMatcher(None, prompt, content)
        tag, _, _, j1, j2 = matcher.get_opcodes()[-1]
        if tag == "insert":
            return content[j1:j2]
        # second parse content with assumption that model wrote code without description
        for entry_point in self._entry_point_variations(entry_point):
            if entry_point in content:
                content = content.split(entry_point)[-1]
                return "".join(content.splitlines(keepends=True)[1:])
        raise ParseError(f"Prompt is not in content:\n{content}")
    @staticmethod
    def _entry_point_variations(entry_point: str) -> List[str]:
        # NOTE: workaround dataset's bug with entry point naming
        return [
            entry_point,
            to_snake(entry_point),
            entry_point[0].lower() + entry_point[1:],
        ]

    def __call__(self, prompt: str, content: str, entry_point: str):
        # NOTE: Model doesn't follow instructions directly:
        # adds description of change and sometimes fixes
        # typos, or other "bugs" in description.

        # Remove Rust-style comments which can include bad characters or code
        if "///" in content or "//" in content:
            # Process line by line to handle both doc comments (///) and regular comments (//)
            lines = content.splitlines()
            cleaned_lines = []
            for line in lines:
                # Remove everything after // or /// on each line
                if "//" in line:
                    line = line.split("//")[0]
                cleaned_lines.append(line)
            content = "\n".join(cleaned_lines)

        if "```" in content:
            content = content.split("```")[1]

        if "fn main()" in content:
            content = content.split("fn main()")[0]
        # first parse with assumption that content has description
        matcher = CSequenceMatcher(None, prompt, content)
        tag, _, _, j1, j2 = matcher.get_opcodes()[-1]
        if tag == "insert":
            return content[j1:j2]
        # second parse content with assumption that model wrote code without description
        for entry_point in self._entry_point_variations(entry_point):
            if entry_point in content:
                content = content.split(entry_point)[-1]
                return "".join(content.splitlines(keepends=True)[1:])
        raise ParseError(f"Prompt is not in content:\n{content}")

class ChatWrapper:

    def __init__(self, model: str, temperature=0.2, top_p=0.95, client=None):
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._client = client

    def __call__(self, prompt: str, n: int) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        while True:
            try:
                response = self._client.chat.completions.create(model=self._model,
                messages=messages,
                temperature=self._temperature,
                top_p=self._top_p,
                n=n)
                content_list = list()
                for i in range(n):
                    message = response.choices[i].message
                    assert message.role == "assistant"
                    content_list.append(message.content)
                return content_list
            except Exception as e:
                print("API EXCEPTION:", e)
                # Wait a moment before retrying
                import time
                time.sleep(2)
                continue


if __name__ == '__main__':

    # Load environment variables from .env file
    load_dotenv()

    # Configurable parameters
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
    TOP_P = float(os.getenv("TOP_P", "0.95"))
    TIMES = int(os.getenv("TIMES", "1"))  # Number of samples per problem
    VERBOSE = os.getenv("VERBOSE", "True").lower() == "true"
    LANGUAGE = os.getenv("LANGUAGE", "rust")
    TASK = os.getenv("TASK", "humanevalsynthesize")
    LIMIT = os.getenv("LIMIT", None)

    # Access API keys from environment variables
    LAMBDALABS_API_KEY = os.getenv("LAMBDALABS_API_KEY")
    CUSTOM_API_BASE = os.getenv("CUSTOM_API_BASE", "https://api.lambdalabs.com/v1")
    MODEL = os.getenv("MODEL", "llama3.2-3b-instruct")

    client = OpenAI(api_key=LAMBDALABS_API_KEY, base_url=CUSTOM_API_BASE)

    print(f"Evaluating on {TASK} for {LANGUAGE}")
    print(f"Settings: TIMES={TIMES}, TEMPERATURE={TEMPERATURE}, TOP_P={TOP_P}")

    # Load descriptions
    if TASK == "humanevalexplainsynthesize":
        with jsonlines.open(f"completions_{LANGUAGE}_humanevalexplaindescribe.jsonl", "r") as f:
            descriptions = [line["raw_generation"][0] for line in f]

    # Load dataset
    samples = [s for s in load_dataset("bigcode/humanevalpack", LANGUAGE)["test"]]
    print(f"Loaded {len(samples)} samples from HumanEvalPack {LANGUAGE} dataset")

    # Initialize the chat wrapper and parser
    chat_wrapper = ChatWrapper(MODEL, temperature=TEMPERATURE, top_p=TOP_P, client=client)
    parse_errors = 0
    parser = ContentParser()

    # Process each sample
    for idx, sample in enumerate(tqdm(samples)):
        if LIMIT and idx >= LIMIT:
            break

        if TASK == "humanevalfix":
            prompt = get_prompt_fix(sample, language=LANGUAGE, mode="tests")
        elif TASK == "humanevalsynthesize":
            prompt = get_prompt_synthesize(sample, language=LANGUAGE)
        elif TASK == "humanevalexplaindescribe":
            prompt, docstring_len = get_prompt_explain_desc(sample, language=LANGUAGE)
            gen = chat_wrapper(prompt, TIMES)
            sample["raw_generation"] = gen
            sample["generation"] = [gen_item[:docstring_len] for gen_item in gen]
            continue
        elif TASK == "humanevalexplainsynthesize":
            desc = descriptions[idx]
            prompt = get_prompt_explain_syn(sample, desc, language=LANGUAGE)

        if VERBOSE:
            print(f"Processing {sample['task_id']} ({idx + 1}/{len(samples)}))...")

        # Generate solutions
        sample["raw_generation"] = chat_wrapper(prompt, TIMES)

        # Parse the generated solutions
        try:
            sample["generation"] = [parser(prompt, generation_item, sample["entry_point"]) for generation_item in sample["raw_generation"]]
        except ParseError as e:
            parse_errors += 1
            print(f"PARSE EXCEPTION for {sample['task_id']}: {e}")
            sample["generation"] = [""]

        # Print details if verbose
        if VERBOSE:
            for i in range(TIMES):
                print(termcolor.colored(sample["entry_point"], "yellow", attrs=["bold"]))
                print(termcolor.colored(prompt, "yellow"))
                print(termcolor.colored(sample["canonical_solution"], "red"))
                print(termcolor.colored(sample["generation"][i], "green")+"\n\n")

    # Print error rate
    if VERBOSE:
        print(f"Parse error rate: {parse_errors / len(samples):.2%}")

    # Save results to file
    results_filename = f"completions_{LANGUAGE}_{TASK}.jsonl"
    with jsonlines.open(results_filename, "w") as writer:
        writer.write_all(samples)

    print(f"Results saved to {results_filename}") 