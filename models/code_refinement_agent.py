from models.code_generation_models import CodeGenerationModel
from typing import List
import re
import termcolor

class CodeRefinementAgent(CodeGenerationModel):
    """Agent that refines code based on prompts"""
    
    system_prompt = """
You are a Rust programming expert. Your task is to solve a Rust problem by implementing the function whose signature and imports you are given.
Do NOT change the function signature. Do not add any imports that are not already present in the problem description.
"""
    
    def __init__(self, model: CodeGenerationModel, verbose: bool = True, disable_system_prompts: bool = False):
        self.model = model
        self._verbose = verbose
        self._disable_system_prompts = disable_system_prompts
        
        # If system prompts are disabled, use empty string
        self.system_prompt = "" if disable_system_prompts else self.system_prompt
        
        if verbose and disable_system_prompts:
            print(termcolor.colored("System prompts disabled for CodeRefinementAgent", "yellow"))

    def _log(self, message: str, color: str = None, always: bool = False, attrs=None, separate_section: bool = False):
        """
        Log a message if verbose is true or always is true
        
        Args:
            message: The message to log
            color: Optional color for the message (using termcolor)
            always: Whether to log regardless of verbose setting
            attrs: Additional attributes for termcolor
            separate_section: Whether to add dividers before and after the message for emphasis
        """
        if self._verbose or always:
            if separate_section:
                print("-" * 40)
            if color:
                attrs = attrs or []
                print(termcolor.colored(message, color, attrs=attrs))
            else:
                print(message)
            if separate_section:
                print("-" * 40)

    def generate_code(self, prompt: str, n: int = 1) -> List[str]:
        """
        Refines code based on the prompt
        
        Args:
            prompt: The prompt to generate code from
            n: Number of completions to generate
            
        Returns:
            List containing the generated code completion(s)
        """
        self._log(f"\nGENERATING INITIAL CODE...", "cyan", attrs=["bold"], separate_section=True)
        self._log(f"Using prompt of length {len(prompt)} characters", "cyan")
        
        codes = self.model.generate_code(prompt, system_prompt=self.system_prompt if not self._disable_system_prompts else "", n=n)
        
        for i, code in enumerate(codes):
            if n > 1:
                self._log(f"\nGENERATED CODE VARIANT {i+1}:", "green", attrs=["bold"])
            else:
                self._log(f"\nGENERATED CODE:", "green", attrs=["bold"])
            self._log(code, separate_section=True)
            
        return codes

    def refine_code(self, prompt: str, code: str, feedback: str, n: int = 1) -> List[str]:
        """
        Refine the code based on the prompt and feedback
        """        
        self._log(f"\nREFINING CODE...", "cyan", attrs=["bold"])
        self._log(f"Received feedback:", "yellow")
        self._log(feedback, separate_section=True)
        
        refinement_prompt = f"""
Original Problem:
####
{prompt}
####

You are given an unrefined implementation of the function and feedback on what is wrong with it.
Old Implementation:
```rust
{code}
```

Feedback from Code Review:
{feedback}

Provide a fixed version of the code that addresses all the issues mentioned in the feedback.
Only output the complete, corrected function with no additional explanation.
Do not include tests in your response. Only include the function.
The function name and signature should not change from the original problem.
If the code currently does not compile, you must fix all the compilation errors!!!
"""
        self._log(f"Created refinement prompt of length {len(refinement_prompt)} characters", "cyan")
        
        refined_codes = self.model.generate_code(refinement_prompt, system_prompt=self.system_prompt if not self._disable_system_prompts else "", n=n)
        
        for i, refined_code in enumerate(refined_codes):
            if n > 1:
                self._log(f"\nREFINED CODE VARIANT {i+1}:", "green", attrs=["bold"])
            else:
                self._log(f"\nREFINED CODE:", "green", attrs=["bold"])
            self._log(refined_code, separate_section=True)
            
        return refined_codes
        
    @property
    def model_name(self) -> str:
        return f"Generator({self.model.model_name})"
