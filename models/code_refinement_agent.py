from models.code_generation_models import CodeGenerationModel
from typing import List
import re

class CodeRefinementAgent(CodeGenerationModel):
    """Agent that refines code based on prompts"""
    
    def __init__(self, model: CodeGenerationModel):
        self.model = model

    def generate_code(self, prompt: str, n: int = 1) -> List[str]:
        """
        Refines code based on the prompt
        
        Args:
            prompt: The prompt to generate code from
            n: Number of completions to generate
            
        Returns:
            List containing the generated code completion(s)
        """
        
        # Handle initial generation case
        return self.model.generate_code(prompt, n=n)

    def refine_code(self, prompt: str, code: str, feedback: str, n: int = 1) -> List[str]:
        """
        Refine the code based on the prompt and feedback
        """        
        refinement_prompt = f"""
You are a Rust programming expert. Your task is to solve the following Rust problem by implementing the function.

Original Problem:
####
{prompt}
####
Do not change the function signature of the function. Do not use any imports that are not already present in the problem description.

You are given an old implementation of the function and feedback on what is wrong with it.
Old Implementation:
```rust
{code}
```

Feedback from Code Review:
{feedback}

Please provide a fixed version of the code that addresses all the issues mentioned in the feedback.
Only output the complete, corrected function with no additional explanation.
Do not include tests in your response. Only include the function.
The function name and signature should not change from the original problem.
"""
        return self.model.generate_code(refinement_prompt, n=n)    
        
        
    @property
    def model_name(self) -> str:
        return f"Generator({self.model.model_name})"
