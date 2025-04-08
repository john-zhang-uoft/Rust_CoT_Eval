#!/usr/bin/env python3
"""
TesterCheckerAgent for reviewing test quality against problem requirements
"""

import re
from typing import Tuple, Dict, Any, Optional
import termcolor

from models.code_generation_models import CodeGenerationModel

class TesterCheckerAgent:
    """
    Agent responsible for checking the quality of tests against the original problem requirements.
    This agent ensures tests actually validate the behavior specified in the prompt.
    """
    system_prompt = """You are a senior software engineer that evaluates the correctnesses of tests written by a junior engineer."""
    
    def __init__(
        self,
        model: CodeGenerationModel,
        language: str = "rust",
        verbose: bool = False,
        sample_idx: Optional[int] = None,
        thread_id: Optional[int] = None,
        rust_dir: Optional[str] = None
    ):
        """
        Initialize a tester checker agent
        
        Args:
            model: The underlying language model
            language: The programming language
            verbose: Whether to print detailed logs
            sample_idx: Optional sample index for file naming
            thread_id: Optional thread ID for parallel processing
            rust_dir: Optional custom path to the Rust project directory
        """
        self._model = model
        self._language = language
        self._verbose = verbose
        self._sample_idx = sample_idx
        self._thread_id = thread_id
        self._rust_dir = rust_dir
    
    def _log(self, message: str, color: str = None, always: bool = False):
        """
        Log a message if verbose is true or always is true
        
        Args:
            message: The message to log
            color: Optional color for the message (using termcolor)
            always: Whether to log regardless of verbose setting
        """
        if self._verbose or always:
            if color:
                print(termcolor.colored(message, color))
            else:
                print(message)
    
    def check_tests(
        self, 
        prompt: str, 
        declaration: str, 
        code: str, 
        tests: str, 
        entry_point: str,
        extra_instructions: str = None
    ) -> Tuple[str, int, Dict[str, Any]]:
        """
        Check the quality of tests against the original problem requirements
        
        Args:
            prompt: The original problem statement
            declaration: The function declaration
            code: The implementation code
            tests: The tests to check
            entry_point: The function name
            extra_instructions: Optional additional instructions to include in the prompt
            
        Returns:
            Tuple containing:
            - Feedback on test quality
            - Confidence score (0-100)
            - Additional details about test issues
        """
        check_prompt = f"""
Below is the problem:
{prompt}

Unit tests written by the junior engineer:
```{self._language}
{tests}
```
"""
        if extra_instructions:
            check_prompt += f"\n\n{extra_instructions}"
        
        check_prompt += """
Your task is to go through tests one by one check and think step by step about the following:
"If we implemented the function as described in the problem statement, would the test pass?"

Also, do the tests collectively cover all requirements in the problem statement?
"""  
        feedback = self._model.generate_code(check_prompt, self.system_prompt)
        
        if self._verbose:
            self._log(f"Test checker response: {feedback}", "cyan")
        
        return feedback

