import re
import json
from typing import Dict, Any

import termcolor
from models.code_generation_models import CodeGenerationModel

class PlannerAgent:
    """Agent responsible for planning, writing pseudocode, evaluating difficulty and setting termination conditions"""
    pseudocode_plans = []
    system_prompt = """You are an expert programmer that writes pseudocode for programming problems.
    Come up with a detailed pseudocode plan for solving problems, do not write any code.
    """

    def __init__(self, model: CodeGenerationModel, verbose: bool = False):
        """
        Initialize a planner agent
        
        Args:
            model: The underlying language model
            verbose: Whether to print detailed logs
        """
        self._model = model
        self._verbose = verbose

    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON from response
        """
        json_match = re.search(r'```json\s*({.*?})\s*```', response, re.DOTALL)
        if json_match:
            plan_data = json.loads(json_match.group(1))
        else:
            json_match = re.search(r'{[\s\S]*"pseudocode"[\s\S]*"difficulty"[\s\S]*"termination_conditions"[\s\S]*"confidence"[\s\S]*}', response)
            if json_match:
                plan_data = json.loads(json_match.group(0))
            else:
                raise ValueError("Could not extract JSON from response")
        
        # Validate required fields
        for field in ["pseudocode", "difficulty", "confidence"]:
            if field not in plan_data:
                plan_data[field] = "Not specified"
            
        return plan_data

    def create_plan(self, prompt: str, declaration: str, entry_point: str) -> Dict[str, Any]:
        """
        Create a plan for solving the problem
        
        Args:
            prompt: The problem statement
            declaration: The function declaration
            entry_point: The function name
            
        Returns:
            Dictionary containing pseudocode, difficulty assessment, and termination conditions
        """
        planning_prompt = f"""
Your task is to:
1. Write clear pseudocode for solving this problem
2. Evaluate the difficulty of the problem on a scale of 1-5 (1=very easy, 5=very difficult)
3. Evaluate the confidence of the solution on a scale of 0-100 (0=no confidence, 100=certain)

Problem:
{prompt}

Think step by step about the completion of the function, then provide your response in JSON format:
{{
  "pseudocode": "detailed pseudocode here",
  "difficulty": number between 1-5,
  "confidence": number between 0-100
}}
"""
        response = self._model.generate_code(planning_prompt, self.system_prompt)
        
        if self._verbose:
            print(termcolor.colored(f"Planner response: {response}", "cyan"))
        
        try:
            plan_data = self.extract_json_from_response(response)
            self.pseudocode_plans.append(plan_data)
            return plan_data
        
        except Exception as e:
            if self._verbose:
                print(termcolor.colored(f"Error parsing planner response: {str(e)}", "red"))
                print(f"Response was: {response}")
            return {
                "pseudocode": "Could not parse pseudocode",
                "difficulty": 3,
                "confidence": 50,
                "error": str(e)
            }
    
    def refine_plan(self, prompt: str, declaration: str, entry_point: str, feedback: str) -> Dict[str, Any]:
        """
        Refine a plan for solving the problem
        
        Args:
            prompt: The problem statement
        """

        planning_prompt = f"""
A team tried to solve the following problem:
{prompt}

A previous failed plan for solving this problem was:
{self.pseudocode_plans[-1]}

It led to the following feedback:
{feedback}

Your task is to use the feedback to come up with a betterplan for solving this problem.
1. Write clear pseudocode for solving this problem
2. Evaluate the difficulty of the problem on a scale of 1-5 (1=very easy, 5=very difficult)
3. Evaluate the confidence of the solution on a scale of 0-100 (0=no confidence, 100=certain)

Think step by step about the completion of the function, then provide your response in JSON format:
{{
  "pseudocode": "detailed pseudocode here",
  "difficulty": number between 1-5,
  "confidence": number between 0-100
}}
"""
        response = self._model.generate_code(planning_prompt, self.system_prompt)
        
        if self._verbose:
            print(termcolor.colored(f"Planner response: {response}", "cyan"))
        
        try:
            plan_data = self.extract_json_from_response(response)
            self.pseudocode_plans.append(plan_data)
            return plan_data
        
        except Exception as e:
            if self._verbose:
                print(termcolor.colored(f"Error parsing planner response: {str(e)}", "red"))
                print(f"Response was: {response}")
            return {
                "pseudocode": "Could not parse pseudocode",
                "difficulty": 3,
                "confidence": 50,
                "error": str(e)
            }

    def evaluate_confidence(self, agents_confidence: Dict[str, int]) -> bool:
        """
        Evaluate if all agents are confident enough to proceed
        
        Args:
            agents_confidence: Dictionary mapping agent names to confidence scores (0-100)
            
        Returns:
            Boolean indicating if a restart with new pseudocode is needed
        """
        # If any agent has confidence below threshold, restart
        very_low_confidence_threshold = 30
        restart_needed = any(conf < very_low_confidence_threshold for conf in agents_confidence.values())
        
        if restart_needed and self._verbose:
            print(termcolor.colored("Low confidence detected. Restarting with new pseudocode.", "yellow"))
        
        return restart_needed
