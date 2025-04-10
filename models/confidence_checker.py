#!/usr/bin/env python3
"""
ConfidenceCheckerAgent for evaluating the confidence level of another agent's output
"""

import re
from typing import Dict, Any, Optional, Tuple, List
import termcolor
import json

from models.code_generation_models import CodeGenerationModel
from parsers.json_parser import parse_json, JSONParseError

class ConfidenceChecker:
    """
    Class responsible for evaluating the confidence level of an agent in its output.
    It continues as if it were the original agent and reports confidence in the output.
    """
    
    def __init__(
        self,
        model: CodeGenerationModel,
        verbose: bool = False
    ):
        """
        Initialize a confidence checker
        
        Args:
            model: The underlying language model
            verbose: Whether to print detailed logs
        """
        self._model = model
        self._verbose = verbose
    
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
    
    def check_confidence(
        self, 
        agent_role_system_message: str,
        agent_generation: str,
        context: str,
        additional_post_prompt: str = ""
    ) -> Tuple[int, str]:
        """
        Check the confidence of the agent in its output
        
        Args:
            agent_role_system_message: System prompt of the agent to check confidence of
            agent_generation: The generation of the agent to check confidence of
            context: What the user said to the agent before the generation
            
        Returns:
            Tuple containing:
            - Confidence score (0-100)
            - Explanation of the confidence assessment
        """
        
        # Create a system message with the agent role
        system_message = {
            "role": "system",
            "content": agent_role_system_message
        }
        
        # Build the message list starting with the system message
        messages = [system_message]
        
        # Add any context as a user message if provided
        if context:
            messages.append({ "role": "user", "content": context })
        
        # Add the previous agent generation to maintain the conversation history
        messages.append({ "role": "assistant", "content": agent_generation })
        
        prompt = f"""
{additional_post_prompt}
How likey is your code correct for solving the problem?
Think step by step about your response and provide a confidence score between 0 and 100 in the following format:

{{
  "confidence": number between 0-100
}}
"""
        # Add the final user message asking for confidence assessment
        messages.append({ "role": "user", "content": prompt })
        
        # Get completion using the chat format
        response = self._model.generate_chat_completion(messages)
        
        if self._verbose:
            self._log(f"Confidence checker response: {response}", "cyan")
        
        # Extract the confidence score from the response
        try:
            # Use the JSON parser to extract the confidence score
            confidence_data = parse_json(
                response, 
                required_fields=["confidence"], 
                default_values={"confidence": 50},
                verbose=self._verbose
            )
            confidence = confidence_data["confidence"]
            
            # Ensure confidence is an integer between 0 and 100
            if isinstance(confidence, str) and confidence.isdigit():
                confidence = int(confidence)
            elif not isinstance(confidence, int):
                try:
                    confidence = int(float(confidence))
                except (ValueError, TypeError):
                    self._log(f"Could not convert confidence to integer: {confidence}", "red")
                    confidence = 50  # Default to medium confidence
            
            # Clamp to range 0-100
            confidence = max(0, min(100, confidence))
            
            return confidence, response
            
        except JSONParseError as e:
            self._log(f"Error extracting confidence score: {str(e)}", "red")
            # Return a default confidence score of 50 (medium confidence)
            return 50, response
    
 