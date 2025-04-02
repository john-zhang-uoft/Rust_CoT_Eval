import os
import time
import openai
from openai import OpenAI
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class CodeGenerationModel(ABC):
    """Base class for code generation models"""
    
    @abstractmethod
    def generate_code(self, prompt: str, n: int = 1) -> List[str]:
        """
        Generate code completions based on the prompt
        
        Args:
            prompt: The prompt to generate code from
            n: Number of completions to generate
            
        Returns:
            List of generated code completions
        """
        pass
    
    @abstractmethod
    def generate_chat_completion(self, messages: List[Dict[str, str]], system_prompt: str = None) -> str:
        """Generate a chat completion using OpenAI Chat API"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model"""
        pass

class OpenAIChatModel(CodeGenerationModel):
    """Implementation of CodeGenerationModel using OpenAI Chat API"""
    
    def __init__(
        self, 
        model: str, 
        temperature: float = 0.2, 
        top_p: float = 0.95, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None
    ):
        """
        Initialize an OpenAI chat model
        
        Args:
            model: The OpenAI model name to use
            temperature: Temperature parameter for generation
            top_p: Top p parameter for generation
            api_key: OpenAI API key (defaults to env var)
            base_url: Custom API base URL (optional)
        """
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        
        # Set up the client
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if base_url:
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self._client = OpenAI(api_key=api_key)
    
    def generate_code(self, prompt: str, system_prompt: str = None, n: int = 1) -> List[str]:
        """Generate code using OpenAI Chat API"""
        messages = [ { "role": "user", "content": prompt } ]
        if system_prompt:
            messages.insert(0, { "role": "system", "content": system_prompt })
        
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    n=n
                )
                
                content_list = []
                for i in range(n):
                    message = response.choices[i].message
                    assert message.role == "assistant"
                    content_list.append(message.content)
                
                return content_list
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"API EXCEPTION: {e}. Retrying ({retry_count}/{max_retries})...")
                    time.sleep(2)  # Wait before retrying
                else:
                    print(f"Failed after {max_retries} attempts: {e}")
                    # Return empty strings as fallback
                    return [""] * n
    
    def generate_chat_completion(self, messages: List[Dict[str, str]], system_prompt: str = None) -> str:
        """Generate a chat completion using OpenAI Chat API"""
        if system_prompt:
            messages.insert(0, { "role": "system", "content": system_prompt })
        
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            top_p=self._top_p,
            n=1
        )   
        return response.choices[0].message.content
    
    @property
    def model_name(self) -> str:
        return self._model

class LambdaLabsModel(OpenAIChatModel):
    """Implementation for Lambda Labs API which uses OpenAI-compatible interface"""
    
    def __init__(
        self, 
        model: str, 
        temperature: float = 0.2, 
        top_p: float = 0.95
    ):
        """
        Initialize a Lambda Labs model
        
        Args:
            model: The model name to use (e.g., 'llama3.2-3b-instruct')
            temperature: Temperature parameter for generation
            top_p: Top p parameter for generation
        """
        api_key = os.getenv("LAMBDALABS_API_KEY")
        base_url = os.getenv("CUSTOM_API_BASE", "https://api.lambdalabs.com/v1")
        
        super().__init__(
            model=model,
            temperature=temperature,
            top_p=top_p,
            api_key=api_key,
            base_url=base_url
        ) 