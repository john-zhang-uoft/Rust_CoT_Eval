#!/usr/bin/env python3
"""
JSON parser module for extracting JSON data from text and validating required fields.
This module is designed to be used by various agents that need to extract structured JSON 
from language model responses.
"""

import re
import json
from typing import Dict, Any, List, Optional, Union, Tuple


class JSONParseError(Exception):
    """Exception raised when JSON parsing fails."""
    pass


class JSONParser:
    """
    Parser for extracting JSON from text and validating required fields.
    
    This class provides methods to extract JSON from various formats including:
    - JSON in markdown code blocks
    - JSON directly in text
    - Partial JSON that needs to be completed
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the JSON parser.
        
        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
    
    def _log(self, message: str):
        """Log message if verbose is True"""
        if self.verbose:
            print(message)
    
    def extract_json(
        self, 
        text: str, 
        required_fields: Optional[List[str]] = None,
        default_values: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract JSON from text and validate required fields.
        
        Args:
            text: Text containing JSON
            required_fields: List of required fields in the JSON
            default_values: Default values for missing fields
            
        Returns:
            Dictionary containing the extracted JSON data
            
        Raises:
            JSONParseError: If JSON parsing fails and no fallback can be created
        """
        # Use default empty lists if parameters are None
        required_fields = required_fields or []
        default_values = default_values or {}
        
        # Try multiple extraction methods
        extraction_methods = [
            self._extract_from_code_block,
            self._extract_from_specific_fields,
            self._extract_any_json
        ]
        
        extraction_errors = []
        
        # Try each extraction method
        for method in extraction_methods:
            try:
                json_dict = method(text, required_fields)
                
                # If we got a result, ensure all required fields are present
                for field in required_fields:
                    if field not in json_dict:
                        if field in default_values:
                            json_dict[field] = default_values[field]
                            self._log(f"Added missing required field '{field}' with default value: {default_values[field]}")
                        else:
                            json_dict[field] = "Not specified"
                            self._log(f"Added missing required field '{field}' with 'Not specified'")
                
                return json_dict
            
            except Exception as e:
                extraction_errors.append(f"{method.__name__}: {str(e)}")
                continue
        
        # If we've exhausted all methods, try to build a fallback response
        if default_values and required_fields:
            self._log(f"All extraction methods failed. Building fallback response.")
            fallback = {}
            for field in required_fields:
                if field in default_values:
                    fallback[field] = default_values[field]
                else:
                    fallback[field] = "Not specified"
            
            # Add parse error information
            fallback["parse_error"] = "Failed to extract valid JSON from text"
            fallback["extraction_errors"] = extraction_errors
            
            return fallback
        
        # If we can't build a fallback, raise an error with all the extraction errors
        error_msg = "Failed to extract JSON using all methods: " + "; ".join(extraction_errors)
        raise JSONParseError(error_msg)
    
    def _normalize_json_string(self, json_str: str) -> str:
        """
        Normalize a JSON string to make it more likely to parse correctly.
        
        Args:
            json_str: The JSON string to normalize
            
        Returns:
            Normalized JSON string
        """
        # Replace various newline formats with standard newline
        normalized = json_str.replace('\r\n', '\n').replace('\r', '\n')
        
        # Replace tabs with spaces
        normalized = normalized.replace('\t', '  ')
        
        # More aggressively remove ALL control characters (including the problematic ones at string starts)
        normalized = ''.join(ch if (ord(ch) >= 32 or ch in '\n\r\t') else ' ' for ch in normalized)
        
        # Special handling for multiline strings - if we find a pattern like "key": "↵ 
        # This is a very common pattern in LLM output that causes JSON parsing issues
        normalized = re.sub(r'"([^"]+)"(\s*):(\s*)"(\s*)\n(\s*)', r'"\1"\2:\3"', normalized)
        
        # Fix common JSON formatting issues
        # Replace single quotes with double quotes (only if unescaped)
        # This is a simplified approach - a proper fix would need a more sophisticated parser
        normalized = re.sub(r"(?<![\\])\'", '"', normalized)
        
        # Handle trailing commas in objects and arrays
        normalized = re.sub(r",(\s*[\]}])", r"\1", normalized)
        
        # More robust handling of multiline strings in JSON
        # This converts multiline strings to proper JSON with \n
        normalized = self._fix_multiline_strings(normalized)
        
        # Ensure properly escaped newlines in strings
        normalized = normalized.replace('\\n', '\\\\n')
        normalized = normalized.replace('\\t', '\\\\t')
        
        # Try to parse, and if it fails, attempt more aggressive normalization
        try:
            json.loads(normalized)
        except json.JSONDecodeError as e:
            self._log(f"Initial normalization wasn't sufficient, applying additional fixes: {str(e)}")
            
            # Extreme case: remove most whitespace from the JSON string
            # Remove all whitespace between the keys and values
            if "control character" in str(e):
                # Use a more aggressive approach: manually reconstruct the JSON
                # Parse out key-value pairs directly
                self._log("Using aggressive control character removal")
                normalized = self._aggressively_fix_json(normalized)
            elif "escape" in str(e) or "Expecting" in str(e):
                # Try to fix syntax issues
                self._log("Attempting to fix JSON syntax")
                # Replace problematic escapes
                normalized = re.sub(r'([^\\])\\([^"\\/bfnrtu])', r'\1\\\\\2', normalized)
                # Fix broken quotes
                normalized = re.sub(r'([^\\])"([^"]*?[^\\])"', r'\1\\"\2\\"', normalized)
        
        self._log(f"Normalized JSON: {normalized[:100]}..." if len(normalized) > 100 else f"Normalized JSON: {normalized}")
        return normalized
        
    def _fix_multiline_strings(self, json_str: str) -> str:
        """
        Fix multiline strings in JSON by properly escaping newlines and indentation.
        This is a common issue with LLM outputs.
        
        Args:
            json_str: The JSON string to fix
            
        Returns:
            Fixed JSON string
        """
        # Find all string literals using regex
        pattern = r'"((?:[^"\\]|\\.)*)"'
        
        def replace_multiline(match):
            # Get the string content
            string_content = match.group(1)
            
            # If the string contains newlines, fix it
            if '\n' in string_content:
                # Replace newlines with \\n and normalize indentation
                fixed = string_content.replace('\n', '\\n')
                # Remove excessive spaces after newlines (common in LLM output)
                fixed = re.sub(r'\\n\s+', '\\n', fixed)
                return f'"{fixed}"'
            
            return match.group(0)
        
        return re.sub(pattern, replace_multiline, json_str)
    
    def _aggressively_fix_json(self, json_str: str) -> str:
        """
        Aggressively fix corrupted JSON by manually reconstructing it.
        This is a last resort for particularly problematic JSON.
        
        Args:
            json_str: The corrupted JSON string
            
        Returns:
            Fixed JSON string
        """
        # First, try to identify the overall structure (object or array)
        if not (json_str.strip().startswith('{') and json_str.strip().endswith('}')):
            # If it's not an object, we can't use this method
            return json_str
            
        # Remove all control characters completely
        clean_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
        
        # Try to extract the main parts of the JSON object
        try:
            # Find all key-value pairs
            key_value_pattern = r'"([^"]+)"\s*:\s*(?:"([^"]*?)"|(\d+)|(\{.*?\})|(\[.*?\])|(true|false|null))'
            matches = re.findall(key_value_pattern, clean_str, re.DOTALL)
            
            # Reconstruct the JSON object with clean formatting
            reconstructed = "{\n"
            for i, match in enumerate(matches):
                key = match[0]
                # Find which value group matched (one of them will have a value)
                value = next((v for v in match[1:] if v), "")
                
                # Add the key-value pair
                reconstructed += f'  "{key}": '
                
                # Handle different value types
                if value in ["true", "false", "null"]:
                    reconstructed += value
                elif value.isdigit() or (value.replace('.', '', 1).isdigit() and value.count('.') == 1):
                    reconstructed += value
                elif value.startswith('{') or value.startswith('['):
                    reconstructed += value
                else:
                    # String value
                    reconstructed += f'"{value}"'
                
                # Add comma if not the last item
                if i < len(matches) - 1:
                    reconstructed += ","
                reconstructed += "\n"
            
            reconstructed += "}"
            
            # Verify the reconstructed JSON is valid
            json.loads(reconstructed)
            return reconstructed
            
        except (json.JSONDecodeError, Exception) as e:
            self._log(f"Aggressive JSON fix failed: {str(e)}")
            # If aggressive reconstruction fails, return the original
            return json_str
    
    def _extract_from_code_block(self, text: str, required_fields: List[str]) -> Dict[str, Any]:
        """
        Extract JSON from a markdown code block.
        
        Args:
            text: Text containing a markdown code block with JSON
            required_fields: List of required fields in the JSON
            
        Returns:
            Dictionary containing the extracted JSON data
            
        Raises:
            JSONParseError: If JSON parsing fails
        """
        self._log("Attempting to extract JSON from code block")
        
        # Look for a code block with or without 'json' language marker
        match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', text, re.DOTALL)
        
        if not match:
            raise JSONParseError("No JSON code block found")
        
        json_str = match.group(1)
        self._log(f"Found JSON code block:\n{json_str}")
        
        # Normalize and parse
        json_str = self._normalize_json_string(json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise JSONParseError(f"Failed to parse JSON code block: {str(e)}")
    
    def _extract_from_specific_fields(self, text: str, required_fields: List[str]) -> Dict[str, Any]:
        """
        Extract JSON that contains specific required fields.
        
        Args:
            text: Text containing JSON
            required_fields: List of required fields in the JSON
            
        Returns:
            Dictionary containing the extracted JSON data
            
        Raises:
            JSONParseError: If JSON parsing fails
        """
        self._log("Attempting to extract JSON with specific fields")
        
        if not required_fields:
            raise JSONParseError("No required fields specified for field-specific extraction")
        
        # Build a regex pattern that searches for JSON containing all required fields
        fields_pattern = r''
        for field in required_fields:
            fields_pattern += fr'["\']?{re.escape(field)}["\']?\s*:'
            fields_pattern += r'[\s\S]*?'
        
        # Look for a JSON object containing all the required fields
        pattern = fr'({{\s*[\s\S]*?{fields_pattern}[\s\S]*?}})'
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            raise JSONParseError(f"No JSON with required fields {required_fields} found")
        
        json_str = match.group(1)
        self._log(f"Found JSON with required fields:\n{json_str}")
        
        # Normalize and parse
        json_str = self._normalize_json_string(json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise JSONParseError(f"Failed to parse JSON with required fields: {str(e)}")
    
    def _extract_any_json(self, text: str, required_fields: List[str]) -> Dict[str, Any]:
        """
        Extract any JSON object from the text.
        
        Args:
            text: Text containing JSON
            required_fields: List of required fields in the JSON (unused, but kept for consistency)
            
        Returns:
            Dictionary containing the extracted JSON data
            
        Raises:
            JSONParseError: If JSON parsing fails
        """
        self._log("Attempting to extract any JSON object")
        
        # Look for any JSON object
        match = re.search(r'({[\s\S]*?})', text, re.DOTALL)
        
        if not match:
            raise JSONParseError("No JSON object found")
        
        json_str = match.group(1)
        self._log(f"Found JSON object:\n{json_str}")
        
        # Normalize and parse
        json_str = self._normalize_json_string(json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise JSONParseError(f"Failed to parse JSON object: {str(e)}")


# Factory function to create a parser
def create_json_parser(verbose: bool = False) -> JSONParser:
    """
    Create a JSON parser.
    
    Args:
        verbose: Whether to print detailed logs
        
    Returns:
        JSONParser instance
    """
    return JSONParser(verbose=verbose)


# Convenience function for one-time parsing
def parse_json(
    text: str, 
    required_fields: Optional[List[str]] = None,
    default_values: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Parse JSON from text.
    
    Args:
        text: Text containing JSON
        required_fields: List of required fields in the JSON
        default_values: Default values for missing fields
        verbose: Whether to print detailed logs
        
    Returns:
        Dictionary containing the extracted JSON data
        
    Raises:
        JSONParseError: If JSON parsing fails and no fallback can be created
    """
    parser = JSONParser(verbose=verbose)
    return parser.extract_json(text, required_fields, default_values) 