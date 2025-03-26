#!/usr/bin/env python3
import re
from typing import Tuple

def parse_prompt(prompt: str) -> Tuple[str, str]:
    """Test implementation of the parse_prompt function"""
    # Handle the Rust-specific prompt structure
    # Look for any function name after fn main(){} that's not main
    fn_defs = re.findall(r'fn\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?\s*\(', prompt)
    fn_defs = [fn_name for fn_name in fn_defs if fn_name != "main"]
    
    if fn_defs:
        # Take the first non-main function as the entry point
        entry_point = fn_defs[0]
        
        # Extract everything after fn main(){} 
        main_match = re.search(r'fn\s+main\s*\(\s*\)\s*\{\s*\}(.*)', prompt, re.DOTALL)
        
        if main_match:
            content_after_main = main_match.group(1).strip()
            
            # Find the complete function signature
            fn_signature_match = re.search(fr'fn\s+{entry_point}(?:<[^>]*>)?\s*\([^{{]*', content_after_main, re.DOTALL)
            
            if fn_signature_match:
                # Get the text between fn main(){} and the function signature (likely imports)
                imports_match = re.search(fr'(.*?)fn\s+{entry_point}', content_after_main, re.DOTALL)
                imports = imports_match.group(1).strip() if imports_match else ""
                
                # Complete declaration includes imports and function signature
                declaration = imports + ("\n" if imports else "") + fn_signature_match.group(0)
                return declaration, entry_point
            
        # Fallback: just try to get the function signature
        declaration_match = re.search(fr'(fn\s+{entry_point}(?:<[^>]*>)?\s*\([^{{]*)', prompt, re.DOTALL)
        declaration = declaration_match.group(1) if declaration_match else ""
        
        return declaration, entry_point
    
    # Generic approach if we couldn't find a non-main function
    # Extract the declaration from the prompt
    declaration_match = re.search(r'declaration:\s*(.+?)(?=\n\n|\Z)', prompt, re.DOTALL | re.IGNORECASE)
    if not declaration_match:
        # Try an alternative pattern if the first one fails
        declaration_match = re.search(r'according to this declaration:\s*(.+?)(?=\n\n|\Z)', prompt, re.DOTALL | re.IGNORECASE)
    
    declaration = declaration_match.group(1).strip() if declaration_match else ""
    
    # Default to solution if we couldn't find any other function name
    return declaration, "solution"

# Test with the example prompt
example_prompt = '''Write a Rust function `has_close_elements(numbers:Vec<f32>, threshold: f32) -> bool` to solve the following problem:
Check if in given list of numbers, are any two numbers closer to each other than
given threshold.

You should implement the function according to this declaration:
fn main(){}

use std::{slice::Iter, cmp::{max, self}, mem::replace, collections::{HashSet, HashMap}, ops::Index, ascii::AsciiExt};
use rand::Rng;
use regex::Regex;
use md5;
use std::any::{Any, TypeId};

fn has_close_elements(numbers:Vec<f32>, threshold: f32) -> bool{
'''

declaration, entry_point = parse_prompt(example_prompt)

print(f"Entry point: {entry_point}")
print("\nDeclaration:")
print("-" * 80)
print(declaration)
print("-" * 80)

# Test with a more complex example
complex_example = '''Write a Rust function that sorts a vector using bubble sort.

You should implement the function according to this declaration:
fn main(){}

use std::cmp::Ordering;

fn bubble_sort<T: Ord>(arr: &mut [T]) -> &mut [T] {
'''

declaration2, entry_point2 = parse_prompt(complex_example)

print(f"\nSecond test - Entry point: {entry_point2}")
print("\nDeclaration:")
print("-" * 80)
print(declaration2)
print("-" * 80) 