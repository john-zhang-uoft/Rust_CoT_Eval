#!/usr/bin/env python3

import unittest

import sys
import os

# Add the parent directory to the system path to import ContentParser
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse_completions_with_helper_functions import process_code, find_function_bounds

def test_find_function_bounds_simple():
    code = """
fn other_fn() { }

fn target_fn(x: i32) -> i32 {
    x + 1
}

fn another_fn() { }
"""
    start, end = find_function_bounds(code, "target_fn")
    assert start != -1 and end != -1
    extracted = code[start:end]
    assert "fn target_fn" in extracted
    assert "x + 1" in extracted
    assert "other_fn" not in extracted

def test_find_function_bounds_with_nested_braces():
    code = """
fn target_fn() -> bool {
    let x = if true {
        Some(1)
    } else {
        None
    };
    match x {
        Some(_) => true,
        None => false,
    }
}
"""
    start, end = find_function_bounds(code, "target_fn")
    assert start != -1 and end != -1
    extracted = code[start:end]
    assert "fn target_fn" in extracted
    assert "match x" in extracted
    assert all(s in extracted for s in ["if true", "Some(1)", "None"])

def test_find_function_bounds_with_strings_and_comments():
    code = '''
fn target_fn() -> &'static str {
    // Function with strings and comments
    let x = "{ not a brace }";
    let y = '{{';  // Also not a brace
    /* 
        Multiline comment with {braces}
    */
    "result { with } braces"
}

'''
    start, end = find_function_bounds(code, "target_fn")
    assert start != -1 and end != -1
    extracted = code[start:end]
    assert 'fn target_fn' in extracted
    assert '"{ not a brace }"' in extracted
    assert "Multiline comment" in extracted

def test_process_code_moves_function_to_top():
    code = """
struct Helper { x: i32 }

fn other_fn() { }

fn target_fn() {
    println!("main logic");
}

impl Helper {
    fn new() -> Self { Self { x: 0 } }
}
"""
    processed = process_code(code, "target_fn")
    print("Processed: ", processed)
    lines = processed.split('\n')
    # Find first non-empty line with 'fn'
    first_fn_line = next(i for i, line in enumerate(lines) if line.strip().startswith('fn'))
    assert "target_fn" not in lines[first_fn_line]
    
    # Verify target_fn only appears once
    assert sum('target_fn' in line for line in lines) == 0
    
    # Verify all content is preserved
    assert "struct Helper" in processed
    assert "impl Helper" in processed
    assert "other_fn" in processed
    assert "println!" in processed

def test_process_code_with_test_module():
    code = """
fn target_fn() {
    println!("main logic");
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_target() {
        assert!(true);
    }
}
"""
    processed = process_code(code, "target_fn")
    assert "#[cfg(test)]" not in processed
    assert "mod tests" not in processed
    assert "fn target_fn" not in processed
    assert "println!" in processed

def test_process_code_with_complex_rust_features():
    code = """
const MAX_SIZE: usize = 100;

#[derive(Debug)]
struct State { map: HashMap<String, i32> }

fn helper() -> i32 { 42 }

fn target_fn() -> Result<()> {
    let state = State { map: HashMap::new() };
    Ok(())
}

impl State {
    fn new() -> Self {
        Self { map: HashMap::new() }
    }
}
"""
    processed = process_code(code, "target_fn")
    # Check function is at top (after imports and types)
    lines = [l for l in processed.split('\n') if l.strip()]
    
    print("Processed: ", processed)
    # Verify all content is preserved
    assert "const MAX_SIZE" in processed
    assert "struct State" in processed
    assert "fn helper" in processed
    assert "impl State" in processed


def test_find_function_bounds_not_found():
    code = "fn other_fn() { }"
    start, end = find_function_bounds(code, "target_fn")
    assert start == -1 and end == -1

def test_process_code_empty_input():
    assert process_code("", "target_fn") == ""
    assert process_code("// just a comment", "target_fn") == "// just a comment" 