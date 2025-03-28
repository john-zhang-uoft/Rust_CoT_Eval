#!/usr/bin/env python3
import sys
import os
import unittest
from unittest.mock import patch

# Add the parent directory to the system path to import ContentParser
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from content_parser import ContentParser, ParseError


class TestContentParser(unittest.TestCase):
    def setUp(self):
        self.parser = ContentParser()

    def test_entry_point_variations(self):
        variations = self.parser._entry_point_variations("testFunction")
        self.assertEqual(variations, ["testFunction", "test_function", "testFunction"])
        
        variations = self.parser._entry_point_variations("calculate_sum")
        self.assertEqual(variations, ["calculate_sum", "calculate_sum", "calculate_sum"])

    def test_get_function_implementation(self):
        # Test normal function
        rust_code = """
fn main() {
    println!("Hello, world!");
}

fn test_func(a: i32, b: i32) -> i32 {
    let result = a + b;
    // This is a comment with braces { }
    println!("The result is: {", result);
    /* This is a multiline comment
       with a brace { */
    let s = "A string with braces {";
    result
}

fn another_func() {
    println!("Another function");
}
"""
        impl = self.parser.get_function_implementation(rust_code, "test_func")
        # Now we shouldn't find the function signature in the result
        self.assertNotIn("fn test_func(a: i32, b: i32) -> i32 {", impl)
        self.assertIn("let result = a + b;", impl)
        self.assertIn("println!(\"The result is: {\", result);", impl)
        self.assertIn("let s = \"A string with braces {\";", impl)
        self.assertIn("result", impl)
        
        # Test with nested functions
        rust_code_nested = """
fn outer_func() {
    println!("Outer function");
    
    fn inner_func() {
        println!("Inner function");
    }
    
    inner_func();
}
"""
        impl = self.parser.get_function_implementation(rust_code_nested, "outer_func")
        # Now we shouldn't find the function signature in the result
        self.assertNotIn("fn outer_func() {", impl)
        self.assertIn("println!(\"Outer function\");", impl)
        self.assertIn("fn inner_func() {", impl)
        self.assertIn("inner_func();", impl)

    def test_parser_with_description(self):
        prompt = "Write a function that adds two numbers"
        content = """
I'll implement a function to add two numbers in Rust:

```rust
fn add_two_numbers(a: i32, b: i32) -> i32 {
    a + b
}
```

This function takes two integers as input and returns their sum.
"""
        entry_point = "add_two_numbers"
        
        result = self.parser(prompt, content, entry_point)
        # We should only get the function body now
        self.assertEqual(result.strip(), "a + b")

    def test_parser_without_description(self):
        prompt = "Implement a function that calculates factorial"
        content = """fn factorial(n: u64) -> u64 {
    if n == 0 || n == 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}"""
        entry_point = "factorial"
        
        result = self.parser(prompt, content, entry_point)
        self.assertNotIn("fn factorial(n: u64)", result)
        self.assertIn("if n == 0 || n == 1", result)
        self.assertIn("n * factorial(n - 1)", result)

    def test_parse_error(self):
        prompt = "Write a sorting function"
        content = "This is just some text without any code"
        entry_point = "sort"
        
        with self.assertRaises(ParseError):
            self.parser(prompt, content, entry_point)


if __name__ == "__main__":
    unittest.main()