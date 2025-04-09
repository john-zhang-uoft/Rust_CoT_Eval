#!/usr/bin/env python3
import sys
import os
import unittest

# Add the parent directory to the system path to import ContentParser
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsers.content_parser import ContentParser, ParseError


class TestExtractAllFunctions(unittest.TestCase):
    def setUp(self):
        self.parser = ContentParser()

    def test_extract_all_functions(self):
        # Test code with multiple functions including main and helpers
        rust_code = """
fn main() {
    let a = 5;
    let b = 7;
    let result = calculate_sum(a, b);
    println!("The sum is: {}", result);
    
    let factorial_result = factorial(5);
    println!("Factorial of 5 is: {}", factorial_result);
}

fn calculate_sum(a: i32, b: i32) -> i32 {
    // A helper function for addition
    add(a, b)
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn factorial(n: u64) -> u64 {
    if n == 0 || n == 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calculate_sum() {
        assert_eq!(calculate_sum(2, 3), 5);
    }
    
    #[test]
    fn test_factorial() {
        assert_eq!(factorial(5), 120);
    }
}
"""
        result = self.parser.extract_all_functions(rust_code, "calculate_sum")
        
        # Check that the main function is removed
        self.assertNotIn("fn main()", result)
        
        # Check that the test module is removed
        self.assertNotIn("#[cfg(test)]", result)
        self.assertNotIn("mod tests", result)
        
        # Check that all non-main functions are included
        self.assertIn("fn calculate_sum(a: i32, b: i32) -> i32", result)
        self.assertIn("fn add(a: i32, b: i32) -> i32", result)
        self.assertIn("fn factorial(n: u64) -> u64", result)
        
        # Check function implementations are preserved
        self.assertIn("add(a, b)", result)  # inside calculate_sum
        self.assertIn("a + b", result)      # inside add
        self.assertIn("n * factorial(n - 1)", result)  # inside factorial

    def test_entry_point_variations(self):
        # Test with a different entry point variation
        rust_code = """
fn main() {
    let result = calculateSum(10, 20);
    println!("Result: {}", result);
}

fn calculateSum(a: i32, b: i32) -> i32 {
    a + b
}
"""
        # Try to extract with snake_case entry point
        result = self.parser.extract_all_functions(rust_code, "calculate_sum")
        
        # Should find the camelCase version
        self.assertIn("fn calculateSum(a: i32, b: i32) -> i32", result)
        self.assertIn("a + b", result)
        self.assertNotIn("fn main()", result)

    def test_no_main_function(self):
        # Test code with no main function
        rust_code = """
fn calculate_sum(a: i32, b: i32) -> i32 {
    a + b
}

fn multiply(a: i32, b: i32) -> i32 {
    a * b
}
"""
        result = self.parser.extract_all_functions(rust_code, "calculate_sum")
        
        # Should include all functions since there's no main
        self.assertIn("fn calculate_sum(a: i32, b: i32) -> i32", result)
        self.assertIn("fn multiply(a: i32, b: i32) -> i32", result)

    def test_only_tests(self):
        # Test code with only test functions
        rust_code = """
fn calculate_sum(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calculate_sum() {
        assert_eq!(calculate_sum(2, 3), 5);
    }
}
"""
        result = self.parser.extract_all_functions(rust_code, "calculate_sum")
        
        # Should keep regular function and remove tests
        self.assertIn("fn calculate_sum(a: i32, b: i32) -> i32", result)
        self.assertNotIn("#[cfg(test)]", result)
        self.assertNotIn("mod tests", result)
        self.assertNotIn("fn test_calculate_sum", result)

    def test_function_called_by_main(self):
        rust_block = """
```rust
```rust
fn main() {}

use std::{slice::Iter, cmp::{max, self}, mem::replace, collections::{HashSet, HashMap}, ops::Index, ascii::AsciiExt};
use rand::Rng;
use regex::Regex;
use md5;
use std::any::{Any, TypeId};

fn solve_161(s: &str) -> String {
    let mut result = String::new();
    let mut has_letters = false;

    for c in s.chars() {
        if c.is_ascii_alphabetic() {
            has_letters = true;
            if c.is_ascii_lowercase() {
                result.push(c.to_ascii_uppercase());
            } else {
                result.push(c.to_ascii_lowercase());
            }
        } else {
            result.push(c);
        }
    }

    if !has_letters {
        result = s.chars().rev().collect();
    }

    result
}

fn main() {
    println!("{}", solve_161("Hello World!"));
    println!("{}", solve_161("12345"));
}
```
"""
        result = self.parser("", rust_block, "solve_161", extract_all=True)
        # print(result)
        # Assert that solve_161 is only defined once
        self.assertEqual(result.count("fn solve_161(s: &str) -> String"), 1)
        # Assert that main is not included
        self.assertNotIn("fn main()", result)
        # Assert that the use of std::ascii::AsciiExt is not included
        self.assertNotIn("use std::ascii::AsciiExt;", result)
        

    def test_real_world_example(self):
        to_parse = """Here is the implementation of the `maximum_120` function in Rust according to the provided plan:

```rust
use std::{slice::Iter, cmp::{max, self}, mem::replace, collections::{HashSet, HashMap}, ops::Index, ascii::AsciiExt};
use rand::Rng;
use regex::Regex;
use md5;
use std::any::{Any, TypeId};

fn maximum_120(arr: Vec<i32>, k: i32) -> Vec<i32> {
    // Check if k is 0, if so return an empty list
    if k == 0 {
        return Vec::new();
    }

    // Sort the input array arr in descending order
    let mut sorted_arr = arr;
    sorted_arr.sort_unstable_by(|a, b| b.cmp(a));

    // Initialize an empty list result
    let mut result: Vec<i32> = Vec::new();

    // Iterate over the sorted array arr and add the first k elements to the result list
    for i in 0..k as usize {
        result.push(sorted_arr[i]);
    }

    // Sort the result list in ascending order
    result.sort_unstable();

    // Return the result list
    result
}

fn main() {
    let arr = vec![1, 2, 3, 4, 5];
    let k = 3;
    let result = maximum_120(arr, k);
    println!("{:?}", result);
}
```
"""
        entry_point = "maximum"
        prompt = "Implement the `maximum_120` function in Rust"
        result = self.parser(prompt, to_parse, entry_point, extract_all=True)
        print(result)

        assert(result.count("fn maximum_120(arr: Vec<i32>, k: i32) -> Vec<i32>") == 1)

if __name__ == "__main__":
    unittest.main() 