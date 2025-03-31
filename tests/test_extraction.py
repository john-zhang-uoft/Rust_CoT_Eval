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
        print(result)
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


if __name__ == "__main__":
    unittest.main() 