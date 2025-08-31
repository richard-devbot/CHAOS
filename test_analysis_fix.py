#!/usr/bin/env python3
"""
Test script to validate the Analysis agent fix for GitHub AI JSON responses
"""
import sys
import os
sys.path.append('.')

def test_analysis_validation():
    """Test the analysis report validation logic"""
    
    print("Testing Analysis report validation...")
    
    # Test cases for different response types
    test_cases = [
        # Case 1: Valid string response
        {"input": "This is a valid analysis report", "expected": "This is a valid analysis report"},
        
        # Case 2: None response
        {"input": None, "expected": "Analysis failed: No report generated from the LLM response."},
        
        # Case 3: Empty string response
        {"input": "", "expected": "Analysis failed: Empty report generated from the LLM response."},
        
        # Case 4: Dictionary with report key (GitHub AI style)
        {"input": {"report": "Analysis from GitHub AI"}, "expected": "Analysis from GitHub AI"},
        
        # Case 5: Dictionary without report key
        {"input": {"analysis": "Wrong key"}, "expected": "{'analysis': 'Wrong key'}"},
        
        # Case 6: Integer response
        {"input": 123, "expected": "123"},
        
        # Case 7: List response
        {"input": ["item1", "item2"], "expected": "['item1', 'item2']"}
    ]
    
    print(f"Running {len(test_cases)} test cases...")
    
    for i, case in enumerate(test_cases, 1):
        result = validate_analysis_report(case["input"])
        success = result == case["expected"]
        status = "✓ PASS" if success else "✗ FAIL"
        
        print(f"Test {i}: {status}")
        if not success:
            print(f"  Input: {case['input']}")
            print(f"  Expected: {case['expected']}")
            print(f"  Got: {result}")
        print()
    
    print("Analysis validation test completed!")

def validate_analysis_report(analysis_report):
    """
    Validation logic extracted from the fixed analysis agent
    """
    # Validate and ensure we have a string report
    if analysis_report is None:
        return "Analysis failed: No report generated from the LLM response."
    elif not isinstance(analysis_report, str):
        # If it's a dict or other type, try to extract the string value
        if isinstance(analysis_report, dict) and "report" in analysis_report:
            analysis_report = analysis_report["report"]
        else:
            analysis_report = str(analysis_report)
    
    # Ensure we have a non-empty string
    if not analysis_report.strip():
        return "Analysis failed: Empty report generated from the LLM response."
    
    return analysis_report

if __name__ == "__main__":
    test_analysis_validation()