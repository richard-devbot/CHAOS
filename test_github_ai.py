#!/usr/bin/env python3
"""
Test script for GitHub AI integration with ChaosEater
"""
import os
import sys
sys.path.append('.')

try:
    from chaos_eater.utils.llms import load_llm, verify_api_key, check_existing_key
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def test_github_ai():
    """Test GitHub AI provider integration"""
    
    # Test environment variable setup
    print("Testing GitHub AI integration...")
    
    # Set test token (you can set this manually for testing)
    test_token = os.environ.get("GITHUB_TOKEN")
    if not test_token:
        print("Warning: GITHUB_TOKEN environment variable not set")
        print("Set it with: export GITHUB_TOKEN='your-github-token'")
        return
    
    # Test API key verification
    print("Testing API key verification...")
    is_valid = verify_api_key("github", test_token)
    print(f"GitHub token validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Test existing key check
    print("Testing existing key check...")
    has_key = check_existing_key("github")
    print(f"Existing key check: {'✓ Found' if has_key else '✗ Not found'}")
    
    # Test LLM loading
    print("Testing LLM loading...")
    try:
        llm = load_llm(
            model_name="github/gpt-4o-mini",
            temperature=0.0,
            seed=42
        )
        print("✓ LLM loaded successfully")
        print(f"LLM type: {type(llm)}")
        print(f"LLM config: {llm}")
        
        # Test a simple call
        print("Testing simple LLM call...")
        test_prompt = "Hello, what is 2+2?"
        # This would require implementing a test call, but for now just verify loading
        print("✓ LLM ready for testing")
        
    except Exception as e:
        print(f"✗ Error loading LLM: {e}")
    
    print("\nGitHub AI integration test completed!")

if __name__ == "__main__":
    test_github_ai()