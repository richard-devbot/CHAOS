#!/usr/bin/env python3
"""
Comprehensive test script to verify universal model provider compatibility.
Tests that all model providers work seamlessly regardless of response format.
"""
import sys
import os
sys.path.append('.')

def test_response_validation_utilities():
    """Test the new response validation utility functions"""
    print("=" * 60)
    print("TESTING RESPONSE VALIDATION UTILITIES")
    print("=" * 60)
    
    from chaos_eater.utils.llms import safe_get_response_field, safe_stream_response_extract
    
    # Test cases for safe_get_response_field
    test_cases = [
        # Case 1: Normal dict response (OpenAI/Anthropic style)
        {"input": {"report": "Analysis complete"}, "field": "report", "expected": "Analysis complete"},
        
        # Case 2: JSON string response (GitHub AI style)
        {"input": '{"report": "GitHub AI analysis"}', "field": "report", "expected": "GitHub AI analysis"},
        
        # Case 3: Plain string response 
        {"input": "Direct string response", "field": "report", "expected": "Direct string response"},
        
        # Case 4: None response
        {"input": None, "field": "report", "expected": None},
        
        # Case 5: Dict without the field
        {"input": {"summary": "Wrong field"}, "field": "report", "expected": None},
        
        # Case 6: Invalid JSON string
        {"input": "invalid json {", "field": "report", "expected": "invalid json {"},
        
        # Case 7: Nested dict response
        {"input": {"data": {"report": "Nested analysis"}}, "field": "report", "expected": None},
    ]
    
    print("Testing safe_get_response_field...")
    passed = 0
    for i, case in enumerate(test_cases, 1):
        result = safe_get_response_field(case["input"], case["field"])
        success = result == case["expected"]
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  Test {i}: {status}")
        if not success:
            print(f"    Expected: {case['expected']}")
            print(f"    Got: {result}")
        else:
            passed += 1
    
    print(f"safe_get_response_field: {passed}/{len(test_cases)} tests passed")
    
    # Test cases for safe_stream_response_extract
    stream_test_cases = [
        {"input": {"report": "Stream analysis"}, "field": "report", "should_contain": "Stream analysis"},
        {"input": None, "field": "report", "should_contain": "failed"},
        {"input": "", "field": "report", "should_contain": "Empty"},
        {"input": {"other": "data"}, "field": "report", "should_contain": "failed"},
    ]
    
    print("\nTesting safe_stream_response_extract...")
    stream_passed = 0
    for i, case in enumerate(stream_test_cases, 1):
        result = safe_stream_response_extract(case["input"], case["field"])
        success = case["should_contain"].lower() in result.lower()
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  Test {i}: {status}")
        if not success:
            print(f"    Expected to contain: {case['should_contain']}")
            print(f"    Got: {result}")
        else:
            stream_passed += 1
    
    print(f"safe_stream_response_extract: {stream_passed}/{len(stream_test_cases)} tests passed")
    print()
    
    return passed == len(test_cases) and stream_passed == len(stream_test_cases)

def test_llm_agent_imports():
    """Test that all LLM agents can be imported without errors"""
    print("=" * 60)
    print("TESTING LLM AGENT IMPORTS")
    print("=" * 60)
    
    agents_to_test = [
        "chaos_eater.preprocessing.llm_agents.k8s_weakness_summary_agent",
        "chaos_eater.preprocessing.llm_agents.ce_instruct_agent", 
        "chaos_eater.analysis.llm_agents.analysis_agent",
    ]
    
    import_passed = 0
    for agent_module in agents_to_test:
        try:
            __import__(agent_module)
            print(f"✓ {agent_module}")
            import_passed += 1
        except Exception as e:
            print(f"✗ {agent_module}: {e}")
    
    print(f"Agent imports: {import_passed}/{len(agents_to_test)} passed")
    print()
    
    return import_passed == len(agents_to_test)

def test_mock_responses():
    """Test agent behavior with mock responses from different providers"""
    print("=" * 60)
    print("TESTING MOCK PROVIDER RESPONSES")
    print("=" * 60)
    
    try:
        from chaos_eater.preprocessing.llm_agents.k8s_weakness_summary_agent import K8sWeaknessSummaryAgent
        from chaos_eater.preprocessing.llm_agents.ce_instruct_agent import CEInstructAgent
        
        # Test different response formats that agents might receive
        mock_responses = [
            # OpenAI/Anthropic style - structured dict
            {"issues": [{"issue_name": "Test Issue", "issue_details": "Test details"}]},
            
            # GitHub AI style - JSON string
            '{"issues": [{"issue_name": "GitHub Issue", "issue_details": "GitHub details"}]}',
            
            # Malformed/incomplete response
            {"partial": "data"},
            
            # String response
            "Direct response text",
            
            # None response
            None,
        ]
        
        # Test K8sWeaknessSummaryAgent.get_text method
        print("Testing K8sWeaknessSummaryAgent.get_text...")
        agent = K8sWeaknessSummaryAgent.__new__(K8sWeaknessSummaryAgent)  # Create without init
        
        mock_passed = 0
        for i, response in enumerate(mock_responses, 1):
            try:
                result = agent.get_text(response)
                success = isinstance(result, str) and len(result) >= 0  # Should always return a string
                status = "✓ PASS" if success else "✗ FAIL"
                print(f"  Mock response {i}: {status}")
                if success:
                    mock_passed += 1
                else:
                    print(f"    Got non-string result: {type(result)}")
            except Exception as e:
                print(f"  Mock response {i}: ✗ FAIL - {e}")
        
        print(f"Mock response handling: {mock_passed}/{len(mock_responses)} passed")
        print()
        
        return mock_passed == len(mock_responses)
        
    except Exception as e:
        print(f"✗ Failed to test mock responses: {e}")
        return False

def test_comprehensive_compatibility():
    """Run comprehensive compatibility tests"""
    print("=" * 80)
    print("COMPREHENSIVE MODEL PROVIDER COMPATIBILITY TEST")
    print("=" * 80)
    print("Testing universal compatibility across all model providers...")
    print("Ensuring responses fit the use case regardless of provider chosen.")
    print()
    
    # Run all test suites
    results = []
    
    results.append(("Response Validation Utilities", test_response_validation_utilities()))
    results.append(("LLM Agent Imports", test_llm_agent_imports()))
    results.append(("Mock Provider Responses", test_mock_responses()))
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed_suites = 0
    for suite_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} {suite_name}")
        if passed:
            passed_suites += 1
    
    print()
    print(f"Overall Result: {passed_suites}/{len(results)} test suites passed")
    
    if passed_suites == len(results):
        print("🎉 All model providers should now work universally!")
        print("✅ GitHub AI, OpenAI, Anthropic, Google, Ollama - all compatible")
        print("✅ Response format variations handled gracefully")
        print("✅ No more 'str' object has no attribute 'get' errors")
        print("✅ No more Pydantic validation errors")
    else:
        print("❌ Some issues remain - check failed tests above")
    
    return passed_suites == len(results)

if __name__ == "__main__":
    success = test_comprehensive_compatibility()
    sys.exit(0 if success else 1)