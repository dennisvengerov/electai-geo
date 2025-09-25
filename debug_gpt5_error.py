#!/usr/bin/env python3
"""
Debug script to identify the GPT-5 error
"""

import asyncio
import os
from geo_analyzer import GEOAnalyzer


async def debug_gpt5_error():
    """Debug the GPT-5 error specifically"""
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    print("🔍 Debug: GPT-5 Error Investigation")
    print(f"   OpenAI Key: {'✅' if openai_key else '❌'}")
    print(f"   Gemini Key: {'✅' if gemini_key else '❌'}")
    
    if not openai_key:
        print("❌ Cannot test without OpenAI API key")
        return
    
    if not gemini_key:
        print("❌ Cannot test without Gemini API key")
        return
    
    # Test GPT-5 directly
    print("\n🧪 Test 1: Direct GPT-5 Call")
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai_key)
        
        response = await client.chat.completions.create(
            model="gpt-5",
            messages=[{
                "role": "user",
                "content": "Generate a simple ranked list of the top 3 lip balm brands with explanations."
            }],
            temperature=0.7,
            max_tokens=500
        )
        
        print("✅ Direct GPT-5 call successful!")
        print(f"   Response: {response.choices[0].message.content[:200]}...")
        
    except Exception as e:
        print(f"❌ Direct GPT-5 call failed: {str(e)}")
        if "model" in str(e).lower():
            print("🤖 GPT-5 model may not be available. Trying GPT-4o...")
            
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": "Generate a simple ranked list of the top 3 lip balm brands with explanations."
                    }],
                    temperature=0.7,
                    max_tokens=500
                )
                print("✅ GPT-4o call successful!")
                print("💡 Solution: Switch to GPT-4o model")
                return "use_gpt4o"
                
            except Exception as e2:
                print(f"❌ GPT-4o also failed: {str(e2)}")
        
        import traceback
        traceback.print_exc()
        return "api_error"
    
    # Test the analyzer's GPT-5 method
    print("\n🧪 Test 2: Analyzer GPT-5 Method")
    try:
        analyzer = GEOAnalyzer(
            openai_api_key=openai_key,
            gemini_api_key=gemini_key,
            use_batch_analysis=False
        )
        
        response = await analyzer._make_ranking_query("best lip balm brands 2024")
        print("✅ Analyzer GPT-5 method successful!")
        print(f"   Response length: {len(response)}")
        print(f"   Response preview: {response[:200]}...")
        
    except Exception as e:
        print(f"❌ Analyzer GPT-5 method failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return "analyzer_error"
    
    # Test the full pipeline with minimal data
    print("\n🧪 Test 3: Minimal Pipeline Test")
    try:
        analyzer = GEOAnalyzer(
            openai_api_key=openai_key,
            gemini_api_key=gemini_key,
            max_concurrent=1,
            use_batch_analysis=True
        )
        
        results = await analyzer.analyze_company_visibility(
            company_name="Burt's Bees",
            industry_context="lip balm",
            num_queries=1
        )
        
        print("✅ Minimal pipeline test successful!")
        print(f"   Analysis method: {results.get('analysis_method')}")
        print(f"   Query results: {len(results.get('query_results', []))}")
        
        # Check the actual response content
        query_results = results.get('query_results', [])
        if query_results:
            first_result = query_results[0]
            response_text = first_result.get('response', '')
            print(f"   First response length: {len(response_text)}")
            print(f"   First response preview: {response_text[:200]}...")
            
            if response_text.startswith("Error:"):
                print(f"❌ GPT-5 error detected in response: {response_text}")
                return "gpt5_error_in_pipeline"
        
    except Exception as e:
        print(f"❌ Minimal pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return "pipeline_error"
    
    print("\n✅ All tests passed - GPT-5 seems to be working!")
    return "success"


if __name__ == "__main__":
    result = asyncio.run(debug_gpt5_error())
    
    print(f"\n🎯 Debug Result: {result}")
    
    if result == "use_gpt4o":
        print("\n💡 SOLUTION: Update the model from 'gpt-5' to 'gpt-4o'")
        print("   The 'gpt-5' model may not be available yet.")
    elif result in ["api_error", "analyzer_error", "gpt5_error_in_pipeline"]:
        print("\n🔧 SOLUTION: Check API keys and model availability")
    elif result == "success":
        print("\n✅ No issues found - the problem may be intermittent")
