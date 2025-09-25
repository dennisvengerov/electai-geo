#!/usr/bin/env python3
"""
Test script to verify the fallback mechanism works when Gemini quota is exceeded
"""

import asyncio
import os
from geo_analyzer import GEOAnalyzer


async def test_fallback_mechanism():
    """Test that the system properly falls back to legacy analysis when Gemini fails"""
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if not openai_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return
    
    print("🧪 Testing fallback mechanism...")
    
    # Initialize analyzer with batch analysis enabled
    analyzer = GEOAnalyzer(
        openai_api_key=openai_key,
        gemini_api_key=gemini_key,  # This might hit quota limits
        max_concurrent=2,
        use_batch_analysis=True
    )
    
    print(f"📊 Batch analysis enabled: {analyzer.use_batch_analysis}")
    print(f"🔑 Gemini key available: {gemini_key is not None}")
    
    def progress_callback(current, total, status):
        print(f"Progress: {current}/{total} - {status}")
    
    try:
        # Run a small analysis that should trigger the quota error and fallback
        results = await analyzer.analyze_company_visibility(
            company_name="Burt's Bees",
            industry_context="lip balm",
            num_queries=3,  # Small number to minimize OpenAI usage
            progress_callback=progress_callback
        )
        
        print("\n✅ Analysis completed!")
        print(f"📈 Analysis method: {results.get('analysis_method', 'unknown')}")
        print(f"🎯 Total queries: {results.get('summary', {}).get('total_queries', 0)}")
        print(f"📊 Total citations: {results.get('summary', {}).get('total_citations', 0)}")
        
        if results.get('analysis_method') == 'legacy_fallback':
            print(f"🔄 Fallback reason: {results.get('fallback_reason', 'Unknown')}")
            print("✅ Fallback mechanism working correctly!")
        elif results.get('analysis_method') == 'batch_gemini':
            print("🤖 Batch analysis succeeded (quota not exceeded)")
        else:
            print("🔧 Used legacy analysis directly")
        
        # Check that we got valid results
        query_results = results.get('query_results', [])
        print(f"📋 Query results: {len(query_results)} processed")
        
        if len(query_results) > 0:
            print("✅ Test passed - got valid results despite quota issues")
        else:
            print("❌ Test failed - no query results returned")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_fallback_mechanism())
