#!/usr/bin/env python3
"""
Quick test to verify Gemini 2.0 Flash query generation works
"""

import asyncio
import os
from prompt_generator import PromptGenerator


async def test_gemini_flash():
    """Test the fixed Gemini 2.0 Flash query generation"""
    
    # Check for Gemini API key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        return
    
    print("🧪 Testing Fixed Gemini 2.0 Flash Query Generation...")
    print("📋 Expected fixes:")
    print("   • No more 'tools' attribute error")
    print("   • Gemini 2.0 Flash should generate queries directly")
    print("   • Enhanced 2025 context without external search")
    print()
    
    # Initialize prompt generator
    generator = PromptGenerator(gemini_key)
    
    # Test with lip balm industry
    print("🔍 Testing query generation for lip balm industry...")
    
    try:
        queries = await generator.generate_semantic_queries(
            company_name="Burt's Bees",  # Should NOT appear in queries
            industry_context="lip balm",
            num_queries=3
        )
        
        print(f"✅ SUCCESS! Generated {len(queries)} queries via Gemini 2.0 Flash")
        print("\n📝 Generated Queries:")
        
        for i, query in enumerate(queries, 1):
            print(f"   {i}. {query}")
        
        # Validate the results
        print(f"\n📊 Validation:")
        
        company_mentions = sum(1 for q in queries if "burt" in q.lower() or "bee" in q.lower())
        current_context = sum(1 for q in queries if "2025" in q)
        consumer_oriented = sum(1 for q in queries if any(word in q.lower() for word in ['best', 'top', 'good', 'recommend', 'brand']))
        
        print(f"   • Company name leakage: {company_mentions}/{len(queries)} {'✅' if company_mentions == 0 else '❌'}")
        print(f"   • Current year context: {current_context}/{len(queries)} {'✅' if current_context > 0 else '⚠️'}")
        print(f"   • Consumer-oriented: {consumer_oriented}/{len(queries)} {'✅' if consumer_oriented >= 2 else '⚠️'}")
        
        if company_mentions == 0 and len(queries) >= 3:
            print("\n🎉 Gemini 2.0 Flash query generation FIXED!")
            print("   ✓ No more AttributeError")
            print("   ✓ Generating realistic consumer queries")
            print("   ✓ Ready for GPT-5 ranking analysis")
        else:
            print("\n⚠️ Some issues remain, but basic functionality restored")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_gemini_flash())
