#!/usr/bin/env python3
"""
Test script for the new batch analysis functionality
"""

import asyncio
import os
import json
from data_models import RawQueryResult
from batch_analyzer import BatchAnalyzer


async def test_batch_analyzer():
    """Test the batch analyzer with sample data"""
    
    # Check for Gemini API key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("Please set your Gemini API key to test batch analysis:")
        print("export GEMINI_API_KEY='your-key-here'")
        return
    
    print("🚀 Testing Batch Analysis with Gemini...")
    
    # Create sample raw query results
    sample_results = [
        RawQueryResult(
            query="best lip balm brands 2025",
            response="Here are the top lip balm brands for 2025: 1. Burt's Bees - Known for natural ingredients and classic formula. 2. ChapStick - The most recognizable brand with reliable moisture. 3. Carmex - Popular for medicated lip balm. 4. EOS - Trendy spherical design and organic options. 5. Aquaphor - Healing ointment great for severely chapped lips.",
            execution_time=2.3,
            timestamp=1640995200.0
        ),
        RawQueryResult(
            query="natural lip care products comparison",
            response="When comparing natural lip care products, several brands stand out. Burt's Bees leads with their beeswax-based formula that has been trusted for decades. Badger Classic Lip Balm offers organic certification. Dr. Bronner's provides a naked lip balm with minimal ingredients. Alba Botanica uses plant-based ingredients for their lip care line.",
            execution_time=1.8,
            timestamp=1640995260.0
        ),
        RawQueryResult(
            query="affordable lip moisturizer recommendations",
            response="For budget-friendly lip moisturizers, ChapStick Classic remains the most affordable option at under $2. Blistex and Carmex are also great budget choices. While slightly more expensive, Burt's Bees offers good value for a natural option. Vaseline Lip Therapy is another economical choice for basic moisture.",
            execution_time=2.1,
            timestamp=1640995320.0
        )
    ]
    
    # Initialize batch analyzer
    analyzer = BatchAnalyzer(gemini_key)
    
    # Test company
    company_name = "Burt's Bees"
    
    print(f"📊 Analyzing visibility for: {company_name}")
    print(f"🔍 Processing {len(sample_results)} sample queries...")
    
    try:
        # Run batch analysis
        result = await analyzer.analyze_batch(
            query_results=sample_results,
            company_name=company_name,
            max_batch_size=10
        )
        
        print("\n✅ Batch analysis completed!")
        print(f"📈 Analysis confidence: {result.aggregate_analysis.get('analysis_confidence', 0):.2%}")
        print(f"🎯 Total citations: {result.aggregate_analysis.get('total_citations', 0)}")
        print(f"📊 Citation rate: {result.aggregate_analysis.get('citation_rate', 0):.2%}")
        
        # Display query analyses
        print("\n📋 Query Analysis Results:")
        for i, qa in enumerate(result.query_analyses, 1):
            print(f"\n  Query {i}: {qa.query_text[:60]}...")
            print(f"    ✓ Cited: {qa.target_company_cited}")
            print(f"    📍 Position: {qa.target_company_position}")
            print(f"    🎭 Mention Type: {qa.mention_type}")
            print(f"    🎯 Confidence: {qa.confidence:.2f}")
            
            if qa.all_companies_mentioned:
                print(f"    🏢 Companies found: {[m.name for m in qa.all_companies_mentioned]}")
        
        # Display competitors
        competitors = result.aggregate_analysis.get('unique_competitors', [])
        if competitors:
            print(f"\n🏆 Top Competitors Found:")
            for comp in competitors[:5]:
                print(f"  • {comp['name']}: {comp['total_mentions']} mentions")
        
        # Test legacy format conversion
        legacy_format = result.to_legacy_format()
        print(f"\n🔄 Legacy format conversion: ✅ {len(legacy_format['query_results'])} results")
        
        print("\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_batch_analyzer())
