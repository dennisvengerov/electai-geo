import asyncio
import json
import time
import os
from typing import List, Dict, Any, Optional, Callable
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from data_models import RawQueryResult, BatchAnalysisResult


class BatchAnalyzer:
    """Batch analyzer using Gemini for comprehensive query result analysis"""
    
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        
        # Configure the model with safety settings - using Gemini 2.0 Flash (latest model)
        self.model = genai.GenerativeModel(
            'gemini-2.5-pro',
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    
    async def analyze_batch(
        self,
        query_results: List[RawQueryResult],
        company_name: str,
        max_batch_tokens: int = 500000,  # 500k tokens per batch
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> BatchAnalysisResult:
        """
        Analyze multiple query results in batches using Gemini
        
        Args:
            query_results: List of raw query results to analyze
            company_name: Target company name to analyze for
            max_batch_size: Maximum number of queries per batch
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete batch analysis result
        """
        start_time = time.time()
        
        if not query_results:
            return BatchAnalysisResult(
                query_analyses=[],
                aggregate_analysis={'error': 'No query results to analyze'},
                analysis_timestamp=time.time(),
                total_execution_time=0.0
            )
        
        # Split into batches based on token count
        batches = self._create_token_based_batches(query_results, max_batch_tokens)
        all_analyses = []
        
        if progress_callback:
            progress_callback(0, len(batches), "Starting batch analysis...")
        
        for i, batch in enumerate(batches):
            if progress_callback:
                progress_callback(i, len(batches), f"Analyzing batch {i+1}/{len(batches)}...")
            
            try:
                print(f"🔍 Analyzing batch {i+1}/{len(batches)} ({len(batch)} queries)...")
                batch_result = await self._analyze_single_batch(batch, company_name)
                
                print(f"✅ Batch {i+1} analysis completed")
                print(f"   Query analyses: {len(batch_result.query_analyses)}")
                print(f"   Has error: {'error' in batch_result.aggregate_analysis}")
                
                all_analyses.extend(batch_result.query_analyses)
                
                # Small delay between batches to avoid rate limiting
                if i < len(batches) - 1:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error analyzing batch {i+1}/{len(batches)}")
                print(f"   Error type: {type(e).__name__}")
                print(f"   Error message: {error_msg}")
                
                import traceback
                print(f"   Full traceback:")
                traceback.print_exc()
                
                # Continue with other batches even if one fails
                continue
        
        # Aggregate results from all batches
        total_time = time.time() - start_time
        final_result = self._aggregate_all_analyses(all_analyses, company_name, total_time)
        
        if progress_callback:
            progress_callback(len(batches), len(batches), f"Batch analysis completed in {total_time:.1f}s")
        
        return final_result
    
    def _create_token_based_batches(self, query_results: List[RawQueryResult], max_batch_tokens: int) -> List[List[RawQueryResult]]:
        """Split query results into batches based on estimated token count"""
        batches = []
        current_batch = []
        current_tokens = 0
        
        for result in query_results:
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            query_tokens = len(result.query) // 4
            response_tokens = len(result.response) // 4
            result_tokens = query_tokens + response_tokens + 50  # Add overhead for formatting
            
            # If adding this result would exceed the limit, start a new batch
            if current_tokens + result_tokens > max_batch_tokens and current_batch:
                batches.append(current_batch)
                current_batch = [result]
                current_tokens = result_tokens
            else:
                current_batch.append(result)
                current_tokens += result_tokens
        
        # Add the last batch if it has content
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    async def _analyze_single_batch(
        self,
        batch: List[RawQueryResult],
        company_name: str
    ) -> BatchAnalysisResult:
        """Analyze a single batch using Gemini"""
        
        print(f"🔍 Starting Gemini 2.5 Pro batch analysis...")
        print(f"   Company: {company_name}")
        print(f"   Batch size: {len(batch)} queries")
        
        # Estimate token count
        total_chars = sum(len(r.query) + len(r.response) for r in batch)
        estimated_tokens = total_chars // 4
        print(f"   Estimated tokens: {estimated_tokens}")
        
        prompt = self._create_batch_analysis_prompt(batch, company_name)
        print(f"   Prompt length: {len(prompt)} characters")
        
        try:
            print(f"📡 Sending request to Gemini 2.5 Pro...")
            
            # Generate content with Gemini
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent analysis
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=100000,
                    response_mime_type="application/json"
                )
            )
            
            print(f"✅ Gemini 2.5 Pro response received")
            print(f"   Response length: {len(response.text) if response.text else 0} characters")
            
            if response.text:
                print(f"📝 Parsing Gemini response as JSON...")
                result = BatchAnalysisResult.from_json(response.text)
                print(f"✅ Successfully parsed {len(result.query_analyses)} query analyses")
                return result
            else:
                print(f"❌ Empty response from Gemini")
                return self._create_error_result("Empty response from Gemini")
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Gemini API call failed")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {error_msg}")
            
            # Check for specific quota errors
            if "quota" in error_msg.lower() or "429" in error_msg:
                print(f"💰 Gemini quota/rate limit issue")
                return self._create_error_result(f"Gemini API quota exceeded: {error_msg}")
            elif "403" in error_msg:
                print(f"🔑 Gemini API access denied")
                return self._create_error_result(f"Gemini API access denied: {error_msg}")
            elif "json" in error_msg.lower():
                print(f"📄 JSON parsing error")
                if response and hasattr(response, 'text'):
                    print(f"   Raw response preview: {response.text[:500]}...")
            else:
                print(f"🚨 General Gemini API error")
            
            import traceback
            print(f"   Full traceback:")
            traceback.print_exc()
            
            return self._create_error_result(f"Gemini API error: {error_msg}")
    
    def _create_batch_analysis_prompt(
        self,
        batch: List[RawQueryResult],
        company_name: str
    ) -> str:
        """Create comprehensive analysis prompt for Gemini"""
        
        queries_text = ""
        for i, result in enumerate(batch, 1):
            # Truncate very long responses to fit in context window
            response_text = result.response
            if len(response_text) > 3000:
                response_text = response_text[:3000] + "... [truncated]"
                
            queries_text += f"""
QUERY {i}: {result.query}
RESPONSE {i}: {response_text}
---
"""
        
        return f"""You are an expert analyst specializing in competitive intelligence, brand mention analysis, and ranking extraction from AI-generated content.

TARGET COMPANY: {company_name}

TASK: Analyze the following {len(batch)} query-response pairs that contain RANKED LISTS and competitive analysis. Extract precise ranking information, citations, and competitive intelligence.

{queries_text}

CRITICAL ANALYSIS REQUIREMENTS:

1. RANKING EXTRACTION: These responses contain numbered lists (1., 2., 3., etc.). Extract the EXACT position of {company_name} and ALL competitors from these rankings.

2. CITATION ANALYSIS: Determine if {company_name} is mentioned:
   - DIRECT: Explicit company name mention
   - INDIRECT: Brand/product names that represent the company  
   - IMPLIED: Descriptive references (e.g., "the Cupertino company" = Apple)
   - NONE: No mention found

3. POSITION TRACKING: When companies appear in numbered lists:
   - Extract the exact numerical position (1, 2, 3, etc.)
   - Note if it's a "top 10", "best 5", "leading brands" type of list
   - Capture the ranking context and criteria

4. COMPETITOR MAPPING: For each numbered list, extract ALL companies and their positions:
   - Company at position 1, Company at position 2, etc.
   - Include the ranking criteria (best, top, recommended, etc.)
   - NORMALIZE company names: treat "Tesla" and "Tesla, Inc." as the same company
   - Distinguish between actual companies vs products/services/generic terms

5. CONTEXT EXTRACTION: Capture the exact text mentioning each company

6. COMPANY NORMALIZATION: Apply these rules when identifying companies:
   - Remove corporate suffixes: "Inc.", "Corp.", "LLC", "Ltd.", "Co.", "Corporation", "Company"
   - Handle special cases: "Alphabet Inc." → "Google", "Meta Platforms" → "Meta", "X Corp" → "X"
   - Treat variations as the same company: "Tesla" = "Tesla, Inc." = "Tesla Motors"
   - Focus on actual companies, not products or generic terms

7. IMPLICIT POSITION TRACKING: When no explicit numbering exists:
   - Assign positions based on mention order in the response
   - First company mentioned = position 1, second = position 2, etc.
   - Only count actual company names, not generic terms

Return JSON in this EXACT format:
{{
    "query_analyses": [
        {{
            "query_id": 1,
            "query_text": "...",
            "target_company_cited": true,
            "citation_context": "exact text where company was mentioned",
            "target_company_position": 3,
            "ranking_type": "top 10 best brands",
            "all_companies_mentioned": [
                {{"name": "Company Name", "position": 1, "context": "Company Name is the leader in...", "confidence": 0.9}},
                {{"name": "Target Company", "position": 3, "context": "Target Company offers...", "confidence": 0.95}},
                {{"name": "Another Company", "position": 5, "context": "Another Company provides...", "confidence": 0.85}}
            ],
            "mention_type": "direct",
            "confidence": 0.95,
            "has_numbered_ranking": true,
            "total_companies_in_ranking": 10
        }}
    ],
    "aggregate_analysis": {{
        "total_citations": 5,
        "citation_rate": 0.25,
        "average_position": 2.3,
        "unique_competitors": [
            {{"name": "Competitor A", "total_mentions": 8, "average_position": 1.2}},
            {{"name": "Competitor B", "total_mentions": 6, "average_position": 2.1}}
        ],
        "competitive_landscape": {{
            "market_leaders": ["Company A", "Company B"],
            "frequent_comparisons": ["Company C vs Company D"],
            "emerging_players": ["Company E"]
        }},
        "analysis_confidence": 0.92
    }}
}}

IMPORTANT GUIDELINES:
- Be extremely thorough in competitor identification
- Look for direct company names, brand names, product names that represent companies
- Consider implied references (e.g., "the Cupertino company" = Apple)
- Include abbreviations and acronyms
- When {company_name} appears in a numbered list, capture its exact position
- For mention_type, use: "direct" (explicit name), "indirect" (brand/product), "implied" (description), "none"
- Set confidence based on clarity of mention/identification
- If a query response contains no clear answer or is an error, mark target_company_cited as false
- Be conservative with citations - only mark as cited if there's clear evidence

Ensure the JSON is properly formatted and complete."""
    
    def _create_error_result(self, error_message: str) -> BatchAnalysisResult:
        """Create a result object for error cases"""
        return BatchAnalysisResult(
            query_analyses=[],
            aggregate_analysis={'error': error_message},
            analysis_timestamp=time.time(),
            total_execution_time=0.0
        )
    
    def _aggregate_all_analyses(
        self,
        all_analyses: List,
        company_name: str,
        total_time: float
    ) -> BatchAnalysisResult:
        """Aggregate analyses from multiple batches"""
        
        if not all_analyses:
            return self._create_error_result("No successful analyses to aggregate")
        
        # Combine all analyses
        total_queries = len(all_analyses)
        cited_analyses = [qa for qa in all_analyses if qa.target_company_cited]
        total_citations = len(cited_analyses)
        
        # Calculate average position
        positions = [qa.target_company_position for qa in cited_analyses 
                    if qa.target_company_position is not None]
        average_position = sum(positions) / len(positions) if positions else 0
        
        # Aggregate competitors
        competitor_mentions = {}
        competitor_positions = {}
        
        for qa in all_analyses:
            for mention in qa.all_companies_mentioned:
                name = mention.name
                if name != company_name:  # Exclude target company
                    if name not in competitor_mentions:
                        competitor_mentions[name] = 0
                        competitor_positions[name] = []
                    
                    competitor_mentions[name] += 1
                    if mention.position is not None:
                        competitor_positions[name].append(mention.position)
        
        # Create competitor summary
        unique_competitors = []
        for name, mentions in competitor_mentions.items():
            positions = competitor_positions.get(name, [])
            avg_pos = sum(positions) / len(positions) if positions else 0
            
            unique_competitors.append({
                'name': name,
                'total_mentions': mentions,
                'average_position': avg_pos
            })
        
        # Sort by mention frequency
        unique_competitors.sort(key=lambda x: x['total_mentions'], reverse=True)
        
        # Create aggregate analysis
        aggregate_analysis = {
            'total_citations': total_citations,
            'citation_rate': total_citations / total_queries if total_queries > 0 else 0,
            'average_position': average_position,
            'unique_competitors': unique_competitors[:20],  # Top 20 competitors
            'analysis_confidence': sum(qa.confidence for qa in all_analyses) / len(all_analyses) if all_analyses else 0
        }
        
        return BatchAnalysisResult(
            query_analyses=all_analyses,
            aggregate_analysis=aggregate_analysis,
            analysis_timestamp=time.time(),
            total_execution_time=total_time
        )
