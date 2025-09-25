import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import re

from prompt_generator import PromptGenerator
from batch_analyzer import BatchAnalyzer
from data_models import RawQueryResult, BatchAnalysisResult


@dataclass
class QueryResult:
    query: str
    response: str
    cited: bool
    context: Optional[str] = None
    position: Optional[int] = None
    execution_time: float = 0.0
    competitors: Optional[List[str]] = None


class GEOAnalyzer:
    """Main class for conducting GEO (Generative Engine Optimization) analysis"""

    def __init__(self, openai_api_key: str, gemini_api_key: str = None, max_concurrent: int = 10, use_batch_analysis: bool = True):
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.max_concurrent = max_concurrent
        self.use_batch_analysis = use_batch_analysis and gemini_api_key is not None
        
        # Query generation always uses Gemini 2.0 Flash
        if not gemini_api_key:
            raise ValueError("Gemini API key is required for query generation")
        self.prompt_generator = PromptGenerator(gemini_api_key)
        
        if self.use_batch_analysis:
            self.batch_analyzer = BatchAnalyzer(gemini_api_key)
        else:
            # Fallback to old method if Gemini API key not provided
            from citation_tracker import CitationTracker
            self.citation_tracker = CitationTracker()

    async def analyze_company_visibility(
        self,
        company_name: str,
        industry_context: str = "",
        num_queries: int = 100,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Analyze company visibility across multiple semantic queries
        
        Args:
            company_name: Name of the company to analyze
            industry_context: Additional context about the company's industry
            num_queries: Number of queries to generate and test
            progress_callback: Callback function for progress updates
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        
        print(f"\n🚀 Starting GEO Analysis Pipeline")
        print(f"   Company: {company_name}")
        print(f"   Industry: {industry_context}")
        print(f"   Queries: {num_queries}")
        print(f"   Batch Analysis: {self.use_batch_analysis}")
        print(f"   Max Concurrent: {self.max_concurrent}")

        if progress_callback:
            progress_callback(0, num_queries + 2, "Generating semantic queries...")

        # Generate semantic queries
        print(f"\n📝 Step 1: Query Generation (Gemini 2.0 Flash)")
        queries = await self.prompt_generator.generate_semantic_queries(
            company_name=company_name,
            industry_context=industry_context,
            num_queries=num_queries)

        print(f"✅ Generated {len(queries)} queries")
        if progress_callback:
            progress_callback(1, num_queries + 2, f"Generated {len(queries)} queries. Collecting responses...")

        # Collect raw responses
        print(f"\n🏆 Step 2: Ranking Generation (GPT-5)")
        raw_results = await self._collect_raw_responses(
            queries=queries,
            progress_callback=progress_callback,
            start_offset=1,
            total_steps=num_queries + 2
        )
        
        print(f"✅ Collected {len(raw_results)} responses")
        
        # Log response quality
        successful_responses = [r for r in raw_results if not r.response.startswith("Error:")]
        error_responses = len(raw_results) - len(successful_responses)
        print(f"   Successful: {len(successful_responses)}")
        print(f"   Errors: {error_responses}")
        
        if successful_responses:
            avg_length = sum(len(r.response) for r in successful_responses) / len(successful_responses)
            print(f"   Avg response length: {avg_length:.0f} chars")

        if progress_callback:
            progress_callback(num_queries + 1, num_queries + 2, "Analyzing results...")

        # Choose analysis method
        print(f"\n📊 Step 3: Analysis Selection")
        if self.use_batch_analysis:
            print(f"   Using: Gemini 2.5 Pro Batch Analysis")
            try:
                # Use new batch analysis with Gemini
                batch_result = await self.batch_analyzer.analyze_batch(
                    raw_results, company_name, progress_callback=progress_callback
                )
                
                print(f"✅ Batch analysis completed")
                print(f"   Query analyses: {len(batch_result.query_analyses)}")
                print(f"   Has errors: {'error' in batch_result.aggregate_analysis}")
                
                # Check if batch analysis was successful
                if batch_result.query_analyses and not batch_result.aggregate_analysis.get('error'):
                    print(f"✅ Batch analysis successful, converting to legacy format")
                    final_results = batch_result.to_legacy_format()
                    final_results.update({
                        'company_name': company_name,
                        'industry_context': industry_context,
                        'analysis_method': 'batch_gemini'
                    })
                else:
                    # Batch analysis failed, fall back to legacy
                    error_msg = batch_result.aggregate_analysis.get('error', 'Unknown error')
                    print(f"❌ Batch analysis failed: {error_msg}")
                    if progress_callback:
                        progress_callback(num_queries + 1, num_queries + 2, "Batch analysis failed, using legacy method...")
                    raise Exception(f"Batch analysis returned empty results: {error_msg}")
                    
            except Exception as e:
                print(f"Batch analysis failed: {str(e)}")
                if progress_callback:
                    progress_callback(num_queries + 1, num_queries + 2, "Falling back to legacy analysis...")
                
                # Fallback to legacy analysis
                query_results = await self._execute_queries_concurrent_legacy(
                    queries=queries,
                    company_name=company_name,
                    progress_callback=progress_callback,
                    start_offset=1,
                    total_steps=num_queries + 2
                )
                summary = self._analyze_results(query_results, company_name)
                final_results = {
                    'company_name': company_name,
                    'industry_context': industry_context,
                    'analysis_timestamp': time.time(),
                    'total_execution_time': time.time() - start_time,
                    'query_results': [self._query_result_to_dict(qr) for qr in query_results],
                    'summary': summary,
                    'analysis_method': 'legacy_fallback',
                    'fallback_reason': str(e)
                }
        else:
            # Use legacy method directly
            query_results = await self._execute_queries_concurrent_legacy(
                queries=queries,
                company_name=company_name,
                progress_callback=progress_callback,
                start_offset=1,
                total_steps=num_queries + 2
            )
            summary = self._analyze_results(query_results, company_name)
            final_results = {
                'company_name': company_name,
                'industry_context': industry_context,
                'analysis_timestamp': time.time(),
                'total_execution_time': time.time() - start_time,
                'query_results': [self._query_result_to_dict(qr) for qr in query_results],
                'summary': summary,
                'analysis_method': 'legacy'
            }

        if progress_callback:
            total_time = time.time() - start_time
            progress_callback(num_queries + 2, num_queries + 2, f"Analysis completed in {total_time:.1f}s")

        return final_results

    async def _collect_raw_responses(
        self,
        queries: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        start_offset: int = 0,
        total_steps: int = 100
    ) -> List[RawQueryResult]:
        """Collect raw responses without analysis"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed = 0
        results = []

        async def execute_single_query(query: str) -> RawQueryResult:
            nonlocal completed

            async with semaphore:
                start_time = time.time()

                try:
                    print(f"🔄 Executing GPT-5 query {completed+1}/{len(queries)}: {query[:60]}...")
                    
                    # Make OpenAI API call for ranking generation
                    response = await self._make_ranking_query(query)
                    execution_time = time.time() - start_time

                    print(f"✅ GPT-5 response received ({len(response)} chars) in {execution_time:.1f}s")
                    
                    # Log a preview of the response
                    response_preview = response[:200] + "..." if len(response) > 200 else response
                    print(f"📝 Response preview: {response_preview}")

                    result = RawQueryResult(
                        query=query,
                        response=response,
                        execution_time=execution_time,
                        timestamp=time.time()
                    )

                    completed += 1
                    if progress_callback:
                        progress_callback(
                            start_offset + completed, total_steps,
                            f"Collected response {completed}/{len(queries)}")

                    return result

                except Exception as e:
                    completed += 1
                    error_msg = str(e)
                    print(f"❌ ERROR in GPT-5 query {completed}/{len(queries)}")
                    print(f"   Query: {query}")
                    print(f"   Error: {error_msg}")
                    print(f"   Error type: {type(e).__name__}")
                    
                    import traceback
                    print(f"   Full traceback:")
                    traceback.print_exc()
                    
                    if progress_callback:
                        progress_callback(
                            start_offset + completed, total_steps,
                            f"Error in query {completed}/{len(queries)}: {str(e)[:50]}"
                        )

                    return RawQueryResult(
                        query=query,
                        response=f"Error: {str(e)}",
                        execution_time=time.time() - start_time,
                        timestamp=time.time()
                    )

        # Execute all queries concurrently
        tasks = [execute_single_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return valid results
        valid_results = [r for r in results if isinstance(r, RawQueryResult)]

        return valid_results

    async def _execute_queries_concurrent_legacy(
            self,
            queries: List[str],
            company_name: str,
            progress_callback: Optional[Callable[[int, int, str],
                                                 None]] = None,
            start_offset: int = 0,
            total_steps: int = 100) -> List[QueryResult]:
        """Execute multiple queries concurrently with rate limiting"""

        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed = 0
        results = []

        async def execute_single_query(query: str) -> QueryResult:
            nonlocal completed

            async with semaphore:
                start_time = time.time()

                try:
                    # Make OpenAI API call for web research
                    response = await self._make_web_research_query(query)

                    # Analyze citation
                    citation_analysis = self.citation_tracker.analyze_citation(
                        response, company_name)

                    execution_time = time.time() - start_time

                    result = QueryResult(
                        query=query,
                        response=response,
                        cited=citation_analysis['cited'],
                        context=citation_analysis.get('context'),
                        position=citation_analysis.get('position'),
                        execution_time=execution_time,
                        competitors=citation_analysis.get('competitors', []))

                    completed += 1
                    if progress_callback:
                        progress_callback(
                            start_offset + completed, total_steps,
                            f"Completed query {completed}/{len(queries)}")

                    return result

                except Exception as e:
                    completed += 1
                    if progress_callback:
                        progress_callback(
                            start_offset + completed, total_steps,
                            f"Error in query {completed}/{len(queries)}: {str(e)[:50]}"
                        )

                    return QueryResult(query=query,
                                       response=f"Error: {str(e)}",
                                       cited=False,
                                       execution_time=time.time() - start_time)

        # Execute all queries concurrently
        tasks = [execute_single_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return valid results
        valid_results = [r for r in results if isinstance(r, QueryResult)]

        return valid_results

    async def _make_ranking_query(self, query: str) -> str:
        """
        Make a ranking query using OpenAI's GPT-5 to generate actual ranked lists
        """
        print(f"🔍 Starting GPT-5 ranking query...")
        print(f"   API Key present: {bool(self.openai_api_key)}")
        print(f"   Query: {query}")
        
        from openai import AsyncOpenAI

        try:
            client = AsyncOpenAI(api_key=self.openai_api_key)
            print(f"✅ OpenAI client initialized successfully")

            # Try GPT-5 first, fallback to GPT-4o if not available
            model_to_use = "gpt-5"
            print(f"📡 Sending request to {model_to_use}...")
            
            try:
                response = await client.chat.completions.create(
                    model=model_to_use,
                    messages=[{
                        "role": "system",
                        "content": (
                            "You are an expert analyst creating comprehensive ranked lists and recommendations. "
                            "ALWAYS provide specific rankings with numbered lists when answering queries. "
                            "Include company names, specific positions, and detailed explanations. "
                            "Format your responses with clear numbered rankings (1., 2., 3., etc.) when relevant. "
                            "Be specific about market positions, features, and competitive advantages. "
                            "Provide comprehensive analysis that would help users make informed decisions."
                        )
                    }, {
                        "role": "user",
                        "content": (
                            f"Provide a comprehensive ranked response to this query with specific numbered rankings where appropriate: {query}\n\n"
                            "Make sure to:\n"
                            "- Include numbered rankings (1., 2., 3., etc.) when listing companies/products\n"
                            "- Mention specific company names and their market positions\n"
                            "- Provide detailed explanations for each ranking\n"
                            "- Include competitive analysis and comparisons\n"
                            "- Be comprehensive and authoritative in your response"
                        )
                    }],
                    temperature=0.7,
                    max_tokens=1500
                )

                response_content = response.choices[0].message.content or ""
                print(f"✅ {model_to_use} API call successful")
                print(f"   Response length: {len(response_content)} characters")
                print(f"   Usage: {response.usage}")
                
                return response_content
                
            except Exception as model_error:
                if "model" in str(model_error).lower() or "404" in str(model_error):
                    print(f"⚠️ {model_to_use} not available, trying GPT-4o...")
                    model_to_use = "gpt-4o"
                    
                    response = await client.chat.completions.create(
                        model=model_to_use,
                        messages=[{
                            "role": "system",
                            "content": (
                                "You are an expert analyst creating comprehensive ranked lists and recommendations. "
                                "ALWAYS provide specific rankings with numbered lists when answering queries. "
                                "Include company names, specific positions, and detailed explanations. "
                                "Format your responses with clear numbered rankings (1., 2., 3., etc.) when relevant. "
                                "Be specific about market positions, features, and competitive advantages. "
                                "Provide comprehensive analysis that would help users make informed decisions."
                            )
                        }, {
                            "role": "user",
                            "content": (
                                f"Provide a comprehensive ranked response to this query with specific numbered rankings where appropriate: {query}\n\n"
                                "Make sure to:\n"
                                "- Include numbered rankings (1., 2., 3., etc.) when listing companies/products\n"
                                "- Mention specific company names and their market positions\n"
                                "- Provide detailed explanations for each ranking\n"
                                "- Include competitive analysis and comparisons\n"
                                "- Be comprehensive and authoritative in your response"
                            )
                        }],
                        temperature=0.7,
                        max_tokens=1500
                    )

                    response_content = response.choices[0].message.content or ""
                    print(f"✅ {model_to_use} API call successful (fallback)")
                    print(f"   Response length: {len(response_content)} characters")
                    print(f"   Usage: {response.usage}")
                    
                    return response_content
                else:
                    raise model_error

        except Exception as e:
            error_msg = str(e)
            print(f"❌ GPT-5 API call failed")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {error_msg}")
            print(f"   API key length: {len(self.openai_api_key) if self.openai_api_key else 0}")
            
            # Check for specific error types
            if "api_key" in error_msg.lower() or "401" in error_msg:
                print(f"🔑 API Key issue detected")
            elif "quota" in error_msg.lower() or "429" in error_msg:
                print(f"💰 Quota/rate limit issue detected")
            elif "model" in error_msg.lower() or "404" in error_msg:
                print(f"🤖 Model issue detected - GPT-5 may not be available")
            
            import traceback
            print(f"   Full traceback:")
            traceback.print_exc()
            
            raise Exception(f"OpenAI API error: {str(e)}")

    def _analyze_results(self, query_results: List[QueryResult], company_name: str) -> Dict[str, Any]:
        """Analyze the query results and generate summary statistics, including competitor positions"""
        total_queries = len(query_results)
        cited_results = [qr for qr in query_results if qr.cited]
        total_citations = len(cited_results)
        # Calculate average position for target company
        positions = [qr.position for qr in cited_results if qr.position is not None]
        average_position = sum(positions) / len(positions) if positions else 0
        # Aggregate competitors and their positions
        all_competitors = []
        competitor_positions_map = {}
        for qr in query_results:
            if qr.competitors:
                all_competitors.extend(qr.competitors)
            # New: aggregate competitor positions
            competitor_positions = getattr(qr, 'competitor_positions', None)
            if competitor_positions:
                for name, pos in competitor_positions.items():
                    if name not in competitor_positions_map:
                        competitor_positions_map[name] = []
                    competitor_positions_map[name].append(pos)
        # Count competitor mentions
        competitor_counts = {}
        for competitor in all_competitors:
            competitor_counts[competitor] = competitor_counts.get(competitor, 0) + 1
        # Build competitor analytics with position info
        competitors = []
        for name, count in sorted(competitor_counts.items(), key=lambda x: x[1], reverse=True):
            pos_list = competitor_positions_map.get(name, [])
            avg_pos = sum(pos_list) / len(pos_list) if pos_list else 0
            competitors.append({
                'name': name,
                'mentions': count,
                'percentage': (count / total_queries) * 100,
                'avg_position': avg_pos,
                'positions': pos_list
            })
        # Calculate execution statistics
        execution_times = [qr.execution_time for qr in query_results if qr.execution_time > 0]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        return {
            'total_queries': total_queries,
            'total_citations': total_citations,
            'citation_rate': (total_citations / total_queries) * 100 if total_queries > 0 else 0,
            'average_position': average_position,
            'competitors': competitors,
            'avg_execution_time': avg_execution_time,
            'successful_queries': len([qr for qr in query_results if not qr.response.startswith('Error:')])
        }

    def _query_result_to_dict(self,
                              query_result: QueryResult) -> Dict[str, Any]:
        """Convert QueryResult dataclass to dictionary for JSON serialization"""
        return {
            'query': query_result.query,
            'response': query_result.response,
            'cited': query_result.cited,
            'context': query_result.context,
            'position': query_result.position,
            'execution_time': query_result.execution_time,
            'competitors': query_result.competitors or []
        }
