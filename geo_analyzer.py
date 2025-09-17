import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import re

from prompt_generator import PromptGenerator
from citation_tracker import CitationTracker

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
    
    def __init__(self, api_key: str, max_concurrent: int = 10):
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.prompt_generator = PromptGenerator(api_key)
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
        
        if progress_callback:
            progress_callback(0, num_queries + 1, "Generating semantic queries...")
        
        # Generate semantic queries
        queries = await self.prompt_generator.generate_semantic_queries(
            company_name=company_name,
            industry_context=industry_context,
            num_queries=num_queries
        )
        
        if progress_callback:
            progress_callback(1, num_queries + 1, f"Generated {len(queries)} queries. Starting analysis...")
        
        # Execute queries concurrently
        query_results = await self._execute_queries_concurrent(
            queries=queries,
            company_name=company_name,
            progress_callback=progress_callback,
            start_offset=1,
            total_steps=num_queries + 1
        )
        
        # Analyze results
        summary = self._analyze_results(query_results, company_name)
        
        total_time = time.time() - start_time
        
        if progress_callback:
            progress_callback(num_queries + 1, num_queries + 1, f"Analysis completed in {total_time:.1f}s")
        
        return {
            'company_name': company_name,
            'industry_context': industry_context,
            'analysis_timestamp': time.time(),
            'total_execution_time': total_time,
            'query_results': [self._query_result_to_dict(qr) for qr in query_results],
            'summary': summary
        }
    
    async def _execute_queries_concurrent(
        self,
        queries: List[str],
        company_name: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        start_offset: int = 0,
        total_steps: int = 100
    ) -> List[QueryResult]:
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
                        response, company_name
                    )
                    
                    execution_time = time.time() - start_time
                    
                    result = QueryResult(
                        query=query,
                        response=response,
                        cited=citation_analysis['cited'],
                        context=citation_analysis.get('context'),
                        position=citation_analysis.get('position'),
                        execution_time=execution_time,
                        competitors=citation_analysis.get('competitors', [])
                    )
                    
                    completed += 1
                    if progress_callback:
                        progress_callback(
                            start_offset + completed,
                            total_steps,
                            f"Completed query {completed}/{len(queries)}"
                        )
                    
                    return result
                    
                except Exception as e:
                    completed += 1
                    if progress_callback:
                        progress_callback(
                            start_offset + completed,
                            total_steps,
                            f"Error in query {completed}/{len(queries)}: {str(e)[:50]}"
                        )
                    
                    return QueryResult(
                        query=query,
                        response=f"Error: {str(e)}",
                        cited=False,
                        execution_time=time.time() - start_time
                    )
        
        # Execute all queries concurrently
        tasks = [execute_single_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = [r for r in results if isinstance(r, QueryResult)]
        
        return valid_results
    
    async def _make_web_research_query(self, query: str) -> str:
        """
        Make a web research query using OpenAI's ChatGPT
        """
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=self.api_key)
        
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        try:
            response = await client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant conducting web research. "
                        "Provide comprehensive, accurate answers based on current information. "
                        "When listing companies, products, or recommendations, be specific and "
                        "include relevant details like rankings, features, or market position."
                    },
                    {
                        "role": "user",
                        "content": f"Research and answer this query comprehensively: {query}"
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _analyze_results(self, query_results: List[QueryResult], company_name: str) -> Dict[str, Any]:
        """Analyze the query results and generate summary statistics"""
        
        total_queries = len(query_results)
        cited_results = [qr for qr in query_results if qr.cited]
        total_citations = len(cited_results)
        
        # Calculate average position
        positions = [qr.position for qr in cited_results if qr.position is not None]
        average_position = sum(positions) / len(positions) if positions else 0
        
        # Aggregate competitors
        all_competitors = []
        for qr in query_results:
            if qr.competitors:
                all_competitors.extend(qr.competitors)
        
        # Count competitor mentions
        competitor_counts = {}
        for competitor in all_competitors:
            competitor_counts[competitor] = competitor_counts.get(competitor, 0) + 1
        
        # Convert to list of dictionaries
        competitors = [
            {
                'name': name,
                'mentions': count,
                'percentage': (count / total_queries) * 100
            }
            for name, count in sorted(competitor_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
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
    
    def _query_result_to_dict(self, query_result: QueryResult) -> Dict[str, Any]:
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
