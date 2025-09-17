import json
import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI

class PromptGenerator:
    """Generates semantically similar prompts for GEO analysis"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def generate_semantic_queries(
        self,
        company_name: str,
        industry_context: str = "",
        num_queries: int = 100
    ) -> List[str]:
        """
        Generate semantically similar queries for testing company visibility
        
        Args:
            company_name: The company to generate queries for
            industry_context: Additional context about the company's industry
            num_queries: Number of queries to generate
            
        Returns:
            List of generated query strings
        """
        
        # Generate queries in batches for better organization
        batch_size = 20
        all_queries = []
        
        for i in range(0, num_queries, batch_size):
            remaining = min(batch_size, num_queries - i)
            batch_queries = await self._generate_query_batch(
                company_name=company_name,
                industry_context=industry_context,
                num_queries=remaining,
                batch_number=i // batch_size + 1
            )
            all_queries.extend(batch_queries)
        
        # Ensure we have exactly the requested number
        return all_queries[:num_queries]
    
    async def _generate_query_batch(
        self,
        company_name: str,
        industry_context: str,
        num_queries: int,
        batch_number: int
    ) -> List[str]:
        """Generate a batch of semantic queries"""
        
        # Create context-aware prompt for query generation
        prompt = self._create_generation_prompt(
            company_name=company_name,
            industry_context=industry_context,
            num_queries=num_queries,
            batch_number=batch_number
        )
        
        try:
            # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
            # do not change this unless explicitly requested by the user
            response = await self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in search query generation and SEO. "
                        "Generate diverse, realistic search queries that users might ask "
                        "when looking for products or services in the given industry. "
                        "Make queries natural and varied in style, length, and approach."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.8,
                max_tokens=2000
            )
            
            result = json.loads(response.choices[0].message.content or "{}")
            return result.get('queries', [])
            
        except Exception as e:
            # Fallback to template-based generation if API fails
            return self._generate_fallback_queries(company_name, industry_context, num_queries)
    
    def _create_generation_prompt(
        self,
        company_name: str,
        industry_context: str,
        num_queries: int,
        batch_number: int
    ) -> str:
        """Create a prompt for generating semantic queries"""
        
        context_info = f" in the {industry_context} industry" if industry_context else ""
        
        query_types = [
            "comparison queries (vs competitors)",
            "best of lists (top 10, best 2025)",
            "problem-solving queries (solutions, alternatives)",
            "review and recommendation queries",
            "buying guide queries",
            "feature-specific queries",
            "price and value queries",
            "use case specific queries",
            "trend and future-focused queries",
            "local and regional queries"
        ]
        
        # Rotate query types for different batches
        primary_type = query_types[batch_number % len(query_types)]
        
        return f"""
Generate {num_queries} diverse, realistic search queries that potential customers might use when researching companies{context_info}. 

Focus primarily on: {primary_type}

Company context: {company_name}{context_info}

Requirements:
1. Queries should be natural and varied (different lengths, styles, formality levels)
2. Include both broad industry queries and specific use-case queries
3. Mix informational, commercial, and navigational intent
4. Some queries should naturally favor established/popular brands
5. Include comparison and competitive analysis queries
6. Add seasonal, trending, and time-sensitive elements where relevant
7. Make queries realistic - what real users would actually search

Return the response in JSON format:
{{
    "queries": [
        "query 1 text",
        "query 2 text",
        ...
    ]
}}

Example types to include:
- "best [product category] 2025"
- "top 10 [industry] companies"
- "[specific use case] recommendations"
- "[product] vs [competitor] comparison"
- "most reliable [product type]"
- "affordable [product category] options"
- "[specific feature] in [product]"
- "reviews of [company/product]"
- "[industry] trends 2025"
- "sustainable [product category]"
"""
    
    def _generate_fallback_queries(
        self,
        company_name: str,
        industry_context: str,
        num_queries: int
    ) -> List[str]:
        """Generate fallback queries using templates if API fails"""
        
        # Extract industry keywords
        industry_keywords = self._extract_industry_keywords(company_name, industry_context)
        
        templates = [
            "best {industry} companies 2025",
            "top 10 {industry} brands",
            "most popular {product} options",
            "{industry} comparison guide",
            "affordable {product} recommendations",
            "premium {product} brands",
            "sustainable {industry} companies",
            "innovative {industry} solutions",
            "{product} reviews and ratings",
            "trusted {industry} providers",
            "leading {product} manufacturers",
            "{industry} market leaders",
            "quality {product} options",
            "reliable {industry} services",
            "expert {product} recommendations"
        ]
        
        queries = []
        for i in range(num_queries):
            template = templates[i % len(templates)]
            
            # Fill in template with industry keywords
            query = template.format(
                industry=industry_keywords.get('industry', industry_context or 'business'),
                product=industry_keywords.get('product', industry_context or 'service')
            )
            
            queries.append(query)
        
        return queries
    
    def _extract_industry_keywords(self, company_name: str, industry_context: str) -> Dict[str, str]:
        """Extract relevant keywords from company name and context"""
        
        # Simple keyword extraction logic
        keywords = {
            'industry': industry_context or 'business',
            'product': industry_context or 'service'
        }
        
        # Common industry mappings
        industry_mappings = {
            'lip balm': {'industry': 'cosmetics', 'product': 'lip care'},
            'lipbalm': {'industry': 'cosmetics', 'product': 'lip care'},
            'electric vehicle': {'industry': 'automotive', 'product': 'electric car'},
            'ev': {'industry': 'automotive', 'product': 'electric vehicle'},
            'smartphone': {'industry': 'technology', 'product': 'mobile phone'},
            'software': {'industry': 'technology', 'product': 'software solution'},
            'cloud': {'industry': 'technology', 'product': 'cloud service'},
            'food': {'industry': 'food', 'product': 'food product'},
            'restaurant': {'industry': 'hospitality', 'product': 'dining'},
            'clothing': {'industry': 'fashion', 'product': 'apparel'},
            'shoes': {'industry': 'fashion', 'product': 'footwear'}
        }
        
        # Check for mappings in company name or context
        search_text = f"{company_name} {industry_context}".lower()
        for key, mapping in industry_mappings.items():
            if key in search_text:
                keywords.update(mapping)
                break
        
        return keywords
