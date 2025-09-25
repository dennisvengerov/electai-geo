import json
import asyncio
from typing import List, Dict, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class PromptGenerator:
    """Generates semantically similar prompts for GEO analysis using Gemini 2.0 Flash"""
    
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        
        # Configure Gemini 2.0 Flash for query generation
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    
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
            print(f"🔍 Starting Gemini 2.0 Flash query generation...")
            print(f"   Company: {company_name}")
            print(f"   Industry: {industry_context}")
            print(f"   Requested queries: {num_queries}")
            print(f"   Batch number: {batch_number}")
            
            # Use Gemini 2.0 Flash for query generation
            print(f"📡 Sending request to Gemini 2.0 Flash...")
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.8,
                    top_p=0.9,
                    top_k=40,
                    max_output_tokens=2000,
                    response_mime_type="application/json"
                )
            )
            
            print(f"✅ Gemini 2.0 Flash response received")
            print(f"   Response text length: {len(response.text) if response.text else 0}")
            
            if response.text:
                print(f"📝 Parsing JSON response...")
                result = json.loads(response.text)
                queries = result.get('queries', [])
                print(f"✅ Successfully generated {len(queries)} queries")
                
                # Log first few queries for verification
                for i, query in enumerate(queries[:3], 1):
                    print(f"   Query {i}: {query}")
                
                return queries
            else:
                print(f"⚠️ Empty response from Gemini, using fallback")
                # Fallback to template-based generation if no response
                return self._generate_fallback_queries(company_name, industry_context, num_queries)
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Gemini query generation failed")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {error_msg}")
            
            # Check for specific error types
            if "quota" in error_msg.lower() or "429" in error_msg:
                print(f"💰 Gemini quota/rate limit issue")
            elif "api_key" in error_msg.lower() or "401" in error_msg:
                print(f"🔑 Gemini API key issue")
            elif "json" in error_msg.lower():
                print(f"📄 JSON parsing error - response may not be valid JSON")
                if hasattr(e, 'response') and e.response:
                    print(f"   Raw response: {e.response}")
            
            import traceback
            print(f"   Full traceback:")
            traceback.print_exc()
            
            print(f"🔄 Falling back to template-based generation...")
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
