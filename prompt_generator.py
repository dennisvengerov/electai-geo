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
        company_name: str,  # Will not be used in queries, only for context
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
                num_queries=remaining
            )
            all_queries.extend(batch_queries)
        
        # Ensure we have exactly the requested number
        return all_queries[:num_queries]
    
    async def _generate_query_batch(
        self,
        company_name: str,
        industry_context: str,
        num_queries: int
    ) -> List[str]:
        """Generate a batch of semantic queries"""
        
        # Create context-aware prompt for query generation
        prompt = self._create_generation_prompt(
            company_name=company_name,
            industry_context=industry_context,
            num_queries=num_queries
        )
        
        try:
            print(f"🔍 Starting Gemini 2.0 Flash query generation...")
            print(f"   Company: {company_name}")
            print(f"   Industry: {industry_context}")
            print(f"   Requested queries: {num_queries}")
            
            # Use Gemini 2.0 Flash for query generation
            print(f"📡 Sending request to Gemini 2.0 Flash...")
            
            # Try to enable Google Search if available in the current SDK
            tools = []
            try:
                # Method 1: Try GoogleSearchRetrieval if available
                if hasattr(genai, 'tools') and hasattr(genai.tools, 'GoogleSearchRetrieval'):
                    tools = [genai.tools.GoogleSearchRetrieval()]
                    print(f"✅ Google Search tool enabled for current trends")
                
                # Method 2: Try function calling approach for search
                elif hasattr(genai.types, 'FunctionDeclaration'):
                    search_function = genai.types.FunctionDeclaration(
                        name="search_current_trends",
                        description=f"Search for current 2025 trends and popular queries in the {industry_context} industry",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": f"Search query about current {industry_context} trends in 2025"
                                }
                            },
                            "required": ["query"]
                        }
                    )
                    
                    search_tool = genai.types.Tool(function_declarations=[search_function])
                    tools = [search_tool]
                    print(f"✅ Search function tool defined for trend research")
                
                else:
                    print(f"⚠️ No search tools available, using enhanced 2025 context prompt")
                    
            except Exception as tool_error:
                print(f"⚠️ Tool setup failed: {str(tool_error)}")
                tools = []
            
            # Make the API call with or without tools
            try:
                if tools:
                    response = await asyncio.to_thread(
                        self.model.generate_content,
                        prompt,
                        tools=tools,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.6,
                            top_p=0.9,
                            top_k=40,
                            max_output_tokens=2000,
                            response_mime_type="application/json"
                        )
                    )
                else:
                    response = await asyncio.to_thread(
                        self.model.generate_content,
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.6,
                            top_p=0.9,
                            top_k=40,
                            max_output_tokens=2000,
                            response_mime_type="application/json"
                        )
                    )
            except Exception as api_error:
                print(f"❌ Gemini API call failed: {str(api_error)}")
                raise api_error
            
            print(f"✅ Gemini 2.0 Flash response received")
            print(f"   Response text length: {len(response.text) if response.text else 0}")
            
            if response.text:
                print(f"📝 Parsing JSON response...")
                result = json.loads(response.text)
                queries = result.get('queries', [])
                print(f"✅ Successfully generated {len(queries)} queries")
                
                # Log first few queries for verification
                print(f"📝 Sample generated queries:")
                for i, query in enumerate(queries[:3], 1):
                    print(f"   {i}. {query}")
                
                # Verify no company names leaked into queries
                company_mentions = sum(1 for q in queries if company_name.lower().replace("'", "").replace(" ", "") in q.lower().replace("'", "").replace(" ", ""))
                if company_mentions > 0:
                    print(f"⚠️  WARNING: {company_mentions} queries contain company name - this will bias results")
                else:
                    print(f"✅ No company name mentions detected - queries are unbiased")
                
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
        company_name: str,  # For context only, not included in queries
        industry_context: str,
        num_queries: int
    ) -> str:
        """Create a focused prompt for generating relevant consumer queries."""
        return f"""
You are an expert market researcher. Your task is to generate a list of realistic search queries that a consumer would use to find products or services in the '{industry_context}' market.

The ultimate goal is to use these queries to test the search visibility of the company '{company_name}'. Therefore, the queries must be highly relevant to '{industry_context}'.

CRITICAL INSTRUCTIONS:
1.  **Strict Relevance:** All queries MUST be about '{industry_context}' itself. Do NOT generate queries about related accessories, components, or sub-industries.
    -   GOOD example (for 'electric cars'): "best long-range electric car"
    -   BAD example (for 'electric cars'): "best tires for electric cars"
    -   GOOD example (for 'smartphones'): "most durable smartphone 2025"
    -   BAD example (for 'smartphones'): "best screen protector for smartphones"

2.  **No Company Names:** Do NOT include '{company_name}' or any other specific company/brand names in the generated queries. The goal is to see if they appear organically in the search results.

3.  **Slight Variations:** Generate queries that are closely related. You can vary them by changing a few words, adding qualifiers, or focusing on slightly different user needs, but stay within the '{industry_context}' topic.

4.  **Query Style:** Create a mix of query types, such as:
    -   **Best of/Rankings:** "best {industry_context}", "top rated {industry_context} 2025"
    -   **Specific Needs:** "{industry_context} for families", "most reliable {industry_context}"
    -   **Comparisons:** "{industry_context} vs competitors", "alternatives to popular {industry_context}"
    -   **Feature-based:** "long range {industry_context}", "{industry_context} with best camera"
    -   **Price-based:** "affordable {industry_context}", "best budget {industry_context}"

Generate {num_queries} diverse but highly relevant search queries for the '{industry_context}' market.

Return the result in a clean JSON format:
{{
    "queries": [
        "query 1 text",
        "query 2 text",
        ...
    ]
}}
"""
    
    def _generate_fallback_queries(
        self,
        company_name: str,  # Not used in queries, only for context
        industry_context: str,
        num_queries: int
    ) -> List[str]:
        """Generate fallback queries using realistic consumer templates"""
        
        # Extract industry keywords
        industry_keywords = self._extract_industry_keywords(company_name, industry_context)
        
        # Realistic consumer query templates (no company names)
        templates = [
            "best {product} 2025",
            "top {product} brands",
            "affordable {product} options", 
            "natural {product} recommendations",
            "long lasting {product}",
            "{product} for sensitive skin",
            "organic {product} brands",
            "drugstore {product} reviews",
            "moisturizing {product}",
            "trending {product} 2025",
            "budget friendly {product}",
            "{product} comparison guide",
            "effective {product} for dry skin",
            "popular {product} right now",
            "sustainable {product} options",
            "{product} with SPF protection",
            "winter {product} recommendations",
            "luxury vs drugstore {product}",
            "cruelty free {product}",
            "unscented {product} brands"
        ]
        
        queries = []
        for i in range(num_queries):
            template = templates[i % len(templates)]
            
            # Fill in template with product keywords only
            query = template.format(
                product=industry_keywords.get('product', industry_context or 'product')
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
