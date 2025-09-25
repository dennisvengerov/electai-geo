import re
import asyncio
import os
from typing import Dict, List, Any, Optional
from simple_entity_discovery import SimpleEntityDiscovery
import spacy

class CitationTracker:
    """Tracks and analyzes company citations in AI responses"""
    
    def __init__(self):
        self.company_patterns = {}
        # Initialize simplified entity discovery (no external dependencies)
        self.entity_discovery = SimpleEntityDiscovery()
        # Load spaCy NER model (English)
        try:
            self.spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.spacy_nlp = spacy.load("en_core_web_sm")
    
    def analyze_citation(self, response: str, company_name: str) -> Dict[str, Any]:
        """
        Analyze if and how a company is cited in the response
        Returns:
            Dictionary with citation analysis results, including competitor positions
        """
        if not response or not company_name:
            return {
                'cited': False,
                'context': None,
                'position': None,
                'competitors': [],
                'competitor_positions': {}
            }
        # Normalize text for analysis
        normalized_response = response.lower()
        normalized_company = company_name.lower()
        # Check for direct citations
        citation_info = self._find_company_mentions(normalized_response, normalized_company, response)
        # Find competitors using enhanced entity discovery
        competitors = self._extract_competitors_enhanced(response, company_name)
        # Determine ranking position if in a list
        position = self._find_ranking_position(response, company_name) if citation_info['cited'] else None
        # Extract all company positions from numbered lists
        competitor_positions = self._extract_all_company_positions(response)
        return {
            'cited': citation_info['cited'],
            'context': citation_info['context'],
            'position': position,
            'competitors': competitors,
            'competitor_positions': competitor_positions
        }
    
    def _find_company_mentions(self, normalized_response: str, normalized_company: str, original_response: str) -> Dict[str, Any]:
        """Find direct mentions of the company in the response"""
        
        # Create various patterns for company name matching
        patterns = self._create_company_patterns(normalized_company)
        
        best_match = None
        best_context = None
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, normalized_response))
            
            if matches:
                # Find the best match (longest or most specific)
                for match in matches:
                    if not best_match or len(match.group()) > len(best_match.group()):
                        best_match = match
                        
                        # Extract context around the match
                        start = max(0, match.start() - 100)
                        end = min(len(original_response), match.end() + 100)
                        best_context = original_response[start:end].strip()
        
        return {
            'cited': bool(best_match),
            'context': best_context
        }
    
    def _create_company_patterns(self, company_name: str) -> List[str]:
        """Create regex patterns for finding company name variations"""
        
        # Escape special regex characters
        escaped_name = re.escape(company_name)
        
        patterns = [
            # Exact match
            rf'\b{escaped_name}\b',
            
            # With common variations
            rf'\b{escaped_name}\'s?\b',  # possessive
            rf'\b{escaped_name}s\b',     # plural
        ]
        
        # Handle multi-word company names
        if ' ' in company_name:
            words = company_name.split()
            
            # Try partial matches for multi-word names
            if len(words) > 1:
                # First word only (for well-known brands)
                patterns.append(rf'\b{re.escape(words[0])}\b')
                
                # Last word only (for product-named companies)
                patterns.append(rf'\b{re.escape(words[-1])}\b')
        
        # Handle special cases
        patterns.extend(self._handle_special_cases(company_name))
        
        return patterns
    
    def _handle_special_cases(self, company_name: str) -> List[str]:
        """Handle special naming patterns and abbreviations"""
        
        special_patterns = []
        
        # Common abbreviation patterns
        if ' ' in company_name:
            words = company_name.split()
            if len(words) <= 3:
                # Create acronym pattern
                acronym = ''.join(word[0].upper() for word in words)
                special_patterns.append(rf'\b{acronym}\b')
        
        # Common company name transformations
        name_lower = company_name.lower()
        
        # Remove common suffixes for matching
        suffixes = ['inc', 'corp', 'ltd', 'llc', 'co', 'company', 'corporation']
        for suffix in suffixes:
            if name_lower.endswith(f' {suffix}'):
                base_name = company_name[:-len(suffix)-1]
                special_patterns.append(rf'\b{re.escape(base_name)}\b')
        
        # Handle ampersands and special characters
        if '&' in company_name:
            # Try 'and' version
            and_version = company_name.replace('&', 'and')
            special_patterns.append(rf'\b{re.escape(and_version)}\b')
        
        return special_patterns
    
    def _extract_competitors(self, response: str, company_name: str) -> List[str]:
        """Extract competitor company names from the response"""
        
        competitors = set()
        
        # Look for list patterns that might contain competitors
        list_patterns = [
            r'\d+\.\s*([A-Z][^.!?]*?)(?=\s*\d+\.|$)',  # Numbered lists
            r'[-•]\s*([A-Z][^.!?]*?)(?=\s*[-•]|$)',     # Bullet lists
            r'([A-Z][a-zA-Z\s&]+?)(?:\s+[-–—]\s|:)',   # Company names followed by dash or colon
        ]
        
        for pattern in list_patterns:
            matches = re.findall(pattern, response, re.MULTILINE)
            for match in matches:
                # Clean up the match
                clean_match = self._clean_company_name(match)
                
                # Skip if it's the target company or too short/generic
                if (clean_match and 
                    len(clean_match) > 2 and 
                    clean_match.lower() != company_name.lower() and
                    not self._is_generic_term(clean_match)):
                    competitors.add(clean_match)
        
        # Look for explicit comparison phrases
        comparison_patterns = [
            rf'(?:compared to|versus|vs\.?|against)\s+([A-Z][a-zA-Z\s&]+)',
            rf'([A-Z][a-zA-Z\s&]+)\s+(?:compared to|versus|vs\.?|against)',
            rf'(?:like|such as|including)\s+([A-Z][a-zA-Z\s&,]+)',
        ]
        
        for pattern in comparison_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                # Split on commas and clean each name
                names = [self._clean_company_name(name.strip()) for name in match.split(',')]
                for name in names:
                    if (name and 
                        len(name) > 2 and 
                        name.lower() != company_name.lower() and
                        not self._is_generic_term(name)):
                        competitors.add(name)
        
        return list(competitors)
    
    def _clean_company_name(self, name: str) -> Optional[str]:
        """Clean and normalize a potential company name"""
        
        if not name:
            return None
        
        # Remove common prefixes/suffixes that aren't part of company names
        name = re.sub(r'^\d+\.\s*', '', name)  # Remove numbering
        name = re.sub(r'^[-•]\s*', '', name)   # Remove bullet points
        name = name.strip()
        
        # Remove trailing punctuation and common suffixes
        name = re.sub(r'[.!?:,;]+$', '', name)
        name = re.sub(r'\s+(?:is|are|offers|provides|has|have).*$', '', name, flags=re.IGNORECASE)
        
        # Clean up whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name if name else None
    
    def _is_generic_term(self, term: str) -> bool:
        """Check if a term is too generic to be a company name"""
        
        generic_terms = {
            'company', 'brand', 'product', 'service', 'business', 'store',
            'website', 'platform', 'solution', 'system', 'app', 'software',
            'the best', 'top rated', 'most popular', 'leading', 'major',
            'industry', 'market', 'sector', 'category', 'type', 'kind',
            'other', 'another', 'many', 'several', 'various', 'different'
        }
        
        return term.lower() in generic_terms or len(term.split()) > 5
    
    def _find_ranking_position(self, response: str, company_name: str) -> Optional[int]:
        """Find the ranking position of the company if mentioned in a numbered list"""
        
        # Look for numbered list patterns
        lines = response.split('\n')
        
        for line in lines:
            # Check if line contains the company and a number
            if company_name.lower() in line.lower():
                # Look for numbered list pattern
                number_match = re.search(r'^(\d+)\.', line.strip())
                if number_match:
                    return int(number_match.group(1))
                
                # Look for ranking words
                ranking_patterns = [
                    r'(?:number|#)\s*(\d+)',
                    r'(\d+)(?:st|nd|rd|th)\s+(?:place|position|rank)',
                    r'ranked\s+(?:#\s*)?(\d+)',
                ]
                
                for pattern in ranking_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        return int(match.group(1))
        
        return None
    
    def _extract_competitors_enhanced(self, response: str, company_name: str) -> list:
        """Extract competitors using both regex/heuristics and spaCy NER, then validate."""
        competitors = set()
        # Use simplified entity discovery (regex/heuristics)
        try:
            competitors_data = self.entity_discovery.discover_competitors(response, company_name)
            competitors.update([comp['name'] for comp in competitors_data if comp.get('confidence', 0) >= 0.65])
        except Exception as e:
            print(f"Entity discovery failed: {e}")
        # Use spaCy NER
        try:
            doc = self.spacy_nlp(response)
            for ent in doc.ents:
                if ent.label_ == "ORG" and ent.text.lower() != company_name.lower():
                    competitors.add(ent.text)
        except Exception as e:
            print(f"spaCy NER failed: {e}")
        # Validate competitors using external API
        competitors_list = list(competitors)
        validated_competitors = self._validate_competitor_list_external_api(competitors_list)
        return validated_competitors

    def _extract_all_company_positions(self, response: str) -> dict:
        """Extract all company names and their positions from numbered lists in the response."""
        import re
        company_positions = {}
        lines = response.split('\n')
        for line in lines:
            # Match lines like '1. Company Name ...' or '2. Another Company ...'
            match = re.match(r'^(\d+)\.\s*([A-Z][^\n\r\d\.:,\-]+)', line.strip())
            if match:
                pos = int(match.group(1))
                name = match.group(2).strip()
                # Clean up name (remove trailing punctuation, etc.)
                name = re.sub(r'[\.:,\-]+$', '', name)
                if name:
                    company_positions[name] = pos
        return company_positions

    def _validate_competitor_list_external_api(self, competitors: list) -> list:
        """Send the competitor list to an external API to filter out non-company words."""
        import os
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("No OpenAI API key found, skipping validation.")
            return competitors
        openai.api_key = api_key
        prompt = (
            "Given the following list of words, return only those that are real company or organization names. "
            "Remove any generic, non-company words (like 'the', 'best', etc). "
            "List only the valid company names, one per line.\nList: " + ", ".join(competitors)
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0
            )
            filtered = response.choices[0].message["content"].split("\n")
            filtered = [name.strip() for name in filtered if name.strip()]
            return [name for name in filtered if name in competitors]
        except Exception as e:
            print(f"External API validation failed: {e}")
            return competitors
