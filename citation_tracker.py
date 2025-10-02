import re
from typing import Dict, List, Any

class CitationTracker:
    """Simple citation tracker for legacy fallback mode. Gemini handles accurate analysis."""
    
    def __init__(self):
        # Company name normalization cache
        self._normalization_cache = {}
    
    def analyze_citation(self, response: str, company_name: str) -> Dict[str, Any]:
        """
        Simple citation analysis - only checks if company is mentioned.
        Gemini batch analysis handles accurate company extraction and positioning.
        """
        if not response or not company_name:
            return {
                'cited': False,
                'context': None,
                'position': None,
                'competitors': [],
                'competitor_positions': {}
            }
        
        # Simple citation detection
        citation_info = self._find_company_mentions(response, company_name)
        
        return {
            'cited': citation_info['cited'],
            'context': citation_info['context'],
            'position': None,  # Position detection handled by Gemini
            'competitors': [],  # Competitor extraction handled by Gemini
            'competitor_positions': {}  # Handled by Gemini
        }
    
    def _find_company_mentions(self, response: str, company_name: str) -> Dict[str, Any]:
        """Simple company mention detection"""
        
        # Normalize for case-insensitive matching
        normalized_response = response.lower()
        normalized_company = company_name.lower()
        
        # Create basic patterns for company name matching
        patterns = self._create_company_patterns(normalized_company)
        
        best_match = None
        best_context = None
        
        for pattern in patterns:
            match = re.search(pattern, normalized_response)
            if match:
                best_match = match
                # Extract context around the match
                start = max(0, match.start() - 100)
                end = min(len(response), match.end() + 100)
                best_context = response[start:end].strip()
                break  # Use first match found
        
        return {
            'cited': bool(best_match),
            'context': best_context
        }
    
    def _create_company_patterns(self, company_name: str) -> List[str]:
        """Create simple regex patterns for company name matching"""
        
        # Escape special regex characters
        escaped_name = re.escape(company_name)
        
        patterns = [
            # Exact match
            rf'\b{escaped_name}\b',
            # With possessive
            rf'\b{escaped_name}\'s\b',
        ]
        
        # Handle multi-word company names - try first word for well-known brands
        if ' ' in company_name:
            words = company_name.split()
            if len(words) > 1:
                first_word = re.escape(words[0])
                patterns.append(rf'\b{first_word}\b')
        
        # Handle common company name variations
        normalized_name = self._normalize_company_name(company_name)
        if normalized_name != company_name:
            escaped_normalized = re.escape(normalized_name.lower())
            patterns.append(rf'\b{escaped_normalized}\b')
        
        return patterns
    
    def _normalize_company_name(self, company_name: str) -> str:
        """
        Normalize company names to handle variations like 'Tesla' vs 'Tesla, Inc.'
        Returns the canonical form of the company name.
        """
        if not company_name:
            return company_name
            
        # Check cache first
        if company_name in self._normalization_cache:
            return self._normalization_cache[company_name]
        
        # Start with the original name
        normalized = company_name.strip()
        
        # Remove common suffixes and punctuation
        suffixes_to_remove = [
            ', Inc.', ', Inc', ' Inc.', ' Inc',
            ', Corp.', ', Corp', ' Corp.', ' Corp',
            ', Corporation', ' Corporation',
            ', Ltd.', ', Ltd', ' Ltd.', ' Ltd',
            ', LLC', ' LLC',
            ', Co.', ', Co', ' Co.', ' Co',
            ', Company', ' Company',
            ', L.P.', ' L.P.',
            ', LP', ' LP',
            ', PLC', ' PLC',
            ', S.A.', ' S.A.',
            ', AG', ' AG',
            ', GmbH', ' GmbH'
        ]
        
        # Apply suffix removal (case insensitive)
        normalized_lower = normalized.lower()
        for suffix in suffixes_to_remove:
            if normalized_lower.endswith(suffix.lower()):
                normalized = normalized[:-len(suffix)]
                break
        
        # Remove trailing punctuation and whitespace
        normalized = re.sub(r'[,.\s]+$', '', normalized).strip()
        
        # Handle special cases
        special_cases = {
            'alphabet': 'Google',  # Alphabet Inc. -> Google
            'google llc': 'Google',  # Google LLC -> Google
            'meta platforms': 'Meta',  # Meta Platforms -> Meta
            'x corp': 'X',  # X Corp -> X (formerly Twitter)
            'twitter': 'X',  # Twitter -> X
        }
        
        normalized_key = normalized.lower()
        if normalized_key in special_cases:
            normalized = special_cases[normalized_key]
        
        # Cache the result
        self._normalization_cache[company_name] = normalized
        return normalized