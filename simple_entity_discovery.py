import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse

@dataclass
class SimpleEntityCandidate:
    name: str
    confidence: float
    source: str
    domain: Optional[str] = None

def extract_domains_from_text(text: str) -> List[Tuple[str, int, int]]:
    """Extract domains from URLs in text"""
    url_pattern = r"https?://[^\s)>\]]+"
    hits = []
    for m in re.finditer(url_pattern, text):
        url = m.group(0).rstrip(".,);]")
        try:
            netloc = urlparse(url).netloc.lower()
            # eTLD+1 approximation
            parts = netloc.split(".")
            if len(parts) >= 2:
                primary_domain = ".".join(parts[-2:])
            else:
                primary_domain = netloc
            hits.append((primary_domain, m.start(), m.end()))
        except Exception:
            continue
    return hits

def extract_company_names(text: str) -> List[Tuple[str, int, int]]:
    """Extract potential company names using improved pattern matching"""
    
    # Enhanced company suffix patterns
    company_suffixes = r"(Inc\.?|LLC|Ltd\.?|GmbH|SAS|SA|AB|BV|NV|PLC|Pty|KK|Co\.|Labs|AI|Group|Holdings|Technologies|Corp\.?|Corporation|Company|Systems|Solutions|Software|Services|Partners|Ventures|Capital|Fund|Institute|Foundation|University|Academy|School|Hospital|Medical|Center|Bank|Financial|Insurance|Real Estate|Realty|Construction|Engineering|Manufacturing|Industries|International|Global|Worldwide|Enterprises|Consulting|Advisory|Management|Development|Research|Innovation|Digital|Tech|Cyber|Cloud|Data|Analytics|Intelligence|Security|Networks|Communications|Media|Entertainment|Gaming|Sports|Fitness|Health|Wellness|Beauty|Fashion|Retail|Commerce|Trade|Marketing|Advertising|Design|Creative|Studio|Agency|Publishing|Broadcasting|News|Times|Post|Herald|Tribune|Journal|Magazine|Books|Press|Films|Pictures|Music|Records)"
    
    # Pattern for proper company names
    pattern = rf"\b([A-Z][a-zA-Z0-9&\-']+(?:\s+[A-Z&][a-zA-Z0-9&\-']*)*(?:\s+{company_suffixes})?)\b"
    matches = []
    
    for m in re.finditer(pattern, text):
        company_name = m.group(1)
        if _is_likely_company_name(company_name):
            matches.append((company_name, m.start(1), m.end(1)))
    
    return matches

def _is_likely_company_name(name: str) -> bool:
    """Filter out obvious non-companies using heuristics"""
    name_lower = name.lower().strip()
    
    # Skip very short names
    if len(name) < 2:
        return False
    
    # Skip phrases that start with descriptive words or contain obvious non-company terms
    descriptor_words = ['features', 'includes', 'focus', 'premium', 'original', 'current', 'new', 'classic', 'popular', 'exclusive', 'free', 'live', 'international', 'specialized', 'reality', 'family', 'friendly', 'ad', 'free', 'excellent']
    first_word = name_lower.split()[0] if ' ' in name_lower else name_lower
    if first_word in descriptor_words:
        return False
        
    # Filter out common entertainment franchises that aren't streaming services
    entertainment_franchises = ['star wars', 'marvel', 'disney content', 'nbc content', 'cbs content', 'reality shows', 'classic films', 'tv shows', 'movie releases']
    if name_lower in entertainment_franchises:
        return False
    
    # Comprehensive list of common words that aren't companies
    non_companies = {
        # Articles, prepositions, conjunctions
        'the', 'and', 'or', 'but', 'for', 'with', 'by', 'from', 'to', 'of', 'in', 'on', 'at',
        # Pronouns and determiners
        'this', 'that', 'these', 'those', 'they', 'them', 'their', 'there', 'then', 'than',
        'when', 'where', 'why', 'what', 'who', 'how', 'which', 'while', 'some', 'many', 'all',
        # Common adjectives
        'new', 'old', 'good', 'bad', 'best', 'better', 'great', 'big', 'small', 'long', 'short', 
        'high', 'low', 'hot', 'cold', 'fast', 'slow', 'easy', 'hard', 'free', 'cheap', 'expensive',
        'popular', 'top', 'leading', 'major', 'main', 'first', 'last', 'next', 'previous',
        'available', 'online', 'digital', 'modern', 'traditional', 'professional', 'local',
        'national', 'international', 'global', 'worldwide', 'premium', 'standard', 'basic',
        # Generic business terms
        'company', 'business', 'service', 'product', 'solution', 'platform', 'system', 'app',
        'website', 'store', 'shop', 'market', 'industry', 'sector', 'category', 'type', 'kind',
        'brand', 'model', 'version', 'option', 'choice', 'alternative', 'provider', 'supplier',
        'customer', 'client', 'user', 'member', 'partner', 'team', 'group', 'organization',
        # Time and quantity words
        'today', 'tomorrow', 'yesterday', 'now', 'soon', 'later', 'before', 'after', 'during',
        'always', 'never', 'sometimes', 'often', 'usually', 'rarely', 'frequently', 'recently',
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'first', 'second', 'third', 'fourth', 'fifth', 'last', 'final', 'initial',
        # Action words that might appear in caps
        'get', 'buy', 'sell', 'use', 'try', 'find', 'search', 'choose', 'select', 'pick',
        'start', 'begin', 'stop', 'end', 'finish', 'continue', 'learn', 'discover', 'explore',
        'read', 'write', 'create', 'make', 'build', 'develop', 'design', 'plan', 'manage',
        # Common false positives from AI responses
        'here', 'below', 'above', 'following', 'previous', 'mentioned', 'listed', 'shown',
        'included', 'featured', 'recommended', 'suggested', 'noted', 'important', 'key',
        'main', 'primary', 'secondary', 'additional', 'extra', 'bonus', 'special', 'unique',
        'custom', 'personalized', 'tailored', 'specific', 'general', 'common', 'typical',
        'standard', 'regular', 'normal', 'average', 'basic', 'simple', 'complex', 'advanced',
        'other', 'others', 'another', 'similar', 'different', 'various', 'several', 'multiple',
        'alternative', 'alternatives', 'option', 'options', 'choice', 'choices', 'example',
        'examples', 'service', 'services', 'provider', 'providers', 'company', 'companies',
        'features', 'includes', 'current', 'content', 'show', 'shows',
        'movie', 'movies', 'film', 'films', 'series', 'episode', 'episodes', 'season', 'seasons',
        'original', 'programming', 'program', 'programs', 'channel', 'channels', 'network',
        'networks', 'platform', 'platforms', 'viewing', 'watch', 'watching', 'stream', 'streaming'
    }
    
    if name_lower in non_companies:
        return False
    
    # Skip single words that are clearly not company names
    if ' ' not in name:
        # Allow single words with clear company indicators
        company_indicators = ['inc', 'llc', 'ltd', 'corp', 'co', 'labs', 'ai', 'tech', 'software', 'systems', 'solutions', 'group', 'holdings', 'capital', 'ventures', 'partners', 'consulting', 'services', 'media', 'entertainment', 'games', 'gaming', 'studio', 'agency', 'publishing', 'broadcasting', 'bank', 'financial', 'insurance', 'hospital', 'medical', 'university', 'college', 'school', 'institute', 'foundation', 'center', 'networks', 'communications', 'security', 'digital', 'cloud', 'data', 'analytics', 'research', 'development', 'innovation', 'manufacturing', 'industries', 'construction', 'engineering', 'real', 'realty', 'retail', 'commerce', 'trade', 'fashion', 'beauty', 'health', 'fitness', 'wellness', 'sports']
        
        has_indicator = any(indicator in name_lower for indicator in company_indicators)
        if not has_indicator:
            # For single words without indicators, be very restrictive
            # Only allow if it starts with capital and has mixed case or numbers
            if not (name[0].isupper() and (any(c.islower() for c in name[1:]) or any(c.isdigit() for c in name))):
                return False
    
    # Require at least one uppercase letter (proper noun)
    if not any(c.isupper() for c in name):
        return False
    
    # Skip if it's all uppercase and short (likely acronym without context)
    if name.isupper() and len(name) < 4 and ' ' not in name:
        return False
    
    return True

class SimpleEntityDiscovery:
    """Simplified entity discovery without external dependencies"""
    
    def __init__(self):
        pass
    
    def discover_competitors(self, response: str, company_name: str) -> List[Dict]:
        """
        Discover competitors from response text using pattern matching
        
        Args:
            response: The AI response text to analyze
            company_name: The target company name to exclude
            
        Returns:
            List of competitor dictionaries with name, confidence, and source
        """
        
        if not response or not company_name:
            return []
        
        competitors = []
        
        # Extract domains first
        domains = extract_domains_from_text(response)
        domain_set = {domain for domain, _, _ in domains}
        
        # Extract company names using pattern matching
        company_names = extract_company_names(response)
        
        # Process candidates
        seen_names = set()
        company_name_lower = company_name.lower()
        
        for name, start, end in company_names:
            name_lower = name.lower()
            
            # Skip the target company
            if name_lower == company_name_lower:
                continue
            
            # Skip duplicates
            if name_lower in seen_names:
                continue
            
            seen_names.add(name_lower)
            
            # Calculate confidence based on various factors
            confidence = 0.6  # Base confidence (higher starting point)
            
            # Boost confidence for names with company suffixes
            has_suffix = bool(re.search(r'\b(Inc\.?|LLC|Ltd\.?|Corp\.?|Co\.)', name, re.IGNORECASE))
            if has_suffix:
                confidence += 0.2
            
            # Boost confidence for multi-word names (more likely to be companies)
            if len(name.split()) > 1:
                confidence += 0.15
            
            # Boost confidence if mentioned near a domain
            near_domain = any(abs(start - d_start) < 50 for _, d_start, _ in domains)
            if near_domain:
                confidence += 0.1
            
            # Boost confidence if it has a matching domain
            potential_domain = name_lower.replace(' ', '').replace('.', '') + '.com'
            if any(potential_domain in domain for domain in domain_set):
                confidence += 0.2
            
            # Boost confidence for well-known technology/entertainment terms
            tech_terms = ['tv', 'plus', 'max', 'pro', 'prime', 'video', 'music', 'studios', 'pictures', 'entertainment', 'media', 'streaming', 'games', 'gaming', 'tech', 'technologies', 'software', 'systems', 'solutions', 'services', 'digital', 'data', 'cloud', 'security', 'networks']
            has_tech_term = any(term in name_lower for term in tech_terms)
            if has_tech_term:
                confidence += 0.1
            
            # Boost confidence for brands mentioned in competitive contexts
            context_window = response[max(0, start-100):end+100].lower()
            competitive_phrases = ['alternative', 'competitor', 'similar to', 'like', 'such as', 'including', 'consider', 'also', 'instead', 'better than', 'compared to', 'vs', 'versus', 'other options', 'also available', 'you might']
            in_competitive_context = any(phrase in context_window for phrase in competitive_phrases)
            if in_competitive_context:
                confidence += 0.15
            
            # Boost confidence for items in lists (numbered or bulleted)
            list_patterns = [r'^\s*[-•*]\s*', r'^\s*\d+[\.\)]\s*']
            context_line = response[max(0, start-50):end+50]
            in_list = any(re.search(pattern, context_line, re.MULTILINE) for pattern in list_patterns)
            if in_list:
                confidence += 0.1
            
            # Only include if confidence is reasonable
            if confidence >= 0.6:
                competitors.append({
                    'name': name,
                    'confidence': min(confidence, 1.0),
                    'source': 'pattern_matching',
                    'domain': None,  # Could be enhanced to match domains
                    'verification_reason': f'Pattern-based detection (confidence: {confidence:.2f})'
                })
        
        # Sort by confidence and return top candidates
        competitors.sort(key=lambda x: x['confidence'], reverse=True)
        return competitors[:20]  # Limit to top 20 competitors