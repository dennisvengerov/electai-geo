from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import time


@dataclass
class RawQueryResult:
    """Raw query result before analysis"""
    query: str
    response: str
    execution_time: float
    timestamp: float


@dataclass
class CompanyMention:
    """Information about a company mentioned in a response"""
    name: str
    position: Optional[int]
    context: str
    confidence: float


@dataclass
class QueryAnalysis:
    """Analysis result for a single query"""
    query_id: int
    query_text: str
    target_company_cited: bool
    citation_context: Optional[str]
    target_company_position: Optional[int]
    all_companies_mentioned: List[CompanyMention]
    mention_type: str  # "direct", "indirect", "implied", "none"
    confidence: float
    ranking_type: Optional[str] = None  # "top 10 best brands", etc.
    has_numbered_ranking: bool = False
    total_companies_in_ranking: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'query_id': self.query_id,
            'query_text': self.query_text,
            'target_company_cited': self.target_company_cited,
            'citation_context': self.citation_context,
            'target_company_position': self.target_company_position,
            'all_companies_mentioned': [
                {
                    'name': mention.name,
                    'position': mention.position,
                    'context': mention.context,
                    'confidence': mention.confidence
                }
                for mention in self.all_companies_mentioned
            ],
            'mention_type': self.mention_type,
            'confidence': self.confidence,
            'ranking_type': self.ranking_type,
            'has_numbered_ranking': self.has_numbered_ranking,
            'total_companies_in_ranking': self.total_companies_in_ranking
        }


@dataclass
class CompetitorSummary:
    """Summary information about a competitor"""
    name: str
    total_mentions: int
    average_position: float
    contexts: List[str]
    confidence: float


@dataclass
class BatchAnalysisResult:
    """Complete batch analysis result"""
    query_analyses: List[QueryAnalysis]
    aggregate_analysis: Dict[str, Any]
    analysis_timestamp: float
    total_execution_time: float

    @classmethod
    def from_json(cls, json_str: str, execution_time: float = 0.0) -> 'BatchAnalysisResult':
        """Create BatchAnalysisResult from Gemini JSON response"""
        try:
            data = json.loads(json_str)
            
            # Parse query analyses
            query_analyses = []
            for qa_data in data.get('query_analyses', []):
                company_mentions = []
                for mention_data in qa_data.get('all_companies_mentioned', []):
                    company_mentions.append(CompanyMention(
                        name=mention_data.get('name', ''),
                        position=mention_data.get('position'),
                        context=mention_data.get('context', ''),
                        confidence=mention_data.get('confidence', 0.0)
                    ))
                
                query_analyses.append(QueryAnalysis(
                    query_id=qa_data.get('query_id', 0),
                    query_text=qa_data.get('query_text', ''),
                    target_company_cited=qa_data.get('target_company_cited', False),
                    citation_context=qa_data.get('citation_context'),
                    target_company_position=qa_data.get('target_company_position'),
                    all_companies_mentioned=company_mentions,
                    mention_type=qa_data.get('mention_type', 'none'),
                    confidence=qa_data.get('confidence', 0.0),
                    ranking_type=qa_data.get('ranking_type'),
                    has_numbered_ranking=qa_data.get('has_numbered_ranking', False),
                    total_companies_in_ranking=qa_data.get('total_companies_in_ranking')
                ))
            
            return cls(
                query_analyses=query_analyses,
                aggregate_analysis=data.get('aggregate_analysis', {}),
                analysis_timestamp=time.time(),
                total_execution_time=execution_time
            )
        
        except Exception as e:
            # Return empty result if parsing fails
            return cls(
                query_analyses=[],
                aggregate_analysis={'error': f'Failed to parse JSON: {str(e)}'},
                analysis_timestamp=time.time(),
                total_execution_time=execution_time
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export"""
        return {
            'query_analyses': [qa.to_dict() for qa in self.query_analyses],
            'aggregate_analysis': self.aggregate_analysis,
            'analysis_timestamp': self.analysis_timestamp,
            'total_execution_time': self.total_execution_time
        }

    def get_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics compatible with existing UI"""
        total_queries = len(self.query_analyses)
        cited_queries = [qa for qa in self.query_analyses if qa.target_company_cited]
        total_citations = len(cited_queries)
        
        # Calculate average position
        positions = [qa.target_company_position for qa in cited_queries 
                    if qa.target_company_position is not None]
        average_position = sum(positions) / len(positions) if positions else 0
        
        # Extract competitors from aggregate analysis
        competitors = []
        unique_competitors = self.aggregate_analysis.get('unique_competitors', [])
        for comp_data in unique_competitors:
            competitors.append({
                'name': comp_data.get('name', ''),
                'mentions': comp_data.get('total_mentions', 0),
                'percentage': (comp_data.get('total_mentions', 0) / total_queries) * 100 if total_queries > 0 else 0,
                'avg_position': comp_data.get('average_position', 0)
            })
        
        return {
            'total_queries': total_queries,
            'total_citations': total_citations,
            'citation_rate': (total_citations / total_queries) * 100 if total_queries > 0 else 0,
            'average_position': average_position,
            'competitors': competitors,
            'avg_execution_time': self.total_execution_time / total_queries if total_queries > 0 else 0,
            'successful_queries': total_queries,
            'analysis_confidence': self.aggregate_analysis.get('analysis_confidence', 0.0)
        }

    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy format for compatibility with existing UI"""
        query_results = []
        
        for qa in self.query_analyses:
            # Extract competitor positions from all companies mentioned
            competitor_positions = {}
            for mention in qa.all_companies_mentioned:
                if mention.position is not None:
                    competitor_positions[mention.name] = mention.position
            
            query_results.append({
                'query': qa.query_text,
                'response': qa.citation_context or 'Analysis performed by Gemini batch processor',
                'cited': qa.target_company_cited,
                'context': qa.citation_context,
                'position': qa.target_company_position,
                'execution_time': self.total_execution_time / len(self.query_analyses) if self.query_analyses else 0,
                'competitors': [mention.name for mention in qa.all_companies_mentioned],
                'competitor_positions': competitor_positions,
                'mention_type': qa.mention_type,
                'confidence': qa.confidence,
                'ranking_type': qa.ranking_type,
                'has_numbered_ranking': qa.has_numbered_ranking,
                'total_companies_in_ranking': qa.total_companies_in_ranking
            })
        
        return {
            'query_results': query_results,
            'summary': self.get_summary_stats(),
            'analysis_timestamp': self.analysis_timestamp,
            'total_execution_time': self.total_execution_time
        }
