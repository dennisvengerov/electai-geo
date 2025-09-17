import json
import csv
import io
from typing import Dict, Any, List
from datetime import datetime

def export_to_csv(analysis_results: Dict[str, Any]) -> str:
    """
    Export analysis results to CSV format
    
    Args:
        analysis_results: The complete analysis results dictionary
        
    Returns:
        CSV data as string
    """
    
    output = io.StringIO()
    
    # Summary section
    output.write("# GEO Analysis Summary\n")
    output.write(f"Company,{analysis_results['company_name']}\n")
    output.write(f"Industry Context,{analysis_results.get('industry_context', 'N/A')}\n")
    output.write(f"Analysis Date,{datetime.fromtimestamp(analysis_results['analysis_timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\n")
    output.write(f"Total Execution Time,{analysis_results['total_execution_time']:.2f} seconds\n")
    output.write("\n")
    
    # Key metrics
    summary = analysis_results['summary']
    output.write("# Key Metrics\n")
    output.write(f"Total Queries,{summary['total_queries']}\n")
    output.write(f"Total Citations,{summary['total_citations']}\n")
    output.write(f"Citation Rate,{summary['citation_rate']:.2f}%\n")
    output.write(f"Average Position,{summary['average_position']:.1f}\n")
    output.write(f"Average Execution Time,{summary['avg_execution_time']:.2f} seconds\n")
    output.write("\n")
    
    # Query results
    output.write("# Query Results\n")
    
    if analysis_results.get('query_results'):
        # Write header
        output.write("Query,Cited,Position,Context,Execution Time,Response Preview\n")
        
        # Write query data
        writer = csv.writer(output)
        for result in analysis_results['query_results']:
            row = [
                result['query'],
                'Yes' if result['cited'] else 'No',
                result.get('position', ''),
                result.get('context', ''),
                f"{result.get('execution_time', 0):.2f}",
                (result['response'][:100] + '...') if len(result['response']) > 100 else result['response']
            ]
            writer.writerow(row)
    
    output.write("\n")
    
    # Competitor analysis
    if summary.get('competitors'):
        output.write("# Competitor Analysis\n")
        output.write("Competitor,Mentions,Percentage\n")
        
        writer = csv.writer(output)
        for competitor in summary['competitors']:
            row = [
                competitor['name'],
                competitor['mentions'],
                f"{competitor['percentage']:.1f}%"
            ]
            writer.writerow(row)
    
    return output.getvalue()

def export_to_json(analysis_results: Dict[str, Any]) -> str:
    """
    Export analysis results to JSON format
    
    Args:
        analysis_results: The complete analysis results dictionary
        
    Returns:
        JSON data as string
    """
    
    # Create a clean copy for export
    export_data = {
        'metadata': {
            'company_name': analysis_results['company_name'],
            'industry_context': analysis_results.get('industry_context', ''),
            'analysis_timestamp': analysis_results['analysis_timestamp'],
            'analysis_date': datetime.fromtimestamp(analysis_results['analysis_timestamp']).isoformat(),
            'total_execution_time': analysis_results['total_execution_time'],
            'export_timestamp': datetime.now().isoformat()
        },
        'summary': analysis_results['summary'],
        'query_results': analysis_results.get('query_results', []),
        'analysis_parameters': {
            'total_queries_analyzed': len(analysis_results.get('query_results', [])),
            'successful_queries': analysis_results['summary'].get('successful_queries', 0)
        }
    }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def format_citation_context(context: str, max_length: int = 200) -> str:
    """
    Format citation context for display
    
    Args:
        context: The citation context string
        max_length: Maximum length for the formatted context
        
    Returns:
        Formatted context string
    """
    
    if not context:
        return "No context available"
    
    # Clean up the context
    context = context.strip()
    
    # Truncate if too long
    if len(context) > max_length:
        context = context[:max_length-3] + "..."
    
    return context

def calculate_visibility_score(citation_rate: float, avg_position: float, total_citations: int) -> Dict[str, Any]:
    """
    Calculate a visibility score based on multiple factors
    
    Args:
        citation_rate: Percentage of queries where company was cited
        avg_position: Average ranking position when cited
        total_citations: Total number of citations
        
    Returns:
        Dictionary with visibility score and breakdown
    """
    
    # Base score from citation rate (0-100)
    base_score = citation_rate
    
    # Position modifier (better positions get higher multiplier)
    if avg_position > 0:
        position_modifier = max(0.5, (11 - min(avg_position, 10)) / 10)
    else:
        position_modifier = 0.5
    
    # Volume modifier (more citations = higher confidence)
    volume_modifier = min(1.2, 0.8 + (total_citations / 50))
    
    # Calculate final score
    visibility_score = base_score * position_modifier * volume_modifier
    visibility_score = min(100, max(0, visibility_score))
    
    # Determine grade
    if visibility_score >= 80:
        grade = "A"
        description = "Excellent visibility"
    elif visibility_score >= 60:
        grade = "B"
        description = "Good visibility"
    elif visibility_score >= 40:
        grade = "C"
        description = "Fair visibility"
    elif visibility_score >= 20:
        grade = "D"
        description = "Poor visibility"
    else:
        grade = "F"
        description = "Very poor visibility"
    
    return {
        'score': round(visibility_score, 1),
        'grade': grade,
        'description': description,
        'components': {
            'citation_rate': citation_rate,
            'position_modifier': position_modifier,
            'volume_modifier': volume_modifier
        }
    }

def generate_insights(analysis_results: Dict[str, Any]) -> List[str]:
    """
    Generate actionable insights from analysis results
    
    Args:
        analysis_results: The complete analysis results dictionary
        
    Returns:
        List of insight strings
    """
    
    insights = []
    summary = analysis_results['summary']
    
    citation_rate = summary['citation_rate']
    avg_position = summary['average_position']
    competitors = summary.get('competitors', [])
    
    # Citation rate insights
    if citation_rate >= 50:
        insights.append(f"✅ Strong visibility with {citation_rate:.1f}% citation rate")
    elif citation_rate >= 25:
        insights.append(f"⚠️ Moderate visibility at {citation_rate:.1f}% - room for improvement")
    else:
        insights.append(f"❌ Low visibility at {citation_rate:.1f}% - significant optimization needed")
    
    # Position insights
    if avg_position > 0:
        if avg_position <= 3:
            insights.append(f"🏆 Excellent average ranking at position {avg_position:.1f}")
        elif avg_position <= 5:
            insights.append(f"👍 Good average ranking at position {avg_position:.1f}")
        else:
            insights.append(f"📈 Average ranking at position {avg_position:.1f} - aim for top 5")
    
    # Competitor insights
    if competitors:
        top_competitor = competitors[0]
        if top_competitor['mentions'] > summary['total_citations']:
            insights.append(f"🏃 {top_competitor['name']} appears more frequently - analyze their strategy")
        
        if len(competitors) > 10:
            insights.append(f"🌐 High competition detected with {len(competitors)} competitors mentioned")
    
    # Performance insights
    avg_time = summary.get('avg_execution_time', 0)
    if avg_time > 5:
        insights.append(f"⏱️ Slow query performance at {avg_time:.1f}s average - consider optimization")
    
    return insights
