import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import json
from datetime import datetime
import time

from geo_analyzer import GEOAnalyzer
from utils import export_to_csv, export_to_json

# Page configuration
st.set_page_config(
    page_title="GEO Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

def main():
    st.title("🔍 GEO Analytics Dashboard")
    st.markdown("**Generative Engine Optimization Analytics** - Test your company's visibility across semantic search queries")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key checks
        api_key_status = st.empty()
        gemini_key_status = st.empty()
        
        # OpenAI API Key
        if not st.session_state.get('openai_api_key'):
            import os
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                st.session_state.openai_api_key = openai_key
                api_key_status.success("✅ OpenAI API Key loaded")
            else:
                api_key_status.error("❌ OpenAI API Key not found in environment")
                st.stop()
        else:
            api_key_status.success("✅ OpenAI API Key loaded")
        
        # Gemini API Key (REQUIRED for new pipeline)
        if not st.session_state.get('gemini_api_key'):
            import os
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                st.session_state.gemini_api_key = gemini_key
                gemini_key_status.success("✅ Gemini API Key loaded")
            else:
                gemini_key_status.error("❌ Gemini API Key not found in environment")
                st.error("🚨 Gemini API Key is REQUIRED for the new pipeline (query generation + analysis)")
                st.info("Please set GEMINI_API_KEY environment variable and restart the app")
                st.stop()
        else:
            gemini_key_status.success("✅ Gemini API Key loaded")
        
        # Analysis method selection
        st.subheader("Analysis Method")
        if st.session_state.get('gemini_api_key'):
            use_batch_analysis = st.radio(
                "Analysis Method:",
                ["Batch Analysis (Gemini)", "Legacy Analysis"],
                index=0,
                help="Batch analysis uses Gemini's 1M context window for more accurate competitor extraction"
            ) == "Batch Analysis (Gemini)"
        else:
            use_batch_analysis = False
            st.info("🔧 Batch analysis requires Gemini API key - using legacy method")
        
        st.session_state.use_batch_analysis = use_batch_analysis
        
        # Query generation improvements
        st.info("🎯 **Improved Query Generation (2025)**")
        st.markdown("""
        - ✅ **Unbiased queries** - No company names mentioned
        - ✅ **Real consumer searches** - "best lip balm 2025", "natural lip care"
        - ✅ **Current trends** - Uses live 2025 data via Google Search
        - ✅ **Organic citations** - Tests true market visibility
        """)
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        num_queries = st.slider("Number of queries", min_value=2, max_value=20, value=5, step=1)
        max_concurrent = st.slider("Max concurrent requests", min_value=1, max_value=10, value=5)
        
        if use_batch_analysis:
            st.success("📊 Using advanced batch analysis with Gemini 2.5 Pro for superior accuracy")
        
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Company Analysis")
        company_name = st.text_input(
            "Enter company name:",
            placeholder="e.g., Burt's Bees, Tesla, Apple",
            help="Enter the company name you want to analyze for GEO visibility"
        )
        
        # Industry context (optional)
        industry_context = st.text_input(
            "Industry/Product context (optional):",
            placeholder="e.g., lip balm, electric vehicles, smartphones",
            help="Provide additional context to generate more relevant queries"
        )
    
    with col2:
        st.subheader("Actions")
        analyze_button = st.button(
            "🚀 Start Analysis",
            disabled=not company_name or st.session_state.analysis_running,
            use_container_width=True
        )
        
        if st.session_state.analysis_results:
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                if st.button("📄 Export CSV", use_container_width=True):
                    csv_data = export_to_csv(st.session_state.analysis_results)
                    st.download_button(
                        "Download CSV",
                        csv_data,
                        f"geo_analysis_{company_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
            
            with export_col2:
                if st.button("📋 Export JSON", use_container_width=True):
                    json_data = export_to_json(st.session_state.analysis_results)
                    st.download_button(
                        "Download JSON",
                        json_data,
                        f"geo_analysis_{company_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json"
                    )
    
    # Progress section
    progress_container = st.container()
    
    # Run analysis
    if analyze_button and company_name:
        st.session_state.analysis_running = True
        st.rerun()
    
    if st.session_state.analysis_running:
        with progress_container:
            st.info("🔄 Analysis in progress...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize analyzer
            analyzer = GEOAnalyzer(
                openai_api_key=st.session_state.openai_api_key,
                gemini_api_key=st.session_state.get('gemini_api_key'),
                max_concurrent=max_concurrent,
                use_batch_analysis=st.session_state.get('use_batch_analysis', False)
            )
            
            # Run analysis
            try:
                results = asyncio.run(
                    analyzer.analyze_company_visibility(
                        company_name=company_name,
                        industry_context=industry_context,
                        num_queries=num_queries,
                        progress_callback=lambda current, total, status: update_progress(
                            progress_bar, status_text, current, total, status
                        )
                    )
                )
                
                st.session_state.analysis_results = results
                st.session_state.analysis_running = False
                progress_bar.progress(1.0)
                status_text.success("✅ Analysis completed!")
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.session_state.analysis_running = False
                st.error(f"❌ Analysis failed: {str(e)}")
                st.rerun()
    
    # Display results
    if st.session_state.analysis_results and not st.session_state.analysis_running:
        display_results(st.session_state.analysis_results, company_name)

def update_progress(progress_bar, status_text, current, total, status):
    """Update progress bar and status text"""
    progress = current / total if total > 0 else 0
    progress_bar.progress(progress)
    status_text.text(f"{status} ({current}/{total})")

def display_results(results, company_name):
    """Display analysis results with visualizations"""
    st.header("📊 Analysis Results")
    
    # Check if we have valid results
    if not results or not results.get('summary'):
        st.error("❌ No analysis results available. Please try running the analysis again.")
        return
    
    # Check if we have any query results
    total_queries = results.get('summary', {}).get('total_queries', 0)
    if total_queries == 0:
        st.error("❌ No queries were processed successfully. Please check your API keys and try again.")
        return
    
    # Analysis method indicator
    analysis_method = results.get('analysis_method', 'unknown')
    if analysis_method == 'batch_gemini':
        st.success("🤖 Analysis completed using advanced Gemini batch processing")
    elif analysis_method == 'legacy':
        st.info("🔧 Analysis completed using legacy citation tracking")
    elif analysis_method == 'legacy_fallback':
        fallback_reason = results.get('fallback_reason', 'Unknown error')
        st.warning(f"⚠️ Batch analysis failed, used legacy method instead. Reason: {fallback_reason}")
        if "quota" in fallback_reason.lower() or "429" in fallback_reason:
            st.info("💡 Tip: Gemini API quota exceeded. Try again later or upgrade your Gemini plan.")
    
    # Professional Company Performance Dashboard
    st.markdown("### 📊 Company Performance Overview")
    
    # Calculate key metrics
    total_queries = results['summary'].get('total_queries', 0)
    total_citations = results['summary'].get('total_citations', 0)
    citation_rate = (total_citations / total_queries * 100) if total_queries > 0 else 0
    avg_position = results['summary'].get('average_position', 0)
    confidence = results['summary'].get('analysis_confidence', 0.0)
    competitors_count = len(results['summary'].get('competitors', []))
    
    # Clean, professional metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Citation Rate",
            value=f"{citation_rate:.1f}%",
            delta=f"{'Strong' if citation_rate >= 50 else 'Moderate' if citation_rate >= 25 else 'Needs Work'}",
            delta_color="normal" if citation_rate >= 25 else "inverse"
        )
    
    with col2:
        position_display = f"#{int(avg_position)}" if avg_position > 0 else "N/A"
        position_delta = "Excellent" if avg_position <= 3 else "Good" if avg_position <= 5 else "Fair" if avg_position > 0 else None
        st.metric(
            label="Average Position", 
            value=position_display,
            delta=position_delta,
            delta_color="normal" if avg_position <= 5 else "inverse"
        )
    
    with col3:
        st.metric(
            label="Total Citations",
            value=str(total_citations),
            delta=f"out of {total_queries} queries"
        )
    
    with col4:
        st.metric(
            label="Competitors Found",
            value=str(competitors_count),
            delta="Active market" if competitors_count > 5 else "Niche market"
        )
    
    # Executive Summary Card
    if citation_rate >= 50:
        st.success("🏆 **Leader Position** - Excellent AI visibility with strong market presence")
    elif citation_rate >= 25:
        st.info("📈 **Challenger Position** - Good visibility with growth opportunities")
    elif citation_rate >= 10:
        st.warning("⚡ **Emerging Position** - Moderate visibility, significant room for improvement")
    else:
        st.error("🚨 **Invisible Position** - Urgent optimization needed for AI discoverability")
    
    st.divider()
    
    # Detailed visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Competitive Matrix", "AI Timeline", "Query Results", "Competitor Analysis", "Raw Data"])
    
    with tab1:
        st.subheader("🎯 Market Position Analysis")
        st.markdown("*Strategic positioning of your company against key competitors*")
        
        competitors = results['summary'].get('competitors', [])
        if competitors and len(competitors) > 0:
            # Top competitors analysis
            top_competitors = competitors[:8]  # Focus on top 8 for clarity
            
            # Create clean comparison chart
            comp_names = [comp['name'] for comp in top_competitors]
            comp_citations = [comp['mentions'] for comp in top_competitors]
            comp_positions = [comp.get('avg_position', 10) for comp in top_competitors]
            
            # Add your company
            comp_names.insert(0, f"{company_name} (YOU)")
            comp_citations.insert(0, total_citations)
            comp_positions.insert(0, avg_position if avg_position > 0 else 10)
            
            # Create side-by-side comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Citation Frequency Comparison**")
                citation_fig = go.Figure(data=[
                    go.Bar(
                        x=comp_citations,
                        y=comp_names,
                        orientation='h',
                        marker_color=['#FF6B6B' if name.endswith('(YOU)') else '#E8F4FD' for name in comp_names],
                        text=[f"{citations}" for citations in comp_citations],
                        textposition='auto'
                    )
                ])
                citation_fig.update_layout(
                    title="Total Mentions",
                    xaxis_title="Number of Citations",
                    height=400,
                    showlegend=False,
                    margin=dict(l=150, r=50, t=50, b=50)
                )
                st.plotly_chart(citation_fig, use_container_width=True)
            
            with col2:
                st.markdown("**🏆 Average Position Comparison**")
                # Invert positions for better visualization (lower position = better = higher bar)
                inverted_positions = [11 - pos for pos in comp_positions]
                position_fig = go.Figure(data=[
                    go.Bar(
                        x=inverted_positions,
                        y=comp_names,
                        orientation='h',
                        marker_color=['#FF6B6B' if name.endswith('(YOU)') else '#E8F4FD' for name in comp_names],
                        text=[f"#{int(pos)}" if pos <= 10 else "N/A" for pos in comp_positions],
                        textposition='auto'
                    )
                ])
                position_fig.update_layout(
                    title="Ranking Quality (Higher = Better Position)",
                    xaxis_title="Position Score (11 - actual position)",
                    height=400,
                    showlegend=False,
                    margin=dict(l=150, r=50, t=50, b=50)
                )
                st.plotly_chart(position_fig, use_container_width=True)
            
            # Strategic insights
            st.markdown("### 💡 Strategic Insights")
            
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                # Market position analysis
                your_rank_citations = sorted(comp_citations, reverse=True).index(total_citations) + 1
                your_rank_position = sorted([p for p in comp_positions if p <= 10]).index(avg_position) + 1 if avg_position <= 10 else len([p for p in comp_positions if p <= 10]) + 1
                
                st.metric(
                    "Citation Rank",
                    f"#{your_rank_citations}",
                    f"of {len(comp_names)} companies"
                )
                
                st.metric(
                    "Position Rank", 
                    f"#{your_rank_position}" if avg_position <= 10 else "Unranked",
                    f"of {len([p for p in comp_positions if p <= 10])} ranked" if avg_position <= 10 else "Not in top positions"
                )
            
            with insight_col2:
                # Opportunity analysis
                st.markdown("**🎯 Key Opportunities**")
                
                # Find best performing competitor
                best_comp_idx = comp_citations[1:].index(max(comp_citations[1:])) + 1
                best_competitor = comp_names[best_comp_idx]
                best_citations = comp_citations[best_comp_idx]
                
                if best_citations > total_citations:
                    gap = best_citations - total_citations
                    st.info(f"📈 Close gap with **{best_competitor.replace(' (YOU)', '')}** (+{gap} citations needed)")
                
                # Position improvement opportunity
                if avg_position > 3:
                    better_positions = [p for p in comp_positions[1:] if p < avg_position and p > 0]
                    if better_positions:
                        target_pos = min(better_positions)
                        st.info(f"🏆 Target position **#{int(target_pos)}** (currently #{int(avg_position)})")
            
            with insight_col3:
                # Market landscape
                st.markdown("**🌍 Market Landscape**")
                
                total_market_citations = sum(comp_citations)
                market_share = (total_citations / total_market_citations * 100) if total_market_citations > 0 else 0
                
                st.metric(
                    "Market Share",
                    f"{market_share:.1f}%",
                    "of total AI mentions"
                )
                
                if citation_rate >= 50:
                    st.success("🏆 **Market Leader**")
                elif citation_rate >= 25:
                    st.info("📈 **Strong Player**") 
                elif citation_rate >= 10:
                    st.warning("⚡ **Emerging Brand**")
                else:
                    st.error("🚨 **Needs Visibility**")
        else:
            st.info("📊 No competitor data available. Run analysis with more queries to identify market players.")
    
    with tab2:
        st.subheader("📊 Performance Breakdown")
        st.markdown("*Detailed analysis of your AI visibility across different query types*")
        
        if results['query_results']:
            # Analyze queries by type and performance
            query_analysis = []
            query_results = results['query_results']
            
            for i, result in enumerate(query_results):
                cited = result.get('cited', False)
                position = result.get('position', None)
                mention_type = result.get('mention_type', 'none')
                query_text = result.get('query', '')
                
                # Smart query categorization
                query_type = "General"
                if any(word in query_text.lower() for word in ['best', 'top', 'recommended', 'leading']):
                    query_type = "🏆 Rankings & Lists"
                elif any(word in query_text.lower() for word in ['affordable', 'cheap', 'budget', 'price']):
                    query_type = "💰 Price-Focused"
                elif any(word in query_text.lower() for word in ['natural', 'organic', 'sustainable', 'eco']):
                    query_type = "🌿 Natural/Eco"
                elif any(word in query_text.lower() for word in ['review', 'rating', 'opinion', 'feedback']):
                    query_type = "⭐ Reviews & Ratings"
                elif any(word in query_text.lower() for word in ['vs', 'versus', 'compare', 'comparison']):
                    query_type = "⚖️ Comparisons"
                elif any(word in query_text.lower() for word in ['sensitive', 'dry', 'winter', 'daily']):
                    query_type = "🎯 Specific Needs"
                
                query_analysis.append({
                    'query_type': query_type,
                    'cited': cited,
                    'position': position if cited else None,
                    'query_text': query_text
                })
            
            # Create performance summary by category
            df_analysis = pd.DataFrame(query_analysis)
            category_performance = df_analysis.groupby('query_type').agg({
                'cited': ['count', 'sum', 'mean'],
                'position': 'mean'
            }).round(2)
            
            category_performance.columns = ['total_queries', 'citations', 'citation_rate', 'avg_position']
            category_performance['citation_rate'] *= 100
            category_performance = category_performance.reset_index()
            if len(category_performance) > 1:
                # Category performance visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📈 Citation Rate by Query Type**")
                    
                    # Create horizontal bar chart for citation rates
                    citation_fig = go.Figure(data=[
                        go.Bar(
                            x=category_performance['citation_rate'],
                            y=category_performance['query_type'],
                            orientation='h',
                            marker_color='#4DABF7',
                            text=[f"{rate:.0f}%" for rate in category_performance['citation_rate']],
                            textposition='auto'
                        )
                    ])
                    citation_fig.update_layout(
                        title="Citation Success Rate",
                        xaxis_title="Citation Rate (%)",
                        height=350,
                        showlegend=False,
                        margin=dict(l=150, r=50, t=50, b=50)
                    )
                    st.plotly_chart(citation_fig, use_container_width=True)
                
                with col2:
                    st.markdown("**🏆 Average Position by Query Type**")
                    
                    # Filter out categories with no citations for position chart
                    positioned_cats = category_performance[category_performance['avg_position'].notna()]
                    
                    if not positioned_cats.empty:
                        # Invert positions for visualization (lower = better = higher bar)
                        inverted_pos = [11 - pos for pos in positioned_cats['avg_position']]
                        
                        position_fig = go.Figure(data=[
                            go.Bar(
                                x=inverted_pos,
                                y=positioned_cats['query_type'],
                                orientation='h',
                                marker_color='#FF8C42',
                                text=[f"#{pos:.1f}" for pos in positioned_cats['avg_position']],
                                textposition='auto'
                            )
                        ])
                        position_fig.update_layout(
                            title="Average Ranking Position",
                            xaxis_title="Position Quality (Higher = Better)",
                            height=350,
                            showlegend=False,
                            margin=dict(l=150, r=50, t=50, b=50)
                        )
                        st.plotly_chart(position_fig, use_container_width=True)
                    else:
                        st.info("No position data available for ranked queries")
                
                # Performance insights table
                st.markdown("### 📋 Category Performance Summary")
                
                # Create a clean summary table
                summary_data = []
                for _, row in category_performance.iterrows():
                    summary_data.append({
                        "Query Category": row['query_type'],
                        "Total Queries": int(row['total_queries']),
                        "Citations": int(row['citations']),
                        "Citation Rate": f"{row['citation_rate']:.1f}%",
                        "Avg. Position": f"#{row['avg_position']:.1f}" if pd.notna(row['avg_position']) else "N/A",
                        "Performance": "🟢 Strong" if row['citation_rate'] >= 50 else "🟡 Moderate" if row['citation_rate'] >= 25 else "🔴 Weak"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # Strategic recommendations
                st.markdown("### 💡 Strategic Recommendations")
                
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    st.markdown("**🎯 Strongest Performance**")
                    best_category = category_performance.loc[category_performance['citation_rate'].idxmax()]
                    st.success(f"**{best_category['query_type']}** - {best_category['citation_rate']:.1f}% citation rate")
                    if pd.notna(best_category['avg_position']):
                        st.info(f"Average position: #{best_category['avg_position']:.1f}")
                    st.markdown("*Continue optimizing for these query types*")
                
                with rec_col2:
                    st.markdown("**📈 Growth Opportunity**")
                    worst_category = category_performance.loc[category_performance['citation_rate'].idxmin()]
                    if worst_category['citation_rate'] < 25:
                        st.error(f"**{worst_category['query_type']}** - {worst_category['citation_rate']:.1f}% citation rate")
                        st.markdown("*Focus content optimization here*")
                    else:
                        st.info("All categories performing well!")
                        
            else:
                st.info("📊 Run analysis with more diverse queries to see category breakdowns")
                
            # Individual query results summary
            st.markdown("### 📝 Individual Query Results")
            query_summary = []
            for i, qa in enumerate(query_analysis):
                query_summary.append({
                    "Query": qa['query_text'][:60] + "..." if len(qa['query_text']) > 60 else qa['query_text'],
                    "Category": qa['query_type'],
                    "Cited": "✅ Yes" if qa['cited'] else "❌ No",
                    "Position": f"#{int(qa['position'])}" if qa['position'] else "N/A"
                })
            
            query_df = pd.DataFrame(query_summary)
            st.dataframe(query_df, use_container_width=True, hide_index=True)
            
        else:
            st.info("📊 No query data available. Run an analysis to see performance breakdowns.")
    
    with tab3:
        st.subheader("📋 Individual Query Results")
        
        if results['query_results']:
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                show_cited_only = st.checkbox("Show only cited queries", value=False)
            with col2:
                search_term = st.text_input("Search in queries:", placeholder="Enter search term...")
            
            # Display results
            query_results = results['query_results']
            
            # Apply filters to raw data
            filtered_results = query_results
            
            if show_cited_only:
                filtered_results = [r for r in filtered_results if r.get('cited', False)]
            
            if search_term:
                filtered_results = [r for r in filtered_results if search_term.lower() in str(r.get('query', '')).lower()]
            
            for result in filtered_results:
                cited = bool(result.get('cited', False))
                query_text = str(result.get('query', 'Unknown query'))
                response_text = str(result.get('response', 'No response'))
                competitor_positions = result.get('competitor_positions', {})
                with st.expander(f"{'✅' if cited else '❌'} {query_text[:80]}..."):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write("**Response:**")
                        st.write(response_text[:500] + "..." if len(response_text) > 500 else response_text)
                        
                        context = result.get('context')
                        if cited and context:
                            st.write("**Citation Context:**")
                            st.info(str(context))
                        # Show all company positions if available
                        if competitor_positions:
                            st.write("**Company Positions in List:**")
                            st.json(competitor_positions)
                    
                    with col2:
                        st.write("**Details:**")
                        st.write(f"Cited: {'Yes' if cited else 'No'}")
                        position = result.get('position')
                        if position:
                            st.write(f"Position: {position}")
                        
                        # Show additional batch analysis data
                        mention_type = result.get('mention_type')
                        if mention_type:
                            st.write(f"Mention Type: {mention_type}")
                        
                        confidence = result.get('confidence')
                        if confidence:
                            st.write(f"Confidence: {confidence:.2f}")
                        
                        # Show ranking information
                        ranking_type = result.get('ranking_type')
                        if ranking_type:
                            st.write(f"Ranking Type: {ranking_type}")
                        
                        has_ranking = result.get('has_numbered_ranking')
                        if has_ranking:
                            total_in_ranking = result.get('total_companies_in_ranking')
                            if total_in_ranking:
                                st.write(f"Ranking Size: {total_in_ranking} companies")
                        
                        exec_time = result.get('execution_time')
                        if exec_time:
                            st.write(f"Time: {exec_time:.2f}s")
    
    with tab4:
        st.subheader("🏆 Competitor Analysis")
        
        competitors = results['summary'].get('competitors', [])
        if competitors:
            # Competitor mention frequency and average position
            competitor_df = pd.DataFrame([
                {
                    'Competitor': comp['name'],
                    'Mentions': comp['mentions'],
                    'Avg Position': comp.get('avg_position', 0),
                    'Positions': ', '.join(str(p) for p in comp.get('positions', []))
                }
                for comp in competitors
            ])
            # Bar chart: Top 10 competitors by average position (lower is better)
            top10 = competitor_df.head(10)
            fig_competitors = px.bar(
                top10,
                x='Competitor',
                y='Avg Position',
                title="Top 10 Competitors by Average Ranking Position (Lower is Better)",
                labels={'Avg Position': 'Average Position'},
                color='Avg Position',
                color_continuous_scale='RdYlGn_r',  # Red (worse position) to Green (better position)
                hover_data=['Mentions', 'Positions']
            )
            fig_competitors.update_xaxes(tickangle=45)
            st.plotly_chart(fig_competitors, use_container_width=True)
            # Table: All competitors with positions
            st.write("**All Competitors:**")
            st.dataframe(competitor_df, use_container_width=True)
        else:
            st.info("No competitor data available from the analysis.")
    
    with tab5:
        st.subheader("📊 Raw Analysis Data")
        
        # Summary data
        st.write("**Analysis Summary:**")
        st.json(results['summary'])
        
        # Query results data
        if results['query_results']:
            st.write("**Query Results:**")
            df = pd.DataFrame(results['query_results'])
            st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
