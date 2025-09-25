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
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        num_queries = st.slider("Number of queries", min_value=2, max_value=20, value=5, step=1)
        max_concurrent = st.slider("Max concurrent requests", min_value=1, max_value=10, value=5)
        
        if use_batch_analysis:
            st.info("📊 Using advanced batch analysis with Gemini for better accuracy")
        
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
    
    # Create a beautiful main performance card
    performance_card = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    ">
        <div style="text-align: center;">
            <h2 style="margin: 0; font-size: 28px; font-weight: 300;">{company_name}</h2>
            <p style="margin: 5px 0 20px 0; opacity: 0.9; font-size: 16px;">AI Visibility Performance</p>
            
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px;">
                <div style="text-align: center; min-width: 120px;">
                    <div style="font-size: 36px; font-weight: 700; margin-bottom: 5px;">{citation_rate:.1f}%</div>
                    <div style="font-size: 14px; opacity: 0.8;">Citation Rate</div>
                </div>
                <div style="text-align: center; min-width: 120px;">
                    <div style="font-size: 36px; font-weight: 700; margin-bottom: 5px;">{"#" + str(int(avg_position)) if avg_position > 0 else "N/A"}</div>
                    <div style="font-size: 14px; opacity: 0.8;">Avg. Position</div>
                </div>
                <div style="text-align: center; min-width: 120px;">
                    <div style="font-size: 36px; font-weight: 700; margin-bottom: 5px;">{total_citations}</div>
                    <div style="font-size: 14px; opacity: 0.8;">Total Citations</div>
                </div>
                <div style="text-align: center; min-width: 120px;">
                    <div style="font-size: 36px; font-weight: 700; margin-bottom: 5px;">{competitors_count}</div>
                    <div style="font-size: 14px; opacity: 0.8;">Competitors</div>
                </div>
            </div>
        </div>
    </div>
    """
    
    st.markdown(performance_card, unsafe_allow_html=True)
    
    # Performance insights
    if citation_rate >= 50:
        st.success("🏆 **Excellent Visibility** - Your company has strong presence in AI recommendations")
    elif citation_rate >= 25:
        st.warning("⚡ **Growing Presence** - Good visibility with room for improvement")
    else:
        st.error("📈 **Optimization Needed** - Low visibility presents significant growth opportunity")
    
    # Detailed visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Competitive Matrix", "AI Timeline", "Query Results", "Competitor Analysis", "Raw Data"])
    
    with tab1:
        st.subheader("🎯 Competitive Positioning Matrix")
        st.markdown("*Position your company against competitors in the AI recommendation landscape*")
        
        competitors = results['summary'].get('competitors', [])
        if competitors and len(competitors) > 0:
            # Prepare data for positioning matrix
            matrix_data = []
            
            # Add target company
            matrix_data.append({
                'Company': company_name,
                'Citation_Frequency': citation_rate,
                'Avg_Position': avg_position if avg_position > 0 else 10,
                'Total_Mentions': total_citations,
                'Is_Target': True,
                'Position_Inverted': 11 - (avg_position if avg_position > 0 else 10)  # Invert for better visualization
            })
            
            # Add competitors
            for comp in competitors[:15]:  # Limit to top 15 for readability
                comp_citation_rate = (comp['mentions'] / total_queries * 100) if total_queries > 0 else 0
                comp_avg_pos = comp.get('avg_position', 10)
                if comp_avg_pos == 0:
                    comp_avg_pos = 10
                
                matrix_data.append({
                    'Company': comp['name'],
                    'Citation_Frequency': comp_citation_rate,
                    'Avg_Position': comp_avg_pos,
                    'Total_Mentions': comp['mentions'],
                    'Is_Target': False,
                    'Position_Inverted': 11 - comp_avg_pos
                })
            
            df_matrix = pd.DataFrame(matrix_data)
            
            # Create the positioning matrix
            fig_matrix = px.scatter(
                df_matrix,
                x='Citation_Frequency',
                y='Position_Inverted',
                size='Total_Mentions',
                color='Is_Target',
                hover_name='Company',
                hover_data={'Citation_Frequency': ':.1f%', 'Avg_Position': True, 'Total_Mentions': True},
                title="Competitive Positioning Matrix",
                labels={
                    'Citation_Frequency': 'Citation Frequency (%)',
                    'Position_Inverted': 'Ranking Quality (Higher = Better Position)',
                    'Total_Mentions': 'Total Mentions'
                },
                color_discrete_map={True: '#FF6B6B', False: '#4DABF7'},
                size_max=60
            )
            
            # Customize the matrix
            fig_matrix.update_layout(
                width=800,
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add quadrant lines
            max_citation = df_matrix['Citation_Frequency'].max()
            max_position = df_matrix['Position_Inverted'].max()
            
            fig_matrix.add_hline(y=max_position/2, line_dash="dash", line_color="gray", opacity=0.5)
            fig_matrix.add_vline(x=max_citation/2, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Add quadrant labels
            fig_matrix.add_annotation(x=max_citation*0.75, y=max_position*0.75, text="Leaders", 
                                    showarrow=False, font=dict(size=14, color="gray"))
            fig_matrix.add_annotation(x=max_citation*0.25, y=max_position*0.75, text="High Quality", 
                                    showarrow=False, font=dict(size=14, color="gray"))
            fig_matrix.add_annotation(x=max_citation*0.75, y=max_position*0.25, text="High Visibility", 
                                    showarrow=False, font=dict(size=14, color="gray"))
            fig_matrix.add_annotation(x=max_citation*0.25, y=max_position*0.25, text="Emerging", 
                                    showarrow=False, font=dict(size=14, color="gray"))
            
            st.plotly_chart(fig_matrix, use_container_width=True)
            
            # Strategic insights
            your_position = df_matrix[df_matrix['Is_Target'] == True].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Your Position:**")
                if your_position['Citation_Frequency'] > max_citation/2 and your_position['Position_Inverted'] > max_position/2:
                    st.success("**Leader Quadrant** - Strong visibility and positioning")
                elif your_position['Citation_Frequency'] < max_citation/2 and your_position['Position_Inverted'] > max_position/2:
                    st.info("**High Quality** - Good positioning, need more visibility")
                elif your_position['Citation_Frequency'] > max_citation/2 and your_position['Position_Inverted'] < max_position/2:
                    st.warning("**High Visibility** - Frequently mentioned but lower positions")
                else:
                    st.error("**Emerging** - Opportunity for growth in both areas")
            
            with col2:
                st.markdown("**📈 Strategic Opportunities:**")
                leaders = df_matrix[(df_matrix['Citation_Frequency'] > max_citation/2) & 
                                  (df_matrix['Position_Inverted'] > max_position/2) & 
                                  (df_matrix['Is_Target'] == False)]
                if not leaders.empty:
                    top_competitor = leaders.iloc[0]
                    st.info(f"📊 Study **{top_competitor['Company']}** strategy")
                    st.info(f"🎯 Target citation rate: {top_competitor['Citation_Frequency']:.1f}%")
        else:
            st.info("No competitor data available for positioning matrix.")
    
    with tab2:
        st.subheader("🕐 AI Recommendation Timeline & Positioning")
        st.markdown("*Track how your company's positioning evolves across different query contexts*")
        
        if results['query_results']:
            # Prepare timeline data
            query_results = results['query_results']
            timeline_data = []
            
            for i, result in enumerate(query_results):
                cited = result.get('cited', False)
                position = result.get('position', None)
                mention_type = result.get('mention_type', 'none')
                ranking_type = result.get('ranking_type', '')
                query_text = result.get('query', '')
                
                # Categorize query types
                query_category = "General"
                if any(word in query_text.lower() for word in ['price', 'cheap', 'affordable', 'budget']):
                    query_category = "Price-Focused"
                elif any(word in query_text.lower() for word in ['natural', 'organic', 'clean', 'pure']):
                    query_category = "Natural/Organic"
                elif any(word in query_text.lower() for word in ['premium', 'luxury', 'high-end', 'best']):
                    query_category = "Premium"
                elif any(word in query_text.lower() for word in ['top', 'leading', 'popular', 'trending']):
                    query_category = "Popularity"
                
                # Determine positioning sentiment
                context = result.get('context', '')
                sentiment = "Neutral"
                if context:
                    if any(word in context.lower() for word in ['leading', 'top', 'best', 'excellent', 'premium']):
                        sentiment = "Positive"
                    elif any(word in context.lower() for word in ['alternative', 'cheaper', 'budget', 'option']):
                        sentiment = "Alternative"
                    elif any(word in context.lower() for word in ['natural', 'organic', 'clean']):
                        sentiment = "Natural"
                
                timeline_data.append({
                    'Query_Index': i + 1,
                    'Query_Category': query_category,
                    'Position': position if position else 11,
                    'Cited': cited,
                    'Mention_Type': mention_type,
                    'Sentiment': sentiment,
                    'Query_Text': query_text[:60] + "..." if len(query_text) > 60 else query_text,
                    'Position_Inverted': 11 - position if position else 0
                })
            
            df_timeline = pd.DataFrame(timeline_data)
            cited_df = df_timeline[df_timeline['Cited'] == True]
            
            if not cited_df.empty:
                # Position trend over query sequence
                fig_timeline = px.line(
                    cited_df,
                    x='Query_Index',
                    y='Position_Inverted',
                    color='Query_Category',
                    title="Position Performance Across Query Types",
                    labels={
                        'Query_Index': 'Query Sequence',
                        'Position_Inverted': 'Ranking Quality (Higher = Better)',
                        'Query_Category': 'Query Type'
                    },
                    markers=True
                )
                
                fig_timeline.update_layout(height=400)
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Positioning heatmap by category and sentiment
                if len(cited_df) > 1:
                    heatmap_data = cited_df.groupby(['Query_Category', 'Sentiment']).agg({
                        'Position': 'mean',
                        'Query_Index': 'count'
                    }).round(1)
                    
                    if not heatmap_data.empty:
                        fig_heatmap = px.imshow(
                            heatmap_data['Position'].unstack(fill_value=0),
                            title="Average Position by Query Type & Sentiment Context",
                            labels=dict(x="Sentiment Context", y="Query Category", color="Avg Position"),
                            color_continuous_scale="RdYlGn_r"
                        )
                        fig_heatmap.update_layout(height=400)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Insights panel
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🎯 Best Performing Categories:**")
                    best_categories = cited_df.groupby('Query_Category')['Position_Inverted'].mean().sort_values(ascending=False)
                    for cat, score in best_categories.head(3).items():
                        st.success(f"**{cat}**: {11-score:.1f} avg position")
                
                with col2:
                    st.markdown("**💡 Positioning Context:**")
                    sentiment_counts = cited_df['Sentiment'].value_counts()
                    for sentiment, count in sentiment_counts.items():
                        if sentiment != "Neutral":
                            st.info(f"**{sentiment}**: {count} mentions")
                
                with col3:
                    st.markdown("**📈 Trends:**")
                    if len(cited_df) >= 3:
                        first_half = cited_df.head(len(cited_df)//2)['Position'].mean()
                        second_half = cited_df.tail(len(cited_df)//2)['Position'].mean()
                        trend = "↗️ Improving" if second_half < first_half else "↘️ Declining" if second_half > first_half else "➡️ Stable"
                        st.metric("Position Trend", trend)
            else:
                st.warning("No citations found to analyze positioning trends.")
        else:
            st.info("No query results available for timeline analysis.")
    
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
                color_continuous_scale='Blues_r',
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
