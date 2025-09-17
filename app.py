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
        
        # API Key check
        api_key_status = st.empty()
        if not st.session_state.get('openai_api_key'):
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                st.session_state.openai_api_key = api_key
                api_key_status.success("✅ OpenAI API Key loaded")
            else:
                api_key_status.error("❌ OpenAI API Key not found in environment")
                st.stop()
        else:
            api_key_status.success("✅ OpenAI API Key loaded")
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        num_queries = st.slider("Number of queries", min_value=2, max_value=20, value=5, step=1)
        max_concurrent = st.slider("Max concurrent requests", min_value=1, max_value=10, value=5)
        
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
                api_key=st.session_state.openai_api_key,
                max_concurrent=max_concurrent
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
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        citation_rate = (results['summary']['total_citations'] / results['summary']['total_queries']) * 100
        st.metric(
            "Citation Rate",
            f"{citation_rate:.1f}%",
            help="Percentage of queries where the company was mentioned"
        )
    
    with col2:
        st.metric(
            "Total Citations",
            results['summary']['total_citations'],
            help="Total number of times the company was mentioned"
        )
    
    with col3:
        avg_position = results['summary'].get('average_position', 0)
        st.metric(
            "Avg. Position",
            f"{avg_position:.1f}" if avg_position > 0 else "N/A",
            help="Average ranking position when mentioned"
        )
    
    with col4:
        st.metric(
            "Competitor Mentions",
            len(results['summary'].get('competitors', [])),
            help="Number of unique competitors mentioned"
        )
    
    # Detailed visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Citation Analysis", "Query Results", "Competitor Analysis", "Raw Data"])
    
    with tab1:
        st.subheader("Citation Frequency Analysis")
        
        # Citation distribution chart
        if results['query_results']:
            df = pd.DataFrame(results['query_results'])
            
            # Citation frequency by query type
            fig_citations = px.histogram(
                df,
                x='cited',
                title="Citation Distribution",
                labels={'cited': 'Company Cited', 'count': 'Number of Queries'},
                color='cited',
                color_discrete_map={True: '#2E8B57', False: '#DC143C'}
            )
            st.plotly_chart(fig_citations, use_container_width=True)
            
            # Position analysis for cited queries
            cited_df = df[df['cited'] == True]
            if not cited_df.empty and 'position' in cited_df.columns:
                fig_positions = px.histogram(
                    cited_df,
                    x='position',
                    title="Ranking Positions When Cited",
                    labels={'position': 'Position in Results', 'count': 'Frequency'},
                    nbins=10
                )
                st.plotly_chart(fig_positions, use_container_width=True)
    
    with tab2:
        st.subheader("Individual Query Results")
        
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
                
                with st.expander(f"{'✅' if cited else '❌'} {query_text[:80]}..."):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write("**Response:**")
                        st.write(response_text[:500] + "..." if len(response_text) > 500 else response_text)
                        
                        context = result.get('context')
                        if cited and context:
                            st.write("**Citation Context:**")
                            st.info(str(context))
                    
                    with col2:
                        st.write("**Details:**")
                        st.write(f"Cited: {'Yes' if cited else 'No'}")
                        position = result.get('position')
                        if position:
                            st.write(f"Position: {position}")
                        exec_time = result.get('execution_time')
                        if exec_time:
                            st.write(f"Time: {exec_time:.2f}s")
    
    with tab3:
        st.subheader("Competitor Analysis")
        
        competitors = results['summary'].get('competitors', [])
        if competitors:
            # Competitor mention frequency
            competitor_df = pd.DataFrame([
                {'Competitor': comp['name'], 'Mentions': comp['mentions'], 'Avg Position': comp.get('avg_position', 0)}
                for comp in competitors
            ])
            
            fig_competitors = px.bar(
                competitor_df.head(10),
                x='Competitor',
                y='Mentions',
                title="Top 10 Competitors by Mention Frequency",
                labels={'Mentions': 'Number of Mentions'}
            )
            fig_competitors.update_xaxes(tickangle=45)
            st.plotly_chart(fig_competitors, use_container_width=True)
            
            # Competitor table
            st.write("**All Competitors:**")
            st.dataframe(competitor_df, use_container_width=True)
        else:
            st.info("No competitor data available from the analysis.")
    
    with tab4:
        st.subheader("Raw Analysis Data")
        
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
