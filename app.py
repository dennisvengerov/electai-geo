import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import json
from datetime import datetime
import time
import os
from dotenv import load_dotenv
import base64
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

from geo_analyzer import GEOAnalyzer
from utils import export_to_csv, export_to_json

# Page configuration
st.set_page_config(
    page_title="GEO Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Remove default Streamlit padding
st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-left: 5rem !important;
            padding-right: 5rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'company_name' not in st.session_state:
    st.session_state.company_name = ""
if 'industry_context' not in st.session_state:
    st.session_state.industry_context = ""

# --- Asset Loading & CSS Injection ---
def load_asset(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        st.error(f"Asset not found: {file_path}")
        return ""

def inject_custom_css():
    css_path = Path(__file__).parent / "assets" / "custom.css"
    bg_image_path = Path(__file__).parent / "assets" / "background.png"
    
    bg_image_b64 = load_asset(bg_image_path)
    if not bg_image_b64: return

    try:
        with open(css_path) as f:
            css = f.read().replace("{bg_image}", bg_image_b64)
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("assets/custom.css not found.")

def main():
    inject_custom_css()
    st.title("ElectAI GEO")

    # --- Progressive UI Flow ---
    company_name = st.text_input(
        "Enter Company Name to Begin",
        key="company_name_input",
        placeholder="e.g., Tesla, Apple, Burt's Bees"
    )

    if company_name:
        st.session_state.company_name = company_name
        industry_context = st.text_input(
            "Industry/Product Context",
            key="industry_context_input",
            placeholder="e.g., electric vehicles, smartphones, lip balm"
        )
        st.session_state.industry_context = industry_context

    if company_name and st.session_state.industry_context:
        num_queries = st.slider(
            "Number of Queries to Analyze",
            min_value=5, max_value=50, value=10, step=5
        )
        if st.button("🚀 Start Analysis", use_container_width=True, key="start_analysis"):
            st.session_state.analysis_running = True
            st.session_state.num_queries = num_queries
            st.rerun()

    # --- Analysis & Results ---
    if st.session_state.analysis_running:
        st.set_page_config(page_title=f"Analyzing {st.session_state.company_name}...")
        run_analysis(
            st.session_state.company_name, 
            st.session_state.industry_context,
            st.session_state.num_queries
        )
    
    if st.session_state.analysis_results:
        st.set_page_config(page_title=f"{st.session_state.company_name} - GEO Results")
        display_results(st.session_state.analysis_results, st.session_state.company_name)

def run_analysis(company_name, industry_context, num_queries):
    """Handles the analysis process and updates session state."""
    
    # Custom animated loading indicator (no white card)
    st.markdown("""
        <div class="loader-container">
            <div class="pulsing-orb"></div>
            <p class="loading-text">⚡ Electrifying Insights...</p>
        </div>
    """, unsafe_allow_html=True)

    progress_bar = st.progress(0, text="Initializing...")
    def update_progress_callback(current, total, status):
        progress_bar.progress(current / total if total > 0 else 0, text=f"{status} ({current}/{total})")

    try:
        analyzer = GEOAnalyzer(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            max_concurrent=min(10, num_queries),
            use_batch_analysis=True
        )
        results = asyncio.run(analyzer.analyze_company_visibility(
            company_name=company_name,
            industry_context=industry_context,
            num_queries=num_queries,
            progress_callback=update_progress_callback
        ))
        st.session_state.analysis_results = results
    except Exception as e:
        st.error(f"Analysis failed: {e}")
    finally:
        st.session_state.analysis_running = False
        st.rerun()
    
def display_results(results, company_name):
    """Display analysis results with visualizations"""
    st.header(f"📊 Analysis Results for: {company_name}")
    
    # Check if we have valid results
    if not results or not results.get('summary'):
        st.error("❌ No analysis results available. Please try running the analysis again.")
        return
    
    # Check if we have any query results
    total_queries = results.get('summary', {}).get('total_queries', 0)
    if total_queries == 0:
        st.error("❌ No queries were processed successfully. Please check your API keys and try again.")
        return
    
    summary = results.get('summary', {})
    total_citations = summary.get('total_citations', 0)
    avg_position = summary.get('average_position', 0)
    competitors_count = len(summary.get('competitors', []))
    citation_rate = (total_citations / total_queries * 100) if total_queries > 0 else 0

    # --- Metrics & Actions ---
    cols = st.columns([1, 1, 1, 1, 1.5])
    with cols[0]:
        st.metric("Citation Rate", f"{citation_rate:.1f}%")
    with cols[1]:
        st.metric("Avg. Position", f"#{avg_position:.1f}" if avg_position > 0 else "N/A")
    with cols[2]:
        st.metric("Total Citations", f"{total_citations}")
    with cols[3]:
        st.metric("Competitors", f"{competitors_count}")
    
    with cols[4]:
        # Share Button Logic
        visibility_score = ((11 - avg_position) if avg_position > 0 else 1) * total_citations
        share_text = (f"ElectAI GEO Analysis for {company_name}:\n"
                      f"- Visibility Score: {visibility_score:,.0f}\n"
                      f"- Citation Rate: {citation_rate:.1f}%\n"
                      f"- Avg. Position: #{avg_position:.1f}")
        st.code(share_text, language=None) # Using st.code for easy copy-paste
        
    st.divider()

    # Detailed visualizations
    tabs = st.tabs(["🏆 Competitive Matrix", "📊 Query Performance", "📄 Raw Data"])
    
    with tabs[0]:
        st.subheader("Market Visibility Leaderboard")
        competitors = results.get('summary', {}).get('competitors', [])

        if not competitors:
            st.info("No competitor data identified.")
        else:
            # Calculate Visibility Score
            data = []
            user_avg_pos = results['summary'].get('average_position', 0)
            user_mentions = results['summary'].get('total_citations', 0)
            user_score = ((11 - user_avg_pos) if user_avg_pos > 0 else 1) * user_mentions
            data.append({
                'Competitor': f"⭐ {company_name} (You)", 
                'Visibility Score': user_score,
                'Mentions': user_mentions,
                'Avg. Position': f"#{user_avg_pos:.1f}" if user_avg_pos > 0 else "N/A"
            })

            for comp in competitors:
                avg_pos = comp.get('avg_position', 0)
                mentions = comp.get('mentions', 0)
                score = ((11 - avg_pos) if avg_pos > 0 else 1) * mentions
                data.append({
                    'Competitor': comp['name'], 
                    'Visibility Score': score,
                    'Mentions': mentions,
                    'Avg. Position': f"#{avg_pos:.1f}" if avg_pos > 0 else "N/A"
                })

            df = pd.DataFrame(data).sort_values('Visibility Score', ascending=False).reset_index(drop=True)
            
            # Chart
            fig = px.bar(
                df.head(10).sort_values('Visibility Score', ascending=True),
                x='Visibility Score', y='Competitor', orientation='h',
                color='Visibility Score', color_continuous_scale='Oranges',
                text='Visibility Score', title="🏆 Top 10 Competitors by Visibility"
            )
            fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.dataframe(df, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Performance by Search Query")
        query_results = results.get('query_results', [])
        if not query_results:
            st.info("No query results to display.")
        else:
            for result in query_results:
                icon = "✅" if result.get('cited') else "❌"
                with st.expander(f"{icon} {result.get('query', 'N/A')}"):
                    st.json(result)

    with tabs[2]:
        st.subheader("Complete Analysis Data (JSON)")
        st.json(results)
    
if __name__ == "__main__":
    main()
