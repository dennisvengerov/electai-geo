# Overview

GEO Analytics Dashboard is a Streamlit-based application for analyzing Generative Engine Optimization (GEO) performance. The system tests company visibility across semantic search queries by generating diverse prompts, submitting them to AI language models, and analyzing whether companies are cited in the responses. It provides comprehensive analytics on citation rates, positioning, and competitive landscape analysis.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Web Interface**: Single-page dashboard with sidebar configuration panel
- **Real-time Progress Tracking**: Live updates during analysis execution using session state management
- **Interactive Visualizations**: Plotly-based charts and graphs for data presentation
- **Export Functionality**: CSV and JSON export capabilities for analysis results

## Core Analysis Engine
- **Modular Design**: Separated concerns across four main components:
  - `GEOAnalyzer`: Main orchestrator handling concurrent query execution
  - `PromptGenerator`: AI-powered semantic query generation using OpenAI API
  - `CitationTracker`: Pattern-matching engine for company mention detection
  - `utils`: Data export and formatting utilities

## Data Processing Pipeline
- **Asynchronous Processing**: Concurrent query execution with configurable limits to optimize performance
- **Batch Processing**: Query generation in batches for better organization and rate limiting
- **Real-time Analytics**: Live calculation of citation rates, positioning metrics, and execution times
- **Result Aggregation**: Comprehensive summary statistics with competitive analysis

## Analysis Methodology
- **Semantic Query Generation**: Uses GPT models to create diverse, contextually relevant queries
- **Citation Detection**: Regular expression-based pattern matching for company mentions
- **Competitive Analysis**: Automatic extraction of competing companies mentioned in responses
- **Position Tracking**: Ranking detection when companies appear in lists or comparisons

# External Dependencies

## AI Services
- **OpenAI API**: Primary service for query generation and semantic analysis
- **AsyncOpenAI Client**: Asynchronous API client for concurrent request handling

## Data Processing Libraries
- **Pandas**: Data manipulation and analysis framework
- **Plotly**: Interactive visualization library for charts and graphs

## Web Framework
- **Streamlit**: Web application framework providing the dashboard interface
- **aiohttp**: Asynchronous HTTP client library for API communications

## Utilities
- **Standard Library**: Uses asyncio for concurrency, json for data serialization, csv for exports, and re for pattern matching