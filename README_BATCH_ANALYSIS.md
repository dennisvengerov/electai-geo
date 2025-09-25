# Batch Analysis Implementation

## Overview

This implementation replaces the inefficient citation tracking system with a modern batch analysis approach using Gemini's 1M context window. The new system provides more accurate citation detection and competitor extraction.

## Key Improvements

### 1. **Accuracy Improvements**
- **LLM-based analysis** instead of regex patterns
- **Context-aware citation detection** (handles "the Cupertino company" = Apple)
- **Sophisticated competitor identification** across multiple query responses
- **Confidence scoring** for each analysis

### 2. **Performance Benefits**
- **Batch processing**: Analyze 50+ queries in a single API call
- **Reduced API costs**: Fewer total API calls than individual analysis
- **Better rate limit handling**: Single Gemini call vs many individual analyses

### 3. **Enhanced Analytics**
- **Cross-query insights**: Patterns visible only when analyzing multiple results together
- **Mention type classification**: Direct, indirect, implied mentions
- **Competitive landscape analysis**: Market leaders, frequent comparisons
- **Analysis confidence metrics**: Data quality indicators

## Architecture Changes

### New Components

1. **`data_models.py`**: New data structures for batch processing
   - `RawQueryResult`: Raw query responses before analysis
   - `QueryAnalysis`: Individual query analysis results
   - `BatchAnalysisResult`: Complete batch analysis output

2. **`batch_analyzer.py`**: Gemini-based batch analysis engine
   - Handles batching for large query sets
   - Sophisticated prompt engineering for accurate analysis
   - Error handling and fallback mechanisms

3. **Updated `geo_analyzer.py`**: Dual-mode analysis support
   - New batch analysis pipeline
   - Legacy analysis fallback
   - Seamless integration with existing UI

### Modified Components

1. **`app.py`**: Enhanced UI with analysis method selection
   - Gemini API key detection
   - Analysis method toggle (Batch vs Legacy)
   - Enhanced result display with confidence metrics

2. **`pyproject.toml`**: Added Gemini dependencies
   - `google-generativeai>=0.8.0`
   - Updated spaCy for better NER

## Usage

### Setup

1. **Install dependencies**:
   ```bash
   pip install google-generativeai>=0.8.0
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export GEMINI_API_KEY="your-gemini-key"  # Optional for batch analysis
   ```

### Running Analysis

#### Via Streamlit App
```bash
streamlit run app.py
```
- Select "Batch Analysis (Gemini)" in the sidebar for enhanced analysis
- Falls back to "Legacy Analysis" if Gemini API key not available

#### Programmatic Usage
```python
from geo_analyzer import GEOAnalyzer

# Initialize with both API keys
analyzer = GEOAnalyzer(
    openai_api_key="your-openai-key",
    gemini_api_key="your-gemini-key",  # Optional
    use_batch_analysis=True
)

# Run analysis
results = await analyzer.analyze_company_visibility(
    company_name="Your Company",
    industry_context="your industry",
    num_queries=10
)
```

### Testing

Run the test script to verify batch analysis:
```bash
python test_batch_analysis.py
```

## Analysis Flow

### New Batch Analysis Pipeline

1. **Query Generation**: OpenAI GPT-5 generates diverse semantic queries
2. **Response Collection**: Concurrent collection of raw AI responses
3. **Batch Analysis**: Single Gemini call analyzes all responses together
4. **Result Processing**: Convert to legacy format for UI compatibility

### Legacy Pipeline (Fallback)

1. **Query Generation**: Same as batch analysis
2. **Individual Analysis**: Each response analyzed separately
3. **Citation Tracking**: Regex-based pattern matching
4. **Result Aggregation**: Manual aggregation of individual results

## Batch Analysis Prompt

The Gemini prompt includes:
- **Target company identification**
- **Citation detection** (direct, indirect, implied)
- **Position tracking** in ranked lists
- **Competitor extraction** across all responses
- **Context extraction** around mentions
- **Cross-query pattern analysis**

## Configuration Options

### Analysis Parameters
- `num_queries`: Number of queries to generate (2-20)
- `max_concurrent`: Concurrent OpenAI requests (1-10)
- `max_batch_size`: Queries per Gemini batch (default: 50)

### Environment Variables
- `OPENAI_API_KEY`: Required for query generation
- `GEMINI_API_KEY`: Optional for batch analysis
- `STREAMLIT_SERVER_PORT`: Custom port for Streamlit

## Migration Strategy

The implementation supports both analysis methods:

1. **Phase 1**: Deploy with dual-mode support
2. **Phase 2**: Users can toggle between batch and legacy analysis
3. **Phase 3**: Monitor accuracy and performance improvements
4. **Phase 4**: Deprecate legacy method once confidence is high

## API Rate Limits

### Batch Analysis
- **Gemini calls**: 1 call per 50 queries
- **OpenAI calls**: Same as legacy (query generation only)

### Legacy Analysis
- **OpenAI calls**: 1 call per query + 1 call per validation
- **Higher total API usage**

## Error Handling

- **Gemini API failures**: Automatic fallback to legacy analysis
- **Batch size optimization**: Automatic splitting for large query sets
- **Individual query errors**: Isolated error handling per query
- **JSON parsing errors**: Graceful degradation with error reporting

## Future Enhancements

1. **Advanced batching**: Dynamic batch size optimization
2. **Multi-model support**: Additional LLM providers
3. **Real-time analysis**: Streaming results for large datasets
4. **Custom prompts**: User-configurable analysis prompts
5. **Export enhancements**: Advanced reporting formats

## Troubleshooting

### Common Issues

1. **Gemini API key not working**:
   - Verify key is correctly set in environment
   - Check API quotas and billing

2. **Batch analysis fails**:
   - System automatically falls back to legacy analysis
   - Check logs for specific error messages

3. **Inconsistent results**:
   - Batch analysis uses low temperature (0.1) for consistency
   - Legacy analysis may show different patterns due to individual processing

### Debug Mode

Set environment variable for verbose logging:
```bash
export DEBUG_BATCH_ANALYSIS=1
```
