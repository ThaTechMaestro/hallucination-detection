# LLM Judge

**Sentence-level hallucination detection using judge LLMs with comprehensive audit trails**

A component of the [Hallucination Detector](../README.md) that leverages large language models (OpenAI GPT or Anthropic Claude) to evaluate the factual groundedness of AI-generated responses at the sentence level.

## Overview

LLM Judge breaks down AI responses into individual sentences and scores each one for hallucination likelihood by comparing against provided context. It provides granular insights into exactly which parts of a response are problematic, making it ideal for systems that need detailed explanations and audit trails.

### Key Features

- **Sentence-Level Precision**: Identifies exactly which sentences contain hallucinations
- **Multiple Judge Models**: Support for OpenAI GPT and Anthropic Claude
- **Production Ready**: Robust error handling, retry logic, and comprehensive logging
- **Complete Audit Trail**: Full traceability from input to output with data integrity guarantees
- **Weighted Scoring**: Token-aware aggregation for more accurate overall assessment
- **Deterministic**: Reproducible results with consistent sentence splitting
- **Scalable**: Handles large datasets with batch processing and progress tracking

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install openai anthropic python-dotenv tiktoken

# Configure API keys
echo "OPENAI_API_KEY=your_key_here" > .env
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
```

### 2. Prepare Data

Create a JSONL file with your evaluation data:

```jsonl
{"id": "q1", "question": "What is the capital of France?", "context": "France is a country in Europe. Paris is the capital and largest city of France.", "response": "The capital of France is Paris, which is also its largest city."}
{"id": "q2", "question": "How tall is Mount Everest?", "context": "Mount Everest is 8,848.86 meters tall.", "response": "Mount Everest is approximately 9,000 meters tall and is located in the Himalayas."}
```

### 3. Run Evaluation

```bash
python main.py --input data.jsonl --provider openai --model gpt-4o-mini
```

### 4. View Results

```bash
ls judge_outputs/
# results.jsonl - Detailed sentence-level results
# summary.csv - Quick overview
# processing_summary.json - Complete audit trail
```

## How It Works

### 1. Sentence Segmentation
```python
# Input response
"The capital of France is Paris. It has a population of 10 million people."

# Segmented sentences
["The capital of France is Paris.", "It has a population of 10 million people."]
```

### 2. Judge LLM Scoring
Each sentence gets scored 0.0-1.0:
- **0.0**: Fully grounded in context
- **0.5**: Partially supported
- **1.0**: Complete hallucination

### 3. Weighted Aggregation
```python
# Token weights: [8, 12] (by sentence length)
# Scores: [0.1, 0.8] (grounded, hallucinated)
# Overall: (8×0.9 + 12×0.2) / 20 = 0.48 → "HALLUCINATION"
```

### 4. Decision & Labels
- **Overall Score ≥ 0.80**: "FACT" 
- **Overall Score < 0.80**: "HALLUCINATION"
- Individual sentences labeled as: Grounded, Partially Supported, Unsupported, Refuted

## Usage

### Command Line

```bash
# Basic usage
python main.py --input data.jsonl

# Custom configuration
python main.py \
  --input large_dataset.jsonl \
  --provider anthropic \
  --model claude-3-sonnet-20240229 \
  --tau 0.75 \
  --out-dir custom_output \
  --max-rows 1000 \
  --timeout 120

# Available options
python main.py --help
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | `product_info.jsonl` | Input JSONL file path |
| `--provider` | `openai` | Judge LLM provider (`openai` or `anthropic`) |
| `--model` | `gpt-4o-mini` | Model name to use |
| `--tau` | `0.80` | Groundedness threshold (0.0-1.0) |
| `--out-dir` | `./judge_outputs` | Output directory |
| `--max-rows` | `0` | Maximum rows to process (0 = all) |
| `--timeout` | `60` | LLM API timeout in seconds |

### Programmatic API

```python
from llm_judge import evaluate_jsonl, evaluate_row, get_provider

# Batch processing
results, stats = evaluate_jsonl(
    input_path="data.jsonl",
    provider="openai",
    model="gpt-4o-mini",
    tau=0.80,
    out_dir="./outputs"
)

# Single evaluation
provider = get_provider("openai", "gpt-4o-mini")
row = {
    "id": "test1",
    "question": "What is machine learning?",
    "context": "Machine learning is a subset of AI that enables computers to learn from data.",
    "response": "Machine learning allows computers to automatically improve through experience."
}

result = evaluate_row(row, provider, tau=0.80)
print(f"Decision: {result.decision}")
print(f"Overall Groundedness: {result.overall_groundedness:.3f}")
```

## Input Format

### Required Fields

```json
{
  "id": "unique_identifier",          // String: Unique ID for this evaluation
  "question": "original_question",    // String: The question that was asked
  "context": "authoritative_source",  // String: Ground truth context
  "response": "llm_generated_answer"  // String: Response to evaluate
}
```

### Validation Rules

- All fields must be present and non-empty strings
- Context should contain authoritative information that responses should be grounded in
- Response should be the actual LLM output to evaluate for hallucinations
- IDs should be unique within your dataset for proper tracking

## Output Files

### Core Results

#### `results.jsonl` - Detailed Results
```json
{
  "id": "q1",
  "sentences": ["Paris is the capital of France.", "It's a beautiful city."],
  "scores": [0.05, 0.75],
  "labels": ["Grounded", "Unsupported"],
  "overall_groundedness": 0.421,
  "decision": "HALLUCINATION",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "input_row": {...},
  "line_number": 1,
  "prompt_sha256": "abc123...",
  "timestamp_utc": "2024-01-15T10:30:00Z"
}
```

#### `summary.csv` - Quick Overview
```csv
id,overall_groundedness,decision,provider,model,timestamp_utc
q1,0.421000,HALLUCINATION,openai,gpt-4o-mini,2024-01-15T10:30:00Z
q2,0.895000,FACT,openai,gpt-4o-mini,2024-01-15T10:30:01Z
```

### Audit Trail

#### `processing_summary.json` - Complete Statistics
```json
{
  "input_file": "data.jsonl",
  "processing_stats": {
    "total_lines": 1000,
    "successfully_processed": 987,
    "invalid_json": 2,
    "missing_fields": 5,
    "processing_errors": 6
  },
  "configuration": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "tau": 0.80
  }
}
```

#### `processing_log.jsonl` - Skipped Rows
```json
{"reason": "missing_fields", "line_number": 42, "missing": ["context"], "timestamp": "..."}
{"reason": "processing_error", "line_number": 127, "error": "API timeout", "timestamp": "..."}
```

#### `traces/` - Debug Information
```
traces/
├── q1.prompt.txt    # Exact prompt sent to judge LLM
├── q1.raw.txt       # Raw response from judge LLM
└── q1.raw.retry.txt # Retry response (if needed)
```

## Configuration

### Threshold Tuning

The `tau` parameter controls decision sensitivity:

```bash
# Strict (flag borderline cases as hallucinations)
python main.py --tau 0.90

# Balanced (recommended default)
python main.py --tau 0.80

# Lenient (only flag clear hallucinations)
python main.py --tau 0.60
```

#### Tuning Guidelines

| Use Case | Recommended Tau | Rationale |
|----------|-----------------|-----------|
| Medical/Legal Content | 0.90+ | High precision needed |
| General Q&A | 0.80 | Balanced accuracy |
| Creative Content | 0.60-0.70 | Allow more flexibility |
| Fact Checking | 0.85+ | Conservative approach |
| News/Journalism | 0.85+ | Accuracy critical |

### Performance Optimization

```bash
# Batch processing for large files
python main.py --max-rows 1000 --out-dir batch_1

# Adjust timeout for complex responses
python main.py --timeout 120

# Speed vs accuracy trade-off
python main.py --model gpt-4o-mini  # Faster
python main.py --model gpt-4o       # More accurate
```

## Error Handling

### Robust Processing Features
- **Graceful Degradation**: Individual failures don't stop batch processing
- **Input Validation**: Comprehensive checking of data format and content
- **Complete Logging**: All errors and skipped rows are logged with specific reasons
- **Data Integrity**: Every result includes original input for verification

### Common Issues & Solutions

#### API Rate Limits
```bash
# Solution: Reduce batch size, check quotas
# Error appears as: "Rate limit exceeded"
```

#### Invalid JSON in Input
```bash
# Check specific problems:
cat judge_outputs/processing_log.jsonl | grep "invalid_json"

# Shows line numbers and specific JSON errors
```

#### Empty Results
```bash
# Verify input format
head -n 3 your_data.jsonl

# Check processing statistics
cat judge_outputs/processing_summary.json
```

#### API Key Issues
```bash
# Verify keys are set
python -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY'))); print('Anthropic:', bool(os.getenv('ANTHROPIC_API_KEY')))"
```


## Testing & Validation

### Data Integrity Verification

```python
# Verify all inputs were processed correctly
import json

# Check processing stats
with open('judge_outputs/processing_summary.json') as f:
    stats = json.load(f)
    
total_valid = stats['processing_stats']['total_lines'] - stats['processing_stats']['empty_lines']
processed = stats['processing_stats']['successfully_processed']

print(f"Success rate: {processed} / {total_valid} ({100*processed/total_valid:.1f}%)")

# Verify results match inputs
with open('judge_outputs/results.jsonl') as f:
    for line in f:
        result = json.loads(line)
        # Each result contains 'input_row' and 'line_number' for verification
        assert result['input_row']['id'] == result['id']
        print(f"Verified: {result['id']}")
```

