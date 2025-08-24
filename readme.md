# Hallucination Detector

A comprehensive implementation of state-of-the-art methods for detecting hallucinations in Large Language Model outputs. This toolkit reproduces and extends detection methodologies from research and industry, providing multiple approaches to identify factual errors, contradictions, and unsupported claims in AI-generated content.

## Overview

This toolkit provides robust detection methods that work with any LLM-generated content, regardless of the underlying system architecture.

Hallucination in LLMs refers to the generation of content that appears plausible but is factually incorrect, unsupported by the input context, or entirely fabricated. This toolkit addresses the critical need for reliable hallucination detection across diverse applications and use cases.

### Why Hallucination Detection Matters

- **Trust & Safety**: Ensure AI systems provide reliable, grounded information
- **Quality Assurance**: Maintain high standards in AI-powered applications
- **Risk Mitigation**: Prevent propagation of misinformation
- **User Experience**: Build confidence in AI-generated content
- **Research & Development**: Enable systematic evaluation of model improvements
- **Compliance**: Meet regulatory requirements for AI transparency

## Detection Methods

### Currently Implemented

| Method | Description | Use Case | Accuracy | Speed | Status |
|--------|-------------|----------|----------|-------|--------|
| **[LLM Judge](./llm_judge/)** | Uses judge LLMs for sentence-level grounding evaluation | General purpose, content verification | High | Medium | Complete |

### Planned Implementations

Based on the AWS methodology and additional research:

| Method | Description | Source | Status | Expected Release |
|--------|-------------|---------|---------|------------------|
| **Embedding Similarity** | Semantic similarity between context and response | AWS Blog | Planning | 2025 |


## Quick Start

Choose the detection method that best fits your use case:

### LLM Judge

```bash
cd llm_judge
pip install -r requirements.txt
python main.py --input your_data.jsonl --provider openai --model gpt-4o-mini
```

[Full LLM Judge Documentation](./llm_judge/README.md)



## Universal Data Format

All methods in this toolkit use a standardized input format for consistency:

```jsonl
{"id": "unique_id", "question": "original_question", "context": "source_context", "response": "llm_response"}
```

This enables easy comparison and ensemble approaches across different detection methods.

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup
```bash
git clone https://github.com/thatechmaestro/hallucination-detector.git
cd hallucination-detector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install specific method
cd llm_judge
pip install -r requirements.txt
```


## Related Projects & References

### Academic Papers
- [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896)


### Industry Resources
- [AWS: Detect hallucinations for RAG-based systems](https://aws.amazon.com/blogs/machine-learning/detect-hallucinations-for-rag-based-systems/)
- [OpenAI: GPT-4 System Card](https://cdn.openai.com/papers/gpt-4-system-card.pdf)


---

**Building Trust in AI through Rigorous Hallucination Detection**