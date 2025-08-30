# CODR: Clinical Ontology-Driven Reasoning

CODR provides agentic AI tools to extract citations from medical documents that stepwise establish ICD-10-CM codes through hierarchical reasoning.

## Overview

CODR recursively traverses the ICD-10 code hierarchy (from Chapter through Block, Category, Subcategory, Subclassifications, and Extensions) providing increasingly specific context for accurate medical code generation. This approach mirrors clinical coding workflows by building evidence-based justifications for each level of specificity.

## Features

- **Hierarchical Code Generation**: Step-by-step traversal of ICD-10-CM taxonomy
- **Citation Extraction**: Identifies supporting evidence from medical documents
- **Multi-LLM Comparison**: Evaluate medical coding knowledge across different language models
- **Clinical Context**: Provides contextual reasoning at each hierarchy level if the code is not yet billable

## Architecture

The system processes medical documents by:
1. Starting at the ICD-10 Chapter level
2. Progressively narrowing to more specific categories
3. Extracting relevant citations at each step
4. Building comprehensive coding justifications
5. Comparing performance across LLM providers

## LLM Provider Support

CODR supports multiple language model providers for comparative analysis of medical coding capabilities, allowing researchers and practitioners to evaluate model performance on clinical coding tasks.