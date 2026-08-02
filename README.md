# AI Portfolio Projects

Professional portfolio repository for two AI internship projects completed in Python.

## Repository Overview

This repository contains:

- **Project-1**: Rule-based AI chatbot with deterministic intent mapping
- **Project-2**: Supervised machine learning classification pipeline using K-Nearest Neighbors on the Iris dataset

## Project Structure

```text
AI-Portfolio-Projects/
├── Project-1/
│   ├── chatbot.py
│   └── README.md
├── Project-2/
│   ├── classification.py
│   └── README.md
└── LICENSE
```

## Technical Summary

### Project-1: Rule-Based AI Chatbot

- Implements a dictionary-driven intent response system
- Uses input normalization for reliable matching
- Handles unknown inputs with fallback behavior
- Includes graceful session termination controls

### Project-2: Data Classification Using AI

- Builds a complete supervised learning pipeline
- Uses feature scaling and train-test split discipline
- Trains and evaluates a KNN classifier
- Reports confusion matrix and weighted F1 score

## Critical Analysis

### Strengths

- Clear separation between symbolic AI logic and statistical learning workflows
- Good foundational coverage of input handling, training flow, and evaluation metrics
- Practical focus on interpretable implementations suitable for learning and demonstration

### Gaps

- No automated tests are included for regression protection
- No dependency pinning or environment file for reproducibility
- Limited packaging and execution standardization across projects

### Recommended Next Steps

- Add unit tests for chatbot intent mapping and ML pipeline invariants
- Introduce `requirements.txt` with explicit version constraints
- Add a unified runner or Makefile for consistent local execution
- Extend documentation with benchmark targets and expected outputs

## Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` for package installation

### Run Project-1

```bash
cd Project-1
python chatbot.py
```

### Run Project-2

```bash
cd Project-2
pip install scikit-learn numpy
python classification.py
```

## License

This repository is distributed under the terms of the LICENSE file.
