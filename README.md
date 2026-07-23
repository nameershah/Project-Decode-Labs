# Rule-Based AI Chatbot

**Decode Labs AI Internship | Project 1 | Batch 2026**

## Overview

ARIA (Automated Rule-based Intelligence Assistant) is a rule-based chatbot built on deterministic control flow. It uses a dictionary-based knowledge base for O(1) intent lookup, input sanitization, and a continuous conversation loop with a clean exit strategy.

## How to Run

```bash
python chatbot.py
```

No external dependencies. Standard Python 3 only.

## Features

- Continuous `while` loop for uninterrupted conversation
- Input sanitization via `.lower().strip()`
- Dictionary knowledge base with 15+ intents (O(1) lookup)
- Graceful fallback response for unknown inputs
- Clean exit on `quit`, `exit`, `bye`, or `goodbye`
- Handles `Ctrl+C` without crashing

## Example Session

```
You: hello
ARIA: Hey! I'm ARIA, your AI assistant. How can I help you today?

You: what is machine learning
ARIA: Machine learning is a subset of AI where systems learn from data and improve over time without being explicitly programmed.

You: exit
ARIA: Goodbye! Session terminated.
```

## Architecture

| Component      | Implementation                          |
|----------------|-----------------------------------------|
| Input loop     | `while True`                            |
| Sanitization   | `.lower().strip()`                      |
| Knowledge base | Python dictionary (O(1) lookup)         |
| Fallback       | `.get(key, default)` default parameter  |
| Exit strategy  | `None` sentinel + `break`              |
