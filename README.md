# PyAnsys Troubleshooting RAG (Terminal)

A terminal-based Retrieval-Augmented Generation (RAG) tool that searches the web in real time and synthesizes PyAnsys troubleshooting guidance using Ollama (Gemma2:2b).

## Features

- LangGraph state machine: search → fetch → generate → handle errors
- DuckDuckGo web search (ddgs)
- Trafilatura extraction with snippet fallback
- Ollama (Gemma2:2b) answer synthesis
- Terminal REPL with progress updates
- Answers end with a clear completion marker

## Requirements

- Python 3.10+
- Ollama installed and running
- Model pulled: `gemma2:2b`

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python cli.py
```

## Usage

- Type a PyAnsys error or troubleshooting question.
- Type `quit` or `exit` to leave.
- The tool searches, fetches sources, and generates a consolidated fix summary.

## Files

- web_search_rag_standalone.py: LangGraph agent and RAG pipeline
- cli.py: Terminal REPL
- requirements.txt: Dependencies

## Notes

- The tool focuses on PyAnsys troubleshooting. General questions are declined.
- SSL verification is disabled for problematic sites during fetching.
- If extraction fails, search snippets are used as fallback sources.
