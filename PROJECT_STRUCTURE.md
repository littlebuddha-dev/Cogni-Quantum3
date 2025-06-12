# CogniQuantum Project Structure

This document outlines the directory and file structure of the CogniQuantum project.

```
.
├── .gitignore
├── cli/
│   ├── init.py
│   ├── handler.py
│   └── main.py
├── examples/
│   ├── sample_questions.txt
│   └── v2_demo_script.sh
├── fetch_llm_v2.py
├── installation_guide.md
├── llm_api/
│   ├── init.py
│   ├── config.py
│   ├── quantum_engine.py
│   ├── cogniquantum/
│   │   ├── init.py
│   │   ├── analyzer.py
│   │   ├── engine.py
│   │   ├── enums.py
│   │   ├── learner.py
│   │   ├── system.py
│   │   └── tracker.py
│   ├── providers/
│   │   ├── init.py
│   │   ├── base.py
│   │   ├── claude.py
│   │   ├── enhanced_claude_v2.py
│   │   ├── enhanced_gemini_v2.py
│   │   ├── enhanced_huggingface_v2.py
│   │   ├── enhanced_ollama_v2.py
│   │   ├── enhanced_openai_v2.py
│   │   ├── gemini.py
│   │   ├── huggingface.py
│   │   ├── ollama.py
│   │   └── openai.py
│   ├── rag/
│   │   ├── init.py
│   │   ├── knowledge_base.py
│   │   ├── manager.py
│   │   └── retriever.py
│   ├── tools/
│   │   ├── init.py
│   │   └── image_retrieval.py
│   └── utils/
│       ├── init.py
│       ├── helper_functions.py
│       └── performance_monitor.py
├── PROJECT_STRUCTURE.md
├── quick_test_v2.py
├── README.md
├── requirements.txt
├── test_all_v2_providers.py
└── tests/
├── init.py
├── test_cli.py
├── test_cogniquantum.py
└── test_providers.py
```
## Directory and File Overview

### Root Directory

- **`.gitignore`**: Specifies files and directories that Git should ignore.
- **`fetch_llm_v2.py`**: A simple entry point that starts the command-line interface. The main logic resides in the `cli/` directory.
- **`installation_guide.md`**: A detailed guide for setting up and installing the project.
- **`PROJECT_STRUCTURE.md`**: This file, outlining the project's structure.
- **`quick_test_v2.py`**: A diagnostic script to quickly verify the user's environment setup (API keys, Ollama connection, etc.).
- **`README.md`**: The main documentation file containing the project's overview, purpose, features, and usage instructions.
- **`requirements.txt`**: Lists the Python libraries and their versions required to run the project.
- **`test_all_v2_providers.py`**: A comprehensive test script that checks the operation of all configured V2 providers in each mode.

### cli/

This directory contains all the logic for the command-line interface.

- **`__init__.py`**: Allows the cli directory to be treated as a Python package.
- **`handler.py`**: The core logic class of the CLI. It handles request processing, provider fallbacks (V2 Enhanced -> Standard), and generating suggestions.
- **`main.py`**: The main entry point of the CLI application. It handles argument parsing (including RAG flags like `--rag` and `--wikipedia`) and orchestrates the execution flow.

### examples/

Contains sample scripts and data files demonstrating the project's functionality.

- **`sample_questions.txt`**: A list of sample questions for testing each mode.
- **`v2_demo_script.sh`**: The latest shell script for trying out various modes and providers.

### llm_api/

The main package containing the core reasoning logic and communication with LLM providers.

- **`__init__.py`**: Allows the llm_api directory to be treated as a Python package.
- **`config.py`**: Provides centralized settings management using Pydantic, loading configuration from a `.env` file. It includes a setting to limit Ollama concurrency to prevent server crashes.
- **`quantum_engine.py`**: Implements the quantum-inspired reasoning approach, which generates a superposition of diverse hypotheses and collapses them into a single, robust solution.

#### cogniquantum/

The core of the CogniQuantum V2 system, broken down by responsibility.

- **`__init__.py`**: Exposes the main `CogniQuantumSystemV2` class.
- **`analyzer.py`**: Contains the `AdaptiveComplexityAnalyzer` for analyzing prompt complexity using multi-language NLP and keyword-based methods.
- **`engine.py`**: Contains the `EnhancedReasoningEngine` which executes different reasoning strategies. For high-complexity problems, it uses a sequential integration method to avoid errors from long prompts.
- **`enums.py`**: Defines enumerations used in the system, such as `ComplexityRegime`.
- **`learner.py`**: Implements the `ComplexityLearner`, which records successful prompt-regime pairings to improve future complexity analysis.
- **`system.py`**: Contains the `CogniQuantumSystemV2` class, which orchestrates the entire problem-solving process, integrating RAG, tool use, and different reasoning pipelines (e.g., parallel, quantum-inspired).
- **`tracker.py`**: Contains data classes for tracking performance metrics and solutions.

#### providers/

Modules responsible for communication with LLM services.

- **`__init__.py`**: A factory function to dynamically import and instantiate the appropriate provider.
- **`base.py`**: Defines the abstract base classes (`LLMProvider`, `EnhancedLLMProvider`). The `EnhancedLLMProvider` now centralizes the core V2 logic.
- **`openai.py`, `claude.py`, etc.**: Standard provider classes for each LLM service.
- **`enhanced_*_v2.py`**: Advanced providers that wrap standard providers to execute complexity-adaptive reasoning, delegating common logic to the base class.

#### rag/

Retrieval-Augmented Generation (RAG) functionality.

- **`__init__.py`**: Exposes the `RAGManager`.
- **`knowledge_base.py`**: Manages loading and vectorizing documents from files/URLs using updated LangChain components.
- **`manager.py`**: The central RAG orchestrator. It uses an LLM to extract optimized search queries from user prompts before retrieving information from Wikipedia or a local knowledge base.
- **`retriever.py`**: A retriever class that uses the recommended `invoke` method for searching the vector store.

#### tools/

A package for external tools.

- **`__init__.py`**: Initializes the tools package. Image retrieval is currently disabled by default.
- **`image_retrieval.py`**: A tool to search for images on the web (currently not activated).

#### utils/

Helper functions and auxiliary classes.

- **`__init__.py`**: Allows the utils directory to be treated as a Python package.
- **`helper_functions.py`**: Provides auxiliary functions, such as reading from pipes/files and formatting JSON.
- **`performance_monitor.py`**: Measures performance metrics like processing time and token usage.

### tests/

Unit and integration tests for the project.

- **`__init__.py`**: Allows the tests directory to be treated as a Python package.
- **`test_cli.py`**: Tests the command-line arguments and functionality.
- **`test_cogniquantum.py`**: Tests the complexity analysis and reasoning logic.
- **`test_providers.py`**: Tests the dynamic loading and caching of providers.
