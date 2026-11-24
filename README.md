# my_rag

A Retrieval-Augmented Generation (RAG) project built with Python 3.12 and managed with [uv](https://docs.astral.sh/uv/). It ingests documents, chunks and embeds them, stores vectors for retrieval, and answers questions with a local LLM ([SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)) via Hugging Face Transformers and PyTorch.

> **Status:** work in progress. The LLM client is implemented; the remaining pipeline modules are scaffolded and under development.

## Project structure

```
my_rag/
├── docs/                        # Sample source documents for ingestion
│   ├── attention-is-all-you-need.pdf
│   ├── Google.txt
│   ├── Microsoft.txt
│   ├── Nvidia.txt
│   ├── SpaceX.txt
│   └── Tesla.txt
├── logs/
│   └── app_logs.py              # Application logging setup
├── src/
│   ├── my_rag/
│   │   └── main.py              # Application entry point
│   ├── api/
│   │   └── routes.py            # API endpoints
│   ├── ingestion/
│   │   └── loader.py            # Document loading (PDF / TXT)
│   ├── chunking/
│   │   └── chunker.py           # Splitting documents into chunks
│   ├── embeddings/
│   │   └── embedder.py          # Chunk embedding generation
│   ├── vectordb/
│   │   └── vector_store.py      # Vector storage and indexing
│   ├── retrieval/
│   │   └── retriever.py         # Similarity search over stored vectors
│   ├── prompts/
│   │   └── prompt_templates.py  # Prompt templates for grounded answers
│   ├── llm/
│   │   └── llm_client.py        # Local LLM inference (SmolLM2-135M-Instruct)
│   └── utils/
│       └── helpers.py           # Shared helpers
├── tests/
│   └── test_app.py              # Test suite
├── pyproject.toml               # Project metadata and dependencies
├── uv.lock                      # Locked dependency versions
└── .python-version              # Pins Python 3.12
```

## RAG pipeline

```
docs/ → ingestion → chunking → embeddings → vectordb
                                                │
question → retrieval (similarity search) → prompt template → LLM → answer
```

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.12 (uv installs it automatically from `.python-version`)
- Optional: CUDA-capable GPU — the LLM client falls back to CPU automatically

## Setup

Install uv (Windows PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or on macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and sync the project:

```bash
git clone <repo-url>
cd my_rag
uv sync
```

`uv sync` creates the `.venv`, installs the pinned Python version, and resolves all dependencies from `uv.lock` (PyTorch, Transformers).

## Usage

Run the application entry point:

```bash
uv run my-rag
```

Or run any module/script inside the managed environment:

```bash
uv run python src/my_rag/main.py
```

Use the LLM client directly:

```python
from llm.llm_client import LLMClient

client = LLMClient()  # loads the model once per process
print(client.generate("What is retrieval-augmented generation?"))
```

The model weights are downloaded from Hugging Face on first use and cached locally.

## Managing dependencies

```bash
uv add <package>        # add a dependency (updates pyproject.toml + uv.lock)
uv remove <package>     # remove a dependency
uv sync                 # reinstall the exact locked environment
uv lock --upgrade       # refresh locked versions
```

Commit both `pyproject.toml` and `uv.lock` so everyone gets identical environments.

## Tests

```bash
uv add --dev pytest   # once, to add the test runner
uv run pytest
```

## License

Not specified yet.