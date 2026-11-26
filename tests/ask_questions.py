"""Ask questions and print the chunks the retriever returns.

Run from the project root after main.py has built the vector store:

    uv run python tests/ask_questions.py
    uv run python tests/ask_questions.py "who founded Google"
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval.retriever import Retriever
from vectordb.vector_store import VectorStore


def show_chunks(question, chunks):
    print(f"\nQuestion: {question}")
    if not chunks:
        print("No chunks with relevance >= 0.3.")
        return

    print(f"Chunks returned: {len(chunks)}")
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(f"Source: {chunk.metadata.get('source', '?')}")
        print(chunk.page_content)
        print("-" * 50)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    db_path = ROOT / "db" / "chroma_db"
    if not db_path.exists():
        raise SystemExit(
            "No vector store at db/chroma_db. Run src/my_rag/main.py first."
        )

    retriever = Retriever(VectorStore(persist_directory=str(db_path)))
    questions = sys.argv[1:]
    if questions:
        for question in questions:
            show_chunks(question, retriever.run(question))
        return

    print("Ask a question. Press Enter on an empty line to exit.")
    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            break
        show_chunks(question, retriever.run(question))


if __name__ == "__main__":
    main()
