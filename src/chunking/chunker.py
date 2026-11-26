from functools import cached_property

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from embeddings.embedder import EMBEDDER_MODEL, EMBEDDER_TOKEN_LIMIT

load_dotenv()


class Chunker:
    CONTENT_TOKEN_LIMIT = EMBEDDER_TOKEN_LIMIT - 2

    def __init__(
        self,
        chunk_size=248,
        chunk_overlap=40,
        show_expected_tokens=False,
    ):
        if chunk_size > self.CONTENT_TOKEN_LIMIT:
            raise ValueError(
                f"chunk_size must be <= {self.CONTENT_TOKEN_LIMIT} content tokens "
                f"({EMBEDDER_TOKEN_LIMIT} including [CLS]/[SEP] for {EMBEDDER_MODEL}), "
                f"got {chunk_size}."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.show_expected_tokens = show_expected_tokens

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(EMBEDDER_MODEL)

    def expected_tokens(self, text):
        return len(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    def split_documents(self, documents):
        """Split documents into chunks sized by embedder tokens, with overlap."""
        print("Splitting documents into chunks...")

        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = text_splitter.split_documents(documents)
        counts = [self.expected_tokens(c.page_content) for c in chunks]

        if chunks:
            for i, (chunk, n) in enumerate(zip(chunks, counts)):
                if i == 5:
                    break
                print(f"\n--- Chunk {i+1} ---")
                print(f"Source: {chunk.metadata['source']}")
                print(f"Length: {len(chunk.page_content)} characters")
                if self.show_expected_tokens:
                    status = "over limit" if n > EMBEDDER_TOKEN_LIMIT else "ok"
                    print(f"Expected tokens: {n}/{EMBEDDER_TOKEN_LIMIT} ({status})")
                print("-" * 50)
            if len(chunks) > 5:
                print(f"\n... and {len(chunks) - 5} more chunks")

            if self.show_expected_tokens:
                self.report_token_stats(chunks, counts)

        self.enforce_token_limit(chunks, counts)
        return chunks

    def enforce_token_limit(self, chunks, counts):
        """Reject chunks the embedder would silently truncate."""
        over = [(c, n) for c, n in zip(chunks, counts) if n > EMBEDDER_TOKEN_LIMIT]
        if not over:
            return

        lines = [
            f"{c.metadata.get('source', '?')}: {n} tokens, "
            f"preview: {c.page_content[:60]!r}"
            for c, n in over[:5]
        ]
        if len(over) > 5:
            lines.append(f"... and {len(over) - 5} more over-limit chunks")
        detail = "\n".join(lines)
        raise ValueError(
            f"{len(over)} chunk(s) exceed the embedder limit of "
            f"{EMBEDDER_TOKEN_LIMIT} tokens.\n{detail}"
        )

    def report_token_stats(self, chunks, counts):
        """Check layer: count expected embedder tokens for every chunk"""
        over = sum(n > EMBEDDER_TOKEN_LIMIT for n in counts)

        print("\n=== Expected token report ===")
        print(f"Chunks: {len(chunks)}")
        print(
            f"Tokens per chunk: min={min(counts)}, max={max(counts)}, "
            f"avg={sum(counts) / len(counts):.0f}"
        )
        print(f"Embedder limit: {EMBEDDER_TOKEN_LIMIT} tokens (incl. [CLS]/[SEP])")
        print(f"Chunks over limit: {over}")

    def run(self, documents) -> list[Document]:
        """Split loaded documents into chunks."""
        return self.split_documents(documents)
