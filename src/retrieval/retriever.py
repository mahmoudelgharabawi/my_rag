from langchain_core.documents import Document

from vectordb.vector_store import VectorStore


class Retriever:
    """Cosine similarity search that keeps chunks scoring at least 0.3."""

    def __init__(
        self,
        vector_store: VectorStore,
        score_threshold: float = 0.3,
        k: int = 4,
    ):
        score_threshold = float(score_threshold)
        if score_threshold < 0.3 or score_threshold > 1:
            raise ValueError(
                "score_threshold must be from 0.3 to 1.0, "
                f"got {score_threshold}."
            )
        self.score_threshold = score_threshold
        self.k = k
        self.retriever = vector_store.load().as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )

    def run(self, query: str) -> list[Document]:
        """Return the nearest chunks whose relevance score is at least the threshold."""
        return self.retriever.invoke(query)
