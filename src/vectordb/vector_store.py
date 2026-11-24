from functools import cached_property

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from embeddings.embedder import EMBEDDER_MODEL


class VectorStore:
    def __init__(self, persist_directory="db/chroma_db"):
        self.persist_directory = persist_directory

    @cached_property
    def embedding_model(self):
        return HuggingFaceEmbeddings(model_name=EMBEDDER_MODEL)

    def create(self, chunks):
        """Create and persist ChromaDB vector store"""
        print("Creating embeddings and storing in ChromaDB...")

        print("--- Creating vector store ---")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print("--- Finished creating vector store ---")

        print(f"Vector store created and saved to {self.persist_directory}")
        return vector_store
