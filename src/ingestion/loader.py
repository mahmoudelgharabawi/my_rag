import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

load_dotenv()


class IngestionPipeline:
    def __init__(self, docs_path="docs"):
        self.docs_path = docs_path

    def load_documents(self):
        """Load all text files from docs directory"""
        print(f"Loading documents from {self.docs_path}...")

        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(
                f"The directory {self.docs_path} does not exist. Please create it and add your company files."
            )

        loader = DirectoryLoader(
            path=self.docs_path,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8-sig"},
        )

        documents = loader.load()

        if len(documents) == 0:
            raise FileNotFoundError(
                f"No text files found in the directory {self.docs_path}. Please add your company files."
            )

        for i, doc in enumerate(documents):
            print(f"\nDocument {i+1}:")
            print(f"  Source: {doc.metadata['source']}")
        return documents

    def run(self) -> list[Document]:
        """Load source documents."""
        print("=== RAG Document Ingestion Pipeline ===\n")
        return self.load_documents()
