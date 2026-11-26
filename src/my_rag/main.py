from chunking.chunker import Chunker
from ingestion.loader import IngestionPipeline
from vectordb.vector_store import VectorStore

if __name__ == "__main__":
    ingestion_pipeline = IngestionPipeline()
    documents = ingestion_pipeline.run()

    chunker = Chunker(show_expected_tokens=True)
    chunks = chunker.run(documents)

    vector_store = VectorStore()
    vector_store.create(chunks)
