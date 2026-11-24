from ingestion.loader import IngestionPipeline
from vectordb.vector_store import VectorStore

if __name__ == "__main__":
    ingestion_pipeline = IngestionPipeline(show_expected_tokens=True)
    documents = ingestion_pipeline.run()

    vector_store = VectorStore()
    vector_store.create(documents)
