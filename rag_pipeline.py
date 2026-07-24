"""End-to-end LlamaIndex RAG pipeline over a persistent ChromaDB store.

Stages:
    1. build_vector_store() -- shared embed model + persistent Chroma vector store
    2. load_documents()     -- read ./documents (falls back to a sample doc)
    3. ingest()             -- chunk + embed documents into the vector store
    4. get_query_engine()   -- build the index and a Qwen-backed query engine

Run:
    python rag_pipeline.py               # ingest if empty, then run a sample query
    python rag_pipeline.py --reingest    # force re-ingestion of ./documents
"""

import argparse
import os

import chromadb
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.vector_stores.chroma import ChromaVectorStore

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "rag"
DOCUMENTS_DIR = "documents"

load_dotenv()


def build_vector_store():
    """Create the shared embed model and a persistent Chroma vector store.

    Returns:
        (embed_model, vector_store, chroma_collection)
    """
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return embed_model, vector_store, chroma_collection


def load_documents():
    """Load documents from ./documents, or a sample doc if the dir is empty."""
    if os.path.isdir(DOCUMENTS_DIR) and os.listdir(DOCUMENTS_DIR):
        return SimpleDirectoryReader(input_dir=DOCUMENTS_DIR).load_data()
    print(f"No files in ./{DOCUMENTS_DIR} -- using a sample document.")
    return [Document.example()]


def ingest(embed_model, vector_store):
    """Chunk + embed documents and persist the resulting nodes into Chroma."""
    documents = load_documents()
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_overlap=0),
            embed_model,
        ],
        vector_store=vector_store,
    )
    nodes = pipeline.run(documents=documents)
    print(f"Ingested {len(nodes)} nodes from {len(documents)} document(s).")
    return nodes


def get_query_engine(embed_model, vector_store):
    """Build the index over the vector store and return a query engine."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add it to your .env file "
            "(see .env.example)."
        )
    llm = HuggingFaceInferenceAPI(model_name=LLM_MODEL_NAME, token=hf_token)
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embed_model
    )
    return index.as_query_engine(llm=llm, response_mode="tree_summarize")


def main():
    parser = argparse.ArgumentParser(description="Run the RAG pipeline.")
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Force re-ingestion of ./documents even if the store is populated.",
    )
    parser.add_argument(
        "--query",
        default="What is this document about?",
        help="Question to ask the query engine.",
    )
    args = parser.parse_args()

    embed_model, vector_store, chroma_collection = build_vector_store()

    if args.reingest or chroma_collection.count() == 0:
        ingest(embed_model, vector_store)
    else:
        print(
            f"Vector store already has {chroma_collection.count()} nodes -- "
            "skipping ingestion (use --reingest to force)."
        )

    query_engine = get_query_engine(embed_model, vector_store)
    response = query_engine.query(args.query)
    print(f"\nQ: {args.query}\nA: {response}")


if __name__ == "__main__":
    main()
