"""Vector-store retriever used by the documentation agent."""

import sys

from langchain_chroma import Chroma

from .config import BaitConfig, get_config
from .utils import get_embeddings


def get_documentation_retriever(config: BaitConfig | None = None):
    """Return a retriever bound to the configured ChromaDB.

    Pass ``config`` for tests or alternate beamline contexts; otherwise the
    cached process config is used.
    """
    if config is None:
        config = get_config()

    embeddings = get_embeddings(config)
    print(f"Loading embedding model: {config.embedding.model}")

    db_path = str(config.db_path)
    print(f"Connecting to vector store at: {db_path}")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

    print("Retriever is ready.")

    search_kwargs = {"k": config.retriever.k}
    if config.retriever.score_threshold is not None:
        search_kwargs["score_threshold"] = config.retriever.score_threshold

    return vectorstore.as_retriever(
        search_type=config.retriever.search_type, search_kwargs=search_kwargs
    )


# --- Test Block ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print("\n--- Testing Retriever ---")
        print(f"Query: '{query}'")

        retriever = get_documentation_retriever()
        results = retriever.invoke(query)

        print(f"\nFound {len(results)} relevant documents:")
        for i, doc in enumerate(results):
            print(f"\n--- Document {i + 1} ---")
            print(doc.page_content)
            print(f"(Source: {doc.metadata.get('source', 'unknown')})")
            print("------------------")
    else:
        print('Usage: python -m bait.retriever "Your test query here"')
