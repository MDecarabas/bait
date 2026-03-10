import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from .config import BaitConfig

# Load configuration
config = BaitConfig()


def get_documentation_retriever():
    """
    Initializes and returns a retriever for our ChromaDB.
    """

    if config.embedding.provider == "huggingface":
        print("✅ Using HuggingFace for embeddings!")
        embeddings = HuggingFaceEmbeddings(model_name=config.embedding.model)
    elif config.embedding.provider == "anl_argo":
        # Initialize ANL Argo embeddings

        embeddings = OpenAIEmbeddings(
            model=config.embedding.model,
            openai_api_base=config.embedding.argo_base_url,
            openai_api_key=config.embedding.api_key,
            check_embedding_ctx_length=False,
        )
        
    print(f"Loading embedding model: {config.embedding.model}")
    # Initialize the same embedding model

    db_path = str(config.db_path)
    print(f"Connecting to vector store at: {db_path}")
    # Connect to the existing, persisted database
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

    print("✅ Retriever is ready.")

    # Build search kwargs based on config
    search_kwargs = {"k": config.retriever.k}
    if config.retriever.score_threshold is not None:
        search_kwargs["score_threshold"] = config.retriever.score_threshold

    # Create a retriever object
    return vectorstore.as_retriever(
        search_type=config.retriever.search_type, search_kwargs=search_kwargs
    )


# --- Test Block ---
if __name__ == "__main__":
    """
    This lets us test the retriever function by running:
    python retriever.py "your test question"
    """
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print("\n--- Testing Retriever ---")
        print(f"Query: '{query}'")

        retriever = get_documentation_retriever()

        # 'invoke' runs the retriever and gets the docs
        results = retriever.invoke(query)

        print(f"\nFound {len(results)} relevant documents:")
        for i, doc in enumerate(results):
            print(f"\n--- Document {i + 1} ---")
            print(doc.page_content)
            print(f"(Source: {doc.metadata.get('source', 'unknown')})")
            print("------------------")
    else:
        print('Usage: python retriever.py "Your test query here"')
