"""
A module for cloning and updating a git repository, and then building its
Sphinx documentation.
"""

import subprocess
import sys
from pathlib import Path
from typing import Union

from git import Repo
from langchain_chroma import Chroma
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import BaitConfig
from .utils import get_embeddings

# Load configuration
config = BaitConfig()


def ingest_git_documentation(repo_url: str, documentation_dir: Union[str, Path]):
    """
    Clones a repository if it doesn't exist, or pulls the latest changes if it does.
    Then, it builds the Sphinx documentation.

    Args:
        repo_url (str): The URL of the git repository to clone.
        documentation_dir (Union[str, Path]): The path to the directory where the
            documentation and repository will be stored.
    """
    documentation_dir = Path(documentation_dir)

    # Get the last part and remove the specific suffix
    repo_name = repo_url.split("/")[-1].removesuffix(".git")
    repo_dir = documentation_dir / repo_name

    # --- 1. Clone or Pull Repository ---
    if not repo_dir.exists():
        print(f"Cloning repository to: {repo_dir}")
        try:
            Repo.clone_from(repo_url, repo_dir)
        except Exception as e:
            print(f"❌ ERROR: Cloning failed: {e}")
            sys.exit(1)
    else:
        print(f"Pulling latest changes in repository: {repo_dir}")
        try:
            repo = Repo(repo_dir)
            origin = repo.remotes.origin
            origin.pull()
        except Exception as e:
            print(f"❌ ERROR: Pulling failed: {e}")
            sys.exit(1)

    # --- 2. Build Sphinx Documentation ---
    docs_path = repo_dir / "docs"
    if not docs_path.exists():
        print(f"❌ ERROR: 'docs' directory not found in repository: {docs_path}")
        sys.exit(1)

    # It's better to run sphinx-build from the original working directory
    # and specify the source and output directories.
    # This avoids issues with `os.chdir`.
    output_dir = docs_path / "_build" / "html"
    command = [
        "sphinx-build",
        "-b",
        "html",  # Build HTML
        str(docs_path),  # Source directory
        str(output_dir),  # Output directory
    ]

    print(f"Running Sphinx build: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ Sphinx build successful. Output in: {output_dir}")

    except FileNotFoundError:
        print("❌ ERROR: 'sphinx-build' command not found.")
        print("Please make sure Sphinx is installed in your Python environment.")
        sys.exit(1)

    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Sphinx build failed with code {e.returncode}.")
        print("\n--- Sphinx Output (stdout) ---")
        print(e.stdout)
        print("\n--- Sphinx Errors (stderr) ---")
        print(e.stderr)
        sys.exit(1)


def load_chunk_embed(HTML_BUILD_DIR: str):
    print(f"Loading docs from {HTML_BUILD_DIR}...")
    loader = ReadTheDocsLoader(HTML_BUILD_DIR)
    docs = loader.load()

    if not docs:
        print("❌ ERROR: No documents were loaded. Check your HTML_BUILD_DIR.")
        sys.exit(1)

    print(f"✅ Loaded {len(docs)} documents.")

    # This splitter tries to keep paragraphs/sentences together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.text_processing.chunk_size,
        chunk_overlap=config.text_processing.chunk_overlap,
    )

    print("Splitting documents into chunks...")
    splits = text_splitter.split_documents(docs)
    print(f"✅ Split {len(docs)} docs into {len(splits)} chunks.")

    print("Initializing embedding model...")
    embeddings = get_embeddings(config)
    print(
        f"✅ Using {config.embedding.provider} for embeddings"
        f" (model: {config.embedding.model})"
    )

    db_path = str(config.db_path)
    print(f"Embedding chunks and saving to vector store at: {db_path}...")
    if Path(db_path).exists():
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        vectorstore.add_documents(splits)
    else:
        Chroma.from_documents(
            documents=splits, embedding=embeddings, persist_directory=db_path
        )

    print("🎉 All done!")
    print(f"Your knowledge base is ready and saved in '{db_path}'.")


if __name__ == "__main__":
    print("🚀 Starting data ingestion process...")
    print("Configuration loaded from config.yaml")
    print(f"Project: {config.project.name}")
    print(f"Data directory: {config.data_dir}")

    docs_output_dir = config.docs_output_dir

    # Process all git repositories
    for repo_url in config.documentation.git_repos:
        print(f"\n📦 Processing repository: {repo_url}")
        ingest_git_documentation(repo_url, docs_output_dir)

        # Load, chunk, and embed from the built HTML
        repo_name = repo_url.split("/")[-1].removesuffix(".git")
        sphinx_path = docs_output_dir / repo_name / "docs" / "_build" / "html"
        if sphinx_path.exists():
            print(f"\n📚 Loading and embedding documentation from: {sphinx_path}")
            load_chunk_embed(str(sphinx_path))
        else:
            print(f"⚠️  Sphinx build path does not exist: {sphinx_path}")

    # Process all local folders
    for local_folder in config.documentation.local_folders:
        print(f"\n📁 Processing local folder: {local_folder}")
        # Local folders are already built, just load and embed
        if Path(local_folder).exists():
            load_chunk_embed(local_folder)
        else:
            print(f"⚠️  WARNING: Local folder does not exist: {local_folder}")
