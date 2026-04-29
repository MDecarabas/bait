"""Clone documentation repos, build Sphinx, and ingest into ChromaDB."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Union

from git import Repo
from langchain_chroma import Chroma
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import BaitConfig, get_config
from .utils import get_embeddings


def ingest_git_documentation(repo_url: str, documentation_dir: Union[str, Path]):
    """Clone or pull a git repo, then build its Sphinx docs."""
    documentation_dir = Path(documentation_dir)
    repo_name = repo_url.split("/")[-1].removesuffix(".git")
    repo_dir = documentation_dir / repo_name

    if not repo_dir.exists():
        print(f"Cloning repository to: {repo_dir}")
        try:
            Repo.clone_from(repo_url, repo_dir)
        except Exception as e:
            print(f"ERROR: Cloning failed: {e}")
            sys.exit(1)
    else:
        print(f"Pulling latest changes in repository: {repo_dir}")
        try:
            repo = Repo(repo_dir)
            origin = repo.remotes.origin
            origin.pull()
        except Exception as e:
            print(f"ERROR: Pulling failed: {e}")
            sys.exit(1)

    docs_path = repo_dir / "docs"
    if not docs_path.exists():
        print(f"ERROR: 'docs' directory not found in repository: {docs_path}")
        sys.exit(1)

    output_dir = docs_path / "_build" / "html"
    command = [
        "sphinx-build",
        "-b",
        "html",
        str(docs_path),
        str(output_dir),
    ]

    print(f"Running Sphinx build: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Sphinx build successful. Output in: {output_dir}")
    except FileNotFoundError:
        print("ERROR: 'sphinx-build' command not found.")
        print("Please make sure Sphinx is installed in your Python environment.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Sphinx build failed with code {e.returncode}.")
        print("\n--- Sphinx Output (stdout) ---")
        print(e.stdout)
        print("\n--- Sphinx Errors (stderr) ---")
        print(e.stderr)
        sys.exit(1)


def load_chunk_embed(html_build_dir: str, config: BaitConfig | None = None):
    """Load HTML docs, split into chunks, embed, and persist to ChromaDB."""
    if config is None:
        config = get_config()

    print(f"Loading docs from {html_build_dir}...")
    loader = ReadTheDocsLoader(html_build_dir)
    docs = loader.load()

    if not docs:
        print("ERROR: No documents were loaded. Check the HTML build directory.")
        sys.exit(1)

    print(f"Loaded {len(docs)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.text_processing.chunk_size,
        chunk_overlap=config.text_processing.chunk_overlap,
    )

    print("Splitting documents into chunks...")
    splits = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} docs into {len(splits)} chunks.")

    print("Initializing embedding model...")
    embeddings = get_embeddings(config)
    print(
        f"Using {config.embedding.provider} for embeddings"
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

    print("All done.")
    print(f"Knowledge base is ready and saved in '{db_path}'.")


def _wipe_chromadb(config: BaitConfig) -> None:
    """Remove the ChromaDB directory if it exists. Used by --rebuild."""
    db_path = config.db_path
    if db_path.exists():
        print(f"Wiping existing ChromaDB at {db_path}...")
        shutil.rmtree(db_path)


def run_ingest(config: BaitConfig | None = None, rebuild: bool = False) -> None:
    """Programmatic ingest entry point.

    With ``rebuild=True``, deletes the ChromaDB before rebuilding so chunks
    are not duplicated when re-running over an existing index.
    """
    if config is None:
        config = get_config()

    print("Starting data ingestion process...")
    print(f"Project: {config.project.name}")
    print(f"Data directory: {config.data_dir}")

    if rebuild:
        _wipe_chromadb(config)

    docs_output_dir = config.docs_output_dir

    for repo_url in config.documentation.git_repos:
        print(f"\nProcessing repository: {repo_url}")
        ingest_git_documentation(repo_url, docs_output_dir)
        repo_name = repo_url.split("/")[-1].removesuffix(".git")
        sphinx_path = docs_output_dir / repo_name / "docs" / "_build" / "html"
        if sphinx_path.exists():
            print(f"\nLoading and embedding documentation from: {sphinx_path}")
            load_chunk_embed(str(sphinx_path), config=config)
        else:
            print(f"WARNING: Sphinx build path does not exist: {sphinx_path}")

    for local_folder in config.documentation.local_folders:
        print(f"\nProcessing local folder: {local_folder}")
        if Path(local_folder).exists():
            load_chunk_embed(local_folder, config=config)
        else:
            print(f"WARNING: Local folder does not exist: {local_folder}")


def main():
    """CLI entry point: ``bait-ingest [--rebuild]``."""
    parser = argparse.ArgumentParser(
        prog="bait-ingest",
        description="Clone, build, and embed Bait documentation sources.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the ChromaDB before ingesting so chunks aren't duplicated.",
    )
    args = parser.parse_args()
    run_ingest(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
