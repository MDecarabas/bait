"""Script entry points for TomoBait CLI commands."""

import os
import subprocess
import sys


def start_backend():
    """Start the FastAPI backend server."""
    # Set PYTHONPATH to include src directory
    env = os.environ.copy()
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src")
    env["PYTHONPATH"] = src_path

    # Start uvicorn with reload
    subprocess.run(
        ["uvicorn", "tomobait.app:api", "--reload", "--port", "8001"],
        env=env,
    )


def start_frontend():
    """Start the Gradio frontend."""
    # Set PYTHONPATH to include src directory
    env = os.environ.copy()
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src")
    env["PYTHONPATH"] = src_path

    # Run frontend module
    subprocess.run(
        [sys.executable, "-m", "tomobait.frontend"],
        env=env,
    )


def lint():
    """Run ruff linting."""
    subprocess.run(["ruff", "check", "."])


def format_code():
    """Run ruff formatting."""
    subprocess.run(["ruff", "format", "."])
