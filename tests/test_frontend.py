"""Tests for the Gradio frontend helpers."""

from __future__ import annotations

from pathlib import Path


def _make_built_repo(repo_root: Path, name: str) -> Path:
    """Create a fake repo build dir matching docs/_build/html and return it."""
    build = repo_root / name / "docs" / "_build" / "html"
    build.mkdir(parents=True)
    (build / "index.html").write_text("<html></html>")
    return build


def test_resolve_allowed_paths_no_sources(write_config):
    """No git_repos and no local_folders → empty list."""
    write_config()
    from bait.config import BaitConfig
    from bait.frontend import _resolve_allowed_paths

    config = BaitConfig()
    assert _resolve_allowed_paths(config) == []


def test_resolve_allowed_paths_one_built_git_repo(write_config, tmp_path):
    """A built git repo's html dir is included."""
    write_config(
        extra={
            "documentation": {
                "git_repos": ["https://github.com/x/repo-one.git"],
            }
        }
    )
    from bait.config import BaitConfig
    from bait.frontend import _resolve_allowed_paths

    config = BaitConfig()
    build = _make_built_repo(config.docs_output_dir, "repo-one")

    paths = _resolve_allowed_paths(config)
    assert paths == [str(build.resolve())]


def test_resolve_allowed_paths_two_git_repos(write_config, tmp_path):
    """Both git_repos appear in allowed_paths when both have build dirs."""
    write_config(
        extra={
            "documentation": {
                "git_repos": [
                    "https://github.com/x/repo-one.git",
                    "https://github.com/x/repo-two.git",
                ]
            }
        }
    )
    from bait.config import BaitConfig
    from bait.frontend import _resolve_allowed_paths

    config = BaitConfig()
    b1 = _make_built_repo(config.docs_output_dir, "repo-one")
    b2 = _make_built_repo(config.docs_output_dir, "repo-two")

    paths = set(_resolve_allowed_paths(config))
    assert paths == {str(b1.resolve()), str(b2.resolve())}


def test_resolve_allowed_paths_git_plus_local(write_config, tmp_path):
    """Local folders are included alongside git build dirs."""
    local_dir = tmp_path / "local-docs"
    local_dir.mkdir()
    write_config(
        extra={
            "documentation": {
                "git_repos": ["https://github.com/x/repo-one.git"],
                "local_folders": [str(local_dir)],
            }
        }
    )
    from bait.config import BaitConfig
    from bait.frontend import _resolve_allowed_paths

    config = BaitConfig()
    build = _make_built_repo(config.docs_output_dir, "repo-one")

    paths = set(_resolve_allowed_paths(config))
    assert paths == {str(build.resolve()), str(local_dir.resolve())}


def test_resolve_allowed_paths_skips_unbuilt_source(write_config):
    """A configured repo with no build dir is silently skipped — no crash."""
    write_config(
        extra={
            "documentation": {
                "git_repos": ["https://github.com/x/never-built.git"],
            }
        }
    )
    from bait.config import BaitConfig
    from bait.frontend import _resolve_allowed_paths

    config = BaitConfig()
    assert _resolve_allowed_paths(config) == []
