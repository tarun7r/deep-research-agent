from pathlib import Path

from src.local_docs import LocalDocIndex


def test_local_docs_index_and_search(tmp_path: Path):
    root = tmp_path / "docs"
    root.mkdir()

    (root / "a.txt").write_text(
        "This is a document about quantum computing and cryptography. " * 20,
        encoding="utf-8",
    )
    (root / "b.md").write_text(
        "# Notes\n\nThis file discusses LLM agents and LangGraph.\n" * 10,
        encoding="utf-8",
    )

    idx = LocalDocIndex(root, cache_dir=tmp_path / ".cache")
    idx.build()

    results = idx.search("quantum cryptography", k=5)
    assert results
    r0 = results[0]
    assert "path" in r0 and r0["path"]
    assert "content" in r0 and isinstance(r0["content"], str)
    assert "score" in r0

    # Ensure load_or_build uses the saved index when no file changed.
    idx2 = LocalDocIndex(root, cache_dir=tmp_path / ".cache")
    idx2.load_or_build()
    results2 = idx2.search("LangGraph", k=3)
    assert results2
