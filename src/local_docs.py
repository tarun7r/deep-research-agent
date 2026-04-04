"""Local document indexing and search (hybrid research).

Design goals:
- No external services required (local-only)
- Minimal dependencies (PDF/DOCX parsers are optional)
- Persistent cache on disk for fast subsequent runs

Supported by default:
- .txt, .md, .markdown
- .html, .htm (strips tags)

Optional (if deps installed):
- PDF via `pypdf`
- DOCX via `python-docx`

The index stores chunk-level text for retrieval.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from src.config import config

logger = logging.getLogger(__name__)


SUPPORTED_EXTS = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".docx"}


def _tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    # Keep simple words + numbers
    return re.findall(r"[a-z0-9]{2,}", text)


def _chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    chunks: list[str] = []
    step = max(1, chunk_size - max(0, overlap))
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def _strip_html(html: str) -> str:
    # Lightweight HTML stripping (no extra deps)
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"\s+", " ", html)
    return html.strip()


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        raise RuntimeError(f"PDF support requires optional dependency pypdf: {e}")

    text_parts: list[str] = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            text_parts.append(t)
    return "\n".join(text_parts).strip()


def _read_docx(path: Path) -> str:
    try:
        import docx  # type: ignore
    except Exception as e:
        raise RuntimeError(f"DOCX support requires optional dependency python-docx: {e}")

    doc = docx.Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(parts).strip()


def _read_file_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext in {".html", ".htm"}:
        return _strip_html(path.read_text(encoding="utf-8", errors="ignore"))
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    return ""


@dataclass
class LocalChunk:
    id: str
    source_path: str
    title: str
    text: str


class LocalDocIndex:
    """Persistent local-doc chunk index with simple TF-IDF scoring."""

    def __init__(self, doc_root: Path, cache_dir: Path = Path(".cache/local_docs")):
        self.doc_root = doc_root
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"

        self._chunks: list[LocalChunk] = []
        self._df: dict[str, int] = {}
        self._n_chunks: int = 0

    def load_or_build(self) -> None:
        if self._try_load():
            return
        self.build()

    def _try_load(self) -> bool:
        if not self.index_file.exists():
            return False
        try:
            raw = json.loads(self.index_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return False

            meta = raw.get("meta", {})
            if meta.get("doc_root") != str(self.doc_root.resolve()):
                return False

            # If any file changed since build, rebuild
            built_at = meta.get("built_at")
            if not built_at:
                return False

            last_build = datetime.fromisoformat(built_at)
            for p in self._iter_docs():
                try:
                    mtime = datetime.fromtimestamp(p.stat().st_mtime)
                except Exception:
                    continue
                if mtime > last_build:
                    return False

            chunks_raw = raw.get("chunks", [])
            df_raw = raw.get("df", {})
            if not isinstance(chunks_raw, list) or not isinstance(df_raw, dict):
                return False

            self._chunks = [
                LocalChunk(
                    id=str(c.get("id", "")),
                    source_path=str(c.get("source_path", "")),
                    title=str(c.get("title", "")),
                    text=str(c.get("text", "")),
                )
                for c in chunks_raw
                if isinstance(c, dict)
            ]
            self._df = {str(k): int(v) for k, v in df_raw.items() if isinstance(v, int)}
            self._n_chunks = int(raw.get("n_chunks") or len(self._chunks))

            logger.info(f"Loaded local-doc index: {len(self._chunks)} chunks")
            return True
        except Exception as e:
            logger.warning(f"Failed to load local-doc index: {e}")
            return False

    def _iter_docs(self) -> Iterable[Path]:
        if not self.doc_root.exists() or not self.doc_root.is_dir():
            return []
        return [p for p in self.doc_root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]

    def build(self) -> None:
        logger.info(f"Building local-doc index from: {self.doc_root}")

        chunks: list[LocalChunk] = []
        df: dict[str, int] = {}

        for path in self._iter_docs():
            try:
                text = _read_file_text(path)
            except Exception as e:
                logger.warning(f"Skipping {path} (read failed): {e}")
                continue

            if not text or len(text.strip()) < 50:
                continue

            file_rel = str(path.relative_to(self.doc_root))
            title = path.stem

            file_chunks = _chunk_text(text)
            for idx, chunk in enumerate(file_chunks):
                cid = f"{file_rel}::chunk-{idx}"
                chunks.append(LocalChunk(id=cid, source_path=file_rel, title=title, text=chunk))

                unique_terms = set(_tokenize(chunk))
                for t in unique_terms:
                    df[t] = df.get(t, 0) + 1

        self._chunks = chunks
        self._df = df
        self._n_chunks = len(chunks)

        payload = {
            "meta": {
                "doc_root": str(self.doc_root.resolve()),
                "built_at": datetime.now().isoformat(),
                "supported_exts": sorted(SUPPORTED_EXTS),
            },
            "n_chunks": self._n_chunks,
            "df": self._df,
            "chunks": [{"id": c.id, "source_path": c.source_path, "title": c.title, "text": c.text} for c in chunks],
        }

        self.index_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Built local-doc index: {self._n_chunks} chunks")

    def search(self, query: str, *, k: int = 5) -> list[dict[str, Any]]:
        query_terms = _tokenize(query)
        if not query_terms or not self._chunks:
            return []

        # Simple TF-IDF scoring over chunks
        n = max(1, self._n_chunks)
        idf: dict[str, float] = {}
        for t in query_terms:
            df = self._df.get(t, 0)
            idf[t] = math.log((n + 1) / (df + 1)) + 1.0

        scored: list[tuple[float, LocalChunk]] = []
        for chunk in self._chunks:
            tokens = _tokenize(chunk.text)
            if not tokens:
                continue
            tf: dict[str, int] = {}
            for tok in tokens:
                if tok in idf:
                    tf[tok] = tf.get(tok, 0) + 1

            if not tf:
                continue

            score = 0.0
            for tok, cnt in tf.items():
                score += (cnt / max(1, len(tokens))) * idf[tok]

            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        out: list[dict[str, Any]] = []
        for score, chunk in scored[: max(1, k)]:
            snippet = chunk.text[:300].replace("\n", " ")
            out.append(
                {
                    "title": chunk.title,
                    "path": chunk.source_path,
                    "snippet": snippet,
                    "content": chunk.text,
                    "score": round(score, 6),
                }
            )

        return out


def get_default_index() -> LocalDocIndex | None:
    if not config.local_docs_enabled:
        return None
    if not config.doc_path:
        return None
    root = Path(config.doc_path).expanduser()
    if not root.exists() or not root.is_dir():
        return None
    return LocalDocIndex(root)
