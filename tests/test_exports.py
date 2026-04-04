from pathlib import Path

from src.utils.exports import ReportExporter


def test_exporter_writes_md_html_txt(tmp_path: Path):
    content = "# Title\n\nHello [x](https://example.com)\n\n- a\n- b\n"
    exporter = ReportExporter()

    base = tmp_path / "out"
    md = exporter.export(content, base, format="markdown")
    html = exporter.export(content, base, format="html")
    txt = exporter.export(content, base, format="txt")

    assert md.exists() and md.suffix == ".md"
    assert html.exists() and html.suffix == ".html"
    assert txt.exists() and txt.suffix == ".txt"


def test_exporter_pdf_docx_optional_deps(tmp_path: Path):
    exporter = ReportExporter()
    base = tmp_path / "out"

    # These should either work (if deps installed) or raise a ValueError with a helpful message.
    for fmt in ("pdf", "docx"):
        try:
            out = exporter.export("# T\n\nBody\n", base, format=fmt)
            assert out.exists()
        except ValueError as e:
            msg = str(e).lower()
            assert "optional" in msg or "install" in msg
