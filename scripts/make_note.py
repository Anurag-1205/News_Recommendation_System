#!/usr/bin/env python3
"""Build the Q6 design note: report/design_note.md -> HTML (markdown-it) -> PDF (headless Chrome).

    make note        # = PYTHONPATH=. .venv/bin/python scripts/make_note.py

The brief asks for ~6 pages, 11 pt, 1-inch margins. Those are set here as CSS (`@page` margin,
body font-size) rather than trusted to a template, and the resulting page count is printed so the
target is checked. Images under report/ (leaderboard screenshots, gitignored) are inlined as
data URIs so the PDF is self-contained. No pandoc or LaTeX on this machine (C-040).
"""
import base64, re, subprocess, sys
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
SRC, HTML, PDF = ROOT / "report/design_note.md", ROOT / "report/design_note.html", ROOT / "report/design_note.pdf"
CHROME = "google-chrome"

CSS = """
@page { size: A4; margin: 1in; }
html { font-size: 11pt; }
body { font-family: "Liberation Serif", "Times New Roman", serif; font-size: 11pt; line-height: 1.32; color: #111; margin: 0; }
h1 { font-size: 17pt; margin: 0 0 4pt 0; }
h2 { font-size: 13pt; margin: 14pt 0 5pt 0; border-bottom: 0.6pt solid #888; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 10pt 0 3pt 0; }
p { margin: 0 0 6pt 0; text-align: justify; }
ul, ol { margin: 0 0 6pt 0; padding-left: 18pt; }
li { margin-bottom: 2pt; }
table { border-collapse: collapse; font-size: 9pt; margin: 4pt 0 8pt 0; width: 100%; page-break-inside: avoid; }
th, td { border-bottom: 0.4pt solid #bbb; padding: 2pt 4pt; vertical-align: top; text-align: left; }
th { border-bottom: 0.8pt solid #333; font-weight: 600; }
code { font-family: "Liberation Mono", "DejaVu Sans Mono", monospace; font-size: 8.8pt; }
pre { font-size: 8.4pt; line-height: 1.2; border: 0.4pt solid #bbb; padding: 4pt 6pt; page-break-inside: avoid; }
pre code { font-size: inherit; }
img { max-width: 100%; }
figure { margin: 6pt 0 8pt 0; page-break-inside: avoid; }
figcaption { font-size: 9pt; color: #333; margin-top: 2pt; }
hr { border: 0; border-top: 0.6pt solid #888; margin: 10pt 0; }
.small { font-size: 9.5pt; }
strong { font-weight: 600; }
"""


def inline_images(html: str) -> str:
    """Inline report/*.png|jpg as data URIs; drop a <figure> whose image is not on disk yet (the
    screenshots arrive when the leaderboards score), and say so on stderr."""
    def drop_missing(m):
        src = re.search(r'src="([^"]+)"', m.group(0))
        if src and not (ROOT / "report" / src.group(1)).exists():
            print(f"  figure dropped, no file: report/{src.group(1)}", file=sys.stderr)
            return ""
        return m.group(0)
    html = re.sub(r"<figure>.*?</figure>", drop_missing, html, flags=re.S)
    def sub(m):
        p = ROOT / "report" / m.group(1)
        mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        return f'src="data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"'
    return re.sub(r'src="([^"]+\.(?:png|jpg|jpeg))"', sub, html)


def main():
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = inline_images(md.render(SRC.read_text()))
    HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>A2 design note</title><style>{CSS}</style></head><body>{body}</body></html>")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True)
    raw = PDF.read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
    words = len(re.sub(r"[`*_#|>\-]", " ", SRC.read_text()).split())
    print(f"{PDF.relative_to(ROOT)}: {pages} pages, {raw.__len__()/1e6:.1f} MB; source {words} words")
    return 0 if pages else 1


if __name__ == "__main__":
    sys.exit(main())
