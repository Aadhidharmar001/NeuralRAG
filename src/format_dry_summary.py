from pathlib import Path
import PyPDF2
from types import SimpleNamespace
from document_classifier import classify_documents
from prompt_templates import TEMPLATES

KB = Path(__file__).parent / 'knowledge-base'
PDF = KB / 'UNIT V-LEARNING.pdf'
OUT = Path(__file__).parent / 'formatted_summary.md'

if not PDF.exists():
    print('PDF not found:', PDF)
    raise SystemExit(1)

reader = PyPDF2.PdfReader(str(PDF))
pages = []
for p in reader.pages:
    try:
        text = p.extract_text() or ''
    except Exception:
        text = ''
    pages.append(SimpleNamespace(page_content=text))

pdf_map = {PDF.name: pages}

doc_type = classify_documents(pdf_map)

title = PDF.stem
full_text = '\n\n'.join(p.page_content for p in pages)
overview = full_text.strip().replace('\n', ' ')[:600]

# Heuristic headings: look for lines that look like headings (all caps or numbered starts)
headings = []
for p in pages:
    lines = [ln.strip() for ln in p.page_content.splitlines() if ln.strip()]
    if not lines:
        continue
    # prefer all-caps short lines
    found = None
    for ln in lines[:6]:
        if len(ln) < 120 and (ln.isupper() or ln.lower().startswith('chapter') or ln[0].isdigit()):
            found = ln
            break
    if not found:
        # fallback: first short line
        for ln in lines[:6]:
            if len(ln) < 120:
                found = ln
                break
    if found:
        headings.append(found)

headings_list_md = '\n'.join(f'- {h}' for h in headings[:12]) if headings else ''

# sections: include page excerpts as mini-sections
sections_md_lines = []
for i, p in enumerate(pages, start=1):
    text = p.page_content.strip().replace('\n', ' ')
    if not text:
        continue
    excerpt = text[:400]
    sections_md_lines.append(f'- Page {i}: {excerpt}')
sections_md = '\n'.join(sections_md_lines)

template = TEMPLATES.get(doc_type, TEMPLATES['General Document'])
filled = template.format(title=title, overview=overview, headings_list=headings_list_md, sections=sections_md)
OUT.write_text(filled, encoding='utf-8')
print('Wrote', OUT)
