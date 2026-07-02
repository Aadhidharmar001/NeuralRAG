import os
import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from document_classifier import classify_documents
from summary_generator import generate_summary

KB = Path(__file__).parent / 'knowledge-base'
PDF = KB / 'UNIT V-LEARNING.pdf'
OUT = Path(__file__).parent / 'dry_summary.md'

if not PDF.exists():
    print('PDF not found:', PDF)
    raise SystemExit(1)

docs = PyPDFLoader(str(PDF)).load()
for doc in docs:
    doc.metadata['source'] = PDF.name
    doc.metadata['filename'] = PDF.name

pdf_map = {PDF.name: docs}
doc_type = classify_documents(pdf_map)
result = generate_summary(docs, doc_type, title=PDF.name)

md = result.get('answer', '')
OUT.write_text(md, encoding='utf-8')
print('Wrote', OUT)
