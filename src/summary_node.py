from typing import List, Optional
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from document_classifier import classify_documents
from summary_generator import generate_summary


def get_knowledge_base_dir(workspace: Optional[str] = None) -> str:
    if workspace:
        return os.path.join("knowledge-base", workspace)
    return "knowledge-base"


def load_all_pdf_pages(workspace: Optional[str] = None) -> dict:
    kb_dir = get_knowledge_base_dir(workspace)
    pdf_map = {}
    if not os.path.exists(kb_dir):
        return pdf_map
    for file_name in sorted(os.listdir(kb_dir)):
        if not file_name.lower().endswith(".pdf"):
            continue
        file_path = os.path.join(kb_dir, file_name)
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            for doc in pages:
                doc.metadata["source"] = file_name
                doc.metadata["filename"] = file_name
            pdf_map[file_name] = pages
        except Exception:
            # keep running — streamlit UI will show warnings
            continue
    return pdf_map


def summarize_node(state: dict) -> dict:
    """Refactored summarize node: classify, choose prompt, generate Markdown summary."""
    pdf_map = load_all_pdf_pages()
    if not pdf_map:
        return {
            "answer": (
                "⚠️ **No documents found.**\n\n"
                "Please upload one or more PDFs using the sidebar, then click Index."
            ),
            "sources": [],
            "source_type": "summary_no_docs",
            "documents": [],
        }

    total_pages = sum(len(p) for p in pdf_map.values())
    all_docs: List[Document] = [doc for pages in pdf_map.values() for doc in pages]

    # Classify document type (simple heuristics)
    doc_type = classify_documents(pdf_map)

    # Title heuristic: first filename
    title = next(iter(pdf_map.keys()))
    preferred_model = state.get("preferred_model")

    # Generate summary via shared generator
    result = generate_summary(all_docs, doc_type, title=title, preferred_model=preferred_model)
    # Attach human-readable sources
    result.setdefault("sources", [f"PDF: {title} ({total_pages} pages)"])
    return result
