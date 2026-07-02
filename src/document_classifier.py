from typing import List


def classify_documents(pdf_map: dict) -> str:
    """Heuristic document classifier based on keyword and heading signals.

    Returns one of the canonical types expected by the summarizer.
    """
    text = "\n".join(
        page.page_content for pages in pdf_map.values() for page in pages
    ).lower()

    # Research paper signals
    if any(k in text for k in ("abstract", "introduction", "method", "methods", "results", "discussion")):
        return "Research Paper"

    # Resume / CV
    if any(k in text for k in ("curriculum vitae", "resume", "education", "skills", "experience", "certifications")):
        return "Resume"

    # Legal document
    if any(k in text for k in ("whereas", "hereinafter", "witnesseth", "agreement", "party", "parties", "clause")):
        return "Legal Document"

    # User manual
    if any(k in text for k in ("installation", "usage", "user manual", "system requirements", "how to")):
        return "User Manual"

    # Technical documentation
    if any(k in text for k in ("api", "endpoint", "sdk", "architecture", "specification", "configuration")):
        return "Technical Documentation"

    # Business report
    if any(k in text for k in ("executive summary", "findings", "recommendation", "reporting", "background")):
        return "Business Report"

    # Textbook / Lecture notes
    if any(k in text for k in ("chapter", "exercise", "example", "definition", "summary", "learning", "lecture")):
        return "Textbook / Study Material"

    # Book chapter
    if "chapter" in text and "chapter" in text[:2000]:
        return "Book Chapter"

    return "General Document"
