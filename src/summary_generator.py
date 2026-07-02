import re
from typing import List, Optional, Tuple

from prompt_templates import TEMPLATES
from utils.config import DEFAULT_MODEL
from utils.rag_components import invoke_with_fallback

# Tunable constants (characters approximating token limits)
SUMMARY_SINGLEPASS_CHARS = 5500
SUMMARY_BATCH_CHARS = 3500
SUMMARY_REDUCE_CHARS = 3000


def normalize_pdf_text(text: str) -> str:
    """Normalize OCR-like PDF text before summary generation.

    Some PDFs are extracted with spaces between nearly every character.
    When that pattern is detected, collapse the letter spacing so the
    summarizer sees readable words instead of character soup.
    """
    if not text:
        return ""

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        tokens = line.split()
        if len(tokens) >= 8:
            single_letter_tokens = sum(1 for token in tokens if len(token) == 1 and token.isalpha())
            if single_letter_tokens / len(tokens) >= 0.6:
                line = re.sub(r"(?<=\w)\s+(?=\w)", "", line)
                line = re.sub(r"\s+", " ", line).strip()

        cleaned_lines.append(line)

    if cleaned_lines:
        single_char_lines = sum(1 for line in cleaned_lines if len(line) == 1 and line.isprintable())
        if single_char_lines / len(cleaned_lines) >= 0.6:
            compact = re.sub(r"\s+", "", "".join(cleaned_lines))
            compact = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", compact)
            compact = re.sub(r"(?<=[a-zA-Z])(?=\d)", " ", compact)
            compact = re.sub(r"(?<=\d)(?=[a-zA-Z])", " ", compact)
            return compact

    return "\n".join(cleaned_lines)


def _extract_headings(text: str) -> List[str]:
    headings = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if len(s) <= 80 and (s.istitle() or s.lower().startswith("chapter") or re.match(r"^\d+\.", s)):
            if len(s.split()) <= 8:
                headings.append(s)
    return headings


def _chunk_pages_for_map(documents: List) -> List[Tuple[str, str]]:
    """Concatenate document pages into batches under SUMMARY_BATCH_CHARS chars."""
    batches: List[Tuple[str, str]] = []
    current_batch: List[str] = []
    current_len = 0

    for doc in documents:
        text = normalize_pdf_text(doc.page_content).strip()
        if not text:
            continue
        snippet = text
        # If the snippet itself is too large, truncate it
        if len(snippet) > SUMMARY_BATCH_CHARS:
            snippet = snippet[:SUMMARY_BATCH_CHARS]

        if current_len + len(snippet) > SUMMARY_BATCH_CHARS and current_batch:
            batches.append(("batch", "\n\n".join(current_batch)))
            current_batch = []
            current_len = 0

        current_batch.append(snippet)
        current_len += len(snippet)

    if current_batch:
        batches.append(("batch", "\n\n".join(current_batch)))

    return batches


def _invoke_llm(prompt: str, preferred_model: Optional[str] = None) -> str:
    resp = invoke_with_fallback(prompt, preferred_model=preferred_model or DEFAULT_MODEL)
    return resp.content


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "rate limit" in message
        or "rate_limit_exceeded" in message
        or "error code: 429" in message
        or "tpd" in message
    )


def _build_local_fallback_summary(documents: List, title: Optional[str], reason: str) -> dict:
    normalized_texts = [normalize_pdf_text(p.page_content).strip() for p in documents]
    all_text = "\n\n".join(text for text in normalized_texts if text)
    headings = _extract_headings(all_text)[:8]

    excerpts = []
    for idx, doc in enumerate(documents[:5], start=1):
        page_num = doc.metadata.get("page", idx - 1)
        excerpt = normalize_pdf_text(doc.page_content).strip().replace("\n", " ")[:280]
        if excerpt:
            label = f"Page {int(page_num) + 1 if isinstance(page_num, int) else page_num}"
            excerpts.append(f"- **{label}**: {excerpt}")

    overview = all_text.replace("\n", " ")[:900]
    heading_block = "\n".join(f"- {heading}" for heading in headings) if headings else "- No clear headings detected"
    excerpt_block = "\n".join(excerpts) if excerpts else "- No readable page excerpts available"

    answer = (
        f"# Summary: {title or 'Document'}\n\n"
        f"**Note:** LLM summarization was unavailable ({reason}). This is a local extractive fallback.\n\n"
        f"## Overview\n{overview}\n\n"
        f"## Detected Headings\n{heading_block}\n\n"
        f"## Page Excerpts\n{excerpt_block}"
    )

    return {"answer": answer, "sources": [], "source_type": "summary", "documents": documents}


def _should_use_local_fallback(all_text: str, documents: List) -> bool:
    if not all_text:
        return True

    compact_text = re.sub(r"\s+", "", all_text)
    if len(compact_text) > 2000 and len(all_text) > 0:
        space_ratio = all_text.count(" ") / len(all_text)
        if space_ratio < 0.08:
            return True

    if len(all_text) > 8_000:
        return True

    if len(documents) > 20 and len(all_text) > 4_000:
        return True

    return False


def _is_rate_limit_message(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        signal in message
        for signal in (
            "429",
            "413",
            "rate limit",
            "rate-limited",
            "rate_limit_exceeded",
            "tokens per day",
            "request too large",
            "too large for model",
            "message size",
            "quota",
        )
    )


def generate_summary(
    documents: List,
    doc_type: str,
    title: Optional[str] = "Document",
    preferred_model: Optional[str] = None,
) -> dict:
    """Map-reduce style summarization that respects the model TPM limit.

    - If total content fits a single-pass budget, run one call.
    - Otherwise: split into batches, summarize each (MAP), then synthesize (REDUCE).
    """
    # Build lightweight context pieces (use full pages but bounded later)
    normalized_texts = [normalize_pdf_text(p.page_content).strip() for p in documents]
    all_text = "\n\n".join(text for text in normalized_texts if text)
    total_chars = len(all_text)

    if _should_use_local_fallback(all_text, documents):
        return _build_local_fallback_summary(
            documents,
            title,
            "document text is too compact or too large for safe single-request summarization",
        )

    template = TEMPLATES.get(doc_type, TEMPLATES["General Document"])

    if total_chars <= SUMMARY_SINGLEPASS_CHARS:
        overview = all_text[:800] + ("..." if len(all_text) > 800 else "")
        headings = _extract_headings(all_text)
        headings_list = "\n".join(f"- {h}" for h in headings) if headings else ""
        sections = headings_list if headings_list else overview
        prompt = template.format(title=title, overview=overview, headings_list=headings_list, sections=sections)
        prompt += "\n\nContext:\n" + all_text
        prompt += "\n\nProduce a clean Markdown summary obeying the guidelines above."
        try:
            ans = _invoke_llm(prompt, preferred_model=preferred_model)
            return {"answer": ans, "sources": [], "source_type": "summary", "documents": documents}
        except Exception as exc:
            if _is_rate_limit_message(exc):
                return _build_local_fallback_summary(documents, title, str(exc))
            return {
                "answer": "The language model is currently unavailable. Please try again later.",
                "sources": [],
                "source_type": "summary",
                "documents": [],
            }

    # MAP-REDUCE path
    batches = _chunk_pages_for_map(documents)
    partials: List[str] = []

    for idx, (_, batch_text) in enumerate(batches):
        map_prompt = (
            f"Summarize the following document SECTION. Extract key points, headings present, tables and diagrams if any.\n\n"
            f"--- SECTION ---\n{batch_text}\n--- END SECTION ---\n\n"
            "Produce concise bullet-point notes or short paragraphs suitable for later synthesis."
        )
        try:
            resp = _invoke_llm(map_prompt, preferred_model=preferred_model)
            partials.append(f"[Batch {idx+1}]\n" + resp)
        except Exception as exc:
            if _is_rate_limit_message(exc):
                return _build_local_fallback_summary(documents, title, str(exc))
            partials.append(f"[Batch {idx+1}] (failed: {exc})")

    if not partials:
        return {
            "answer": "The language model is currently unavailable. Please try again later.",
            "sources": [],
            "source_type": "summary",
            "documents": [],
        }

    combined = "\n\n".join(partials)
    if len(combined) > SUMMARY_REDUCE_CHARS:
        combined = combined[:SUMMARY_REDUCE_CHARS] + "\n\n[...additional sections omitted...]"

    reduce_prompt = (
        f"You are an expert document analyst. Below are extracted key-point summaries from all sections.\n\n"
        f"Document title: {title}\n\n"
        "Guidelines:\n"
        "- Generate summaries only from the retrieved context. Do not use outside knowledge.\n"
        "- Do not infer missing information. Do not create headings unsupported by the document.\n"
        "- Preserve headings when present and summarise tables/figures.\n\n"
        f"--- EXTRACTED KEY POINTS ---\n{combined}\n--- END ---\n\n"
        "Produce a final clean Markdown summary using only supported headings and content."
    )

    try:
        final = _invoke_llm(reduce_prompt, preferred_model=preferred_model)
        return {"answer": final, "sources": [], "source_type": "summary", "documents": documents}
    except Exception as exc:
        if _is_rate_limit_message(exc):
            return _build_local_fallback_summary(documents, title, str(exc))
        return {
            "answer": "The language model is currently unavailable. Please try again later.",
            "sources": [],
            "source_type": "summary",
            "documents": [],
        }
