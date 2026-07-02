import streamlit as st
import os
import re            # ← P2: domain validation
import json
import time
from dotenv import load_dotenv
from typing import List, Optional
from typing_extensions import TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph

from utils.config import DEFAULT_MODEL
from utils.rag_components import invoke_with_fallback


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION — must be FIRST Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralRAG · Intelligent Research Agent",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE          = 1000
CHUNK_OVERLAP       = 200
LLM_MODEL_ID        = DEFAULT_MODEL
RETRIEVAL_K         = 3
RETRIEVAL_THRESHOLD = 0.1
HISTORY_WINDOW      = 5
WEB_MAX_RESULTS     = 3

# ── P2: Domain validation regex ──────────────────────────────────────────────
_DOMAIN_RE = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$"
)

# ── SUMMARIZATION: intent-detection patterns ──────────────────────────────────
# Matches summary-style queries so they bypass chunk retrieval and use full-doc pipeline.
_SUMMARY_INTENT_RE = re.compile(
    r"\b("
    r"summar(?:ize|ise|y|ies|ization|isation)|"
    r"overview|"
    r"give\s+(?:me\s+)?(?:a\s+)?(?:brief\s+|quick\s+|short\s+)?(?:overview|summary|intro(?:duction)?)|"
    r"what\s+is\s+this\s+(?:pdf|document|file|report|paper)\s+about|"
    r"what(?:'s|\s+is)\s+(?:the\s+)?(?:document|pdf|file|report|paper)\s+about|"
    r"explain\s+(?:this\s+)?(?:document|pdf|file|report|paper)|"
    r"describe\s+(?:this\s+)?(?:document|pdf|file|report|paper)|"
    r"(?:tell|explain)\s+me\s+(?:about\s+)?this\s+(?:document|pdf|file|report|paper)|"
    r"what\s+(?:does\s+)?this\s+(?:document|pdf|file|report)\s+(?:say|cover|contain|discuss)|"
    r"summarize\s+(?:the\s+)?project|"
    r"project\s+summar(?:y|ize)|"
    r"document\s+summar(?:y|ize)|"
    r"full\s+summar(?:y|ize)|"
    r"complete\s+summar(?:y|ize)"
    r")\b",
    re.IGNORECASE,
)


def is_summary_intent(question: str) -> bool:
    """Return True when the question is a document-summarization request."""
    return bool(_SUMMARY_INTENT_RE.search(question))


# ─────────────────────────────────────────────────────────────────────────────
# P5 — WORKSPACE-READY PATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_knowledge_base_dir(workspace: Optional[str] = None) -> str:
    if workspace:
        return os.path.join("knowledge-base", workspace)
    return "knowledge-base"


def get_persist_directory(workspace: Optional[str] = None) -> str:
    if workspace:
        return os.path.join("chroma_db", workspace)
    return "chroma_db"


def get_trusted_sources_file(workspace: Optional[str] = None) -> str:
    if workspace:
        os.makedirs("trusted_sources", exist_ok=True)
        return os.path.join("trusted_sources", f"{workspace}.json")
    return "trusted_sources.json"


KNOWLEDGE_BASE_DIR   = get_knowledge_base_dir()
PERSIST_DIRECTORY    = get_persist_directory()
TRUSTED_SOURCES_FILE = get_trusted_sources_file()


# ─────────────────────────────────────────────────────────────────────────────
# TRUSTED-SOURCES HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_trusted_sources(workspace: Optional[str] = None) -> List[str]:
    path = get_trusted_sources_file(workspace)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("trusted_sources", [])
    except (json.JSONDecodeError, OSError):
        return []


def save_trusted_sources(domains: List[str], workspace: Optional[str] = None) -> None:
    path = get_trusted_sources_file(workspace)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"trusted_sources": domains}, fh, indent=2)


def domain_from_url(url: str) -> str:
    stripped = url.replace("https://", "").replace("http://", "").replace("www.", "")
    return stripped.split("/")[0].split("?")[0].lower()


# ─────────────────────────────────────────────────────────────────────────────
# P1 — ENFORCE TRUST POLICY
# ─────────────────────────────────────────────────────────────────────────────

def is_trusted(url: str, trusted_domains: List[str]) -> bool:
    if not trusted_domains:
        return False
    d = domain_from_url(url)
    return any(d == td.lower() or d.endswith("." + td.lower()) for td in trusted_domains)


# ─────────────────────────────────────────────────────────────────────────────
# P2 — DOMAIN VALIDATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def validate_domain(raw: str) -> tuple[bool, str]:
    clean = (
        raw.strip()
        .lower()
        .replace("https://", "")
        .replace("http://", "")
        .replace("www.", "")
        .split("/")[0]
        .split("?")[0]
        .split(":")[0]
    )
    if not clean:
        return False, "Domain cannot be empty."
    if not _DOMAIN_RE.match(clean):
        return False, (
            f"'{clean}' is not a valid domain. "
            "Enter a bare domain like **who.int** or **arxiv.org** — "
            "no schemes, paths, or single words."
        )
    return True, clean


# ─────────────────────────────────────────────────────────────────────────────
# PDF UPLOAD HELPER
# ─────────────────────────────────────────────────────────────────────────────

def save_uploaded_pdfs(uploaded_files, workspace: Optional[str] = None) -> List[str]:
    kb_dir = get_knowledge_base_dir(workspace)
    os.makedirs(kb_dir, exist_ok=True)
    saved = []
    for uf in uploaded_files:
        dest = os.path.join(kb_dir, uf.name)
        with open(dest, "wb") as fh:
            fh.write(uf.getbuffer())
        saved.append(uf.name)
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# UNCHANGED HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_source_label(document: Document) -> str:
    source = document.metadata.get("source")
    title  = document.metadata.get("title")
    page   = document.metadata.get("page")

    if source and isinstance(source, str) and source.startswith(("http://", "https://")):
        if title:
            label = f"Web: {title} ({source})"
        else:
            label = f"Web: {source}"
    else:
        filename = os.path.basename(str(source)) if source else "uploaded document"
        if page is not None:
            label = f"PDF: {filename} (page {int(page) + 1})"
        else:
            label = f"PDF: {filename}"

    return label


def _build_source_summary(documents: List[Document]) -> List[str]:
    seen_sources = []
    for document in documents:
        label = _normalize_source_label(document)
        if label not in seen_sources:
            seen_sources.append(label)
    return seen_sources


def _format_history(history: List[dict]) -> str:
    if not history:
        return "No prior conversation."
    formatted_turns = []
    for message in history[-HISTORY_WINDOW:]:
        role    = message.get("role", "user").capitalize()
        content = message.get("content", "").strip()
        if content:
            formatted_turns.append(f"{role}: {content}")
    return "\n".join(formatted_turns) if formatted_turns else "No prior conversation."


def _friendly_model_error_message(error: Exception) -> str:
    message = str(error).lower()
    if any(signal in message for signal in ("429", "rate limit", "rate_limit_exceeded", "tokens per day", "quota")):
        return (
            "The selected model is temporarily rate-limited. Please retry in a bit or switch to another model from the sidebar."
        )
    return "The language model is currently unavailable. Please try again later."


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── DESIGN TOKENS ───────────────────────────────────────────────────────── */
:root {
  /* Backgrounds — layered depth */
  --bg-base:       #070b14;
  --bg-surface:    #0c1120;
  --bg-elevated:   #111827;
  --bg-card:       #131d2e;
  --bg-hover:      #1a2540;

  /* Borders */
  --border-subtle: rgba(56,189,248,0.08);
  --border-dim:    rgba(56,189,248,0.14);
  --border-mid:    rgba(56,189,248,0.28);
  --border-bright: rgba(56,189,248,0.55);

  /* Accent palette */
  --accent:        #38bdf8;
  --accent-dim:    rgba(56,189,248,0.12);
  --accent-mid:    rgba(56,189,248,0.22);
  --accent-glow:   rgba(56,189,248,0.35);

  /* Semantic colours */
  --green:         #34d399;
  --green-dim:     rgba(52,211,153,0.12);
  --green-border:  rgba(52,211,153,0.28);
  --amber:         #fbbf24;
  --amber-dim:     rgba(251,191,36,0.12);
  --amber-border:  rgba(251,191,36,0.28);
  --red:           #f87171;
  --red-dim:       rgba(248,113,113,0.12);
  --red-border:    rgba(248,113,113,0.28);
  --purple:        #a78bfa;
  --purple-dim:    rgba(167,139,250,0.12);
  --purple-border: rgba(167,139,250,0.28);

  /* Typography */
  --text-high:     #f1f5f9;
  --text-mid:      #94a3b8;
  --text-low:      #4b5e75;
  --font-sans:     'Inter', system-ui, sans-serif;
  --font-mono:     'JetBrains Mono', 'Fira Code', monospace;

  /* Shape */
  --radius-sm:  6px;
  --radius-md:  10px;
  --radius-lg:  16px;
  --radius-xl:  20px;

  /* Shadows */
  --shadow-card:  0 1px 3px rgba(0,0,0,0.4), 0 4px 16px rgba(0,0,0,0.25);
  --shadow-glow:  0 0 0 1px var(--border-bright), 0 0 20px var(--accent-glow), 0 0 40px rgba(56,189,248,0.08);
  --shadow-soft:  0 2px 8px rgba(0,0,0,0.3);
}

/* ── RESETS ──────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
  background: var(--bg-base) !important;
  color: var(--text-high) !important;
  font-family: var(--font-sans) !important;
  font-size: 15px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

/* Subtle grid backdrop — just enough to feel alive */
body::before {
  content: '';
  position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(56,189,248,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(56,189,248,0.025) 1px, transparent 1px);
  background-size: 56px 56px;
  pointer-events: none;
  z-index: 0;
}

/* Radial glow — top-right atmosphere */
body::after {
  content: '';
  position: fixed; top: -160px; right: -160px;
  width: 700px; height: 700px;
  background: radial-gradient(circle, rgba(56,189,248,0.055) 0%, transparent 65%);
  pointer-events: none;
  z-index: 0;
}

/* ── STREAMLIT CHROME OVERRIDES ──────────────────────────────────────────── */
[data-testid='stHeader']                            { background: transparent !important; height: 0 !important; min-height: 0 !important; overflow: visible !important; }
#MainMenu                                           { display: none !important; }
footer                                              { display: none !important; }
[data-testid='stDecoration']                        { display: none !important; }
[data-testid='stToolbar'] [data-testid='stToolbarActions']   { display: none !important; }
[data-testid='stToolbar'] [data-testid='stAppDeployButton']  { display: none !important; }
[data-testid='stStatusWidget']                      { display: none !important; }
div[data-testid="stChatMessage"]                    { background: transparent !important; padding: 0 !important; }
.block-container { padding: 2rem 2.5rem 5rem !important; max-width: 900px !important; margin: 0 auto !important; }

/* ── SCROLLBAR ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-dim); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── SIDEBAR ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] > div {
  padding: 0 !important;
}

/* Brand lockup */
.sb-brand {
  display: flex; align-items: center; gap: 11px;
  padding: 1.5rem 1.25rem 1.25rem;
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 0;
}
.sb-brand-icon {
  width: 34px; height: 34px;
  background: linear-gradient(135deg, var(--accent-mid), var(--accent-dim));
  border: 1px solid var(--border-mid);
  border-radius: var(--radius-md);
  display: flex; align-items: center; justify-content: center;
  font-size: 17px;
  box-shadow: var(--shadow-soft);
  flex-shrink: 0;
}
.sb-brand-text { line-height: 1.2; }
.sb-brand-name {
  font-family: var(--font-sans);
  font-weight: 700;
  font-size: 1rem;
  letter-spacing: -0.01em;
  color: var(--text-high);
}
.sb-brand-name em { font-style: normal; color: var(--accent); }
.sb-brand-sub {
  font-size: 0.72rem;
  color: var(--text-low);
  letter-spacing: 0.02em;
}

/* Section card container */
.sb-section {
  margin: 0;
  padding: 1rem 1.25rem;
  border-bottom: 1px solid var(--border-subtle);
}
.sb-section-last { border-bottom: none; }

/* Section header label */
.sb-label {
  display: flex; align-items: center; gap: 6px;
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-low);
  margin-bottom: 0.75rem;
}
.sb-label-icon { font-size: 12px; opacity: 0.7; }

/* File row */
.sb-file-row {
  display: flex; align-items: center; gap: 8px;
  padding: 7px 10px;
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  margin-bottom: 4px;
  transition: border-color 0.18s, background 0.18s;
}
.sb-file-row:hover { border-color: var(--border-dim); background: var(--bg-hover); }
.sb-file-icon { color: var(--accent); font-size: 13px; flex-shrink: 0; }
.sb-file-name {
  font-size: 0.78rem;
  color: var(--text-mid);
  font-family: var(--font-mono);
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Status pill */
.sb-status {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 9px;
  border-radius: 100px;
  font-size: 0.72rem;
  font-weight: 600;
  font-family: var(--font-mono);
  letter-spacing: 0.03em;
  margin-bottom: 8px;
}
.sb-status.ok     { background: var(--green-dim);  color: var(--green);  border: 1px solid var(--green-border); }
.sb-status.warn   { background: var(--amber-dim);  color: var(--amber);  border: 1px solid var(--amber-border); }
.sb-status.err    { background: var(--red-dim);    color: var(--red);    border: 1px solid var(--red-border); }
.sb-status.info   { background: var(--purple-dim); color: var(--purple); border: 1px solid var(--purple-border); }
.sb-status .dot   { width: 5px; height: 5px; border-radius: 50%; background: currentColor; flex-shrink: 0; animation: pulse 2.2s infinite; }

/* System stat row */
.sb-stat-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 5px 0;
  font-size: 0.78rem;
}
.sb-stat-label { color: var(--text-low); }
.sb-stat-value { color: var(--text-mid); font-family: var(--font-mono); font-size: 0.72rem; }
.sb-stat-value.ok     { color: var(--green); }
.sb-stat-value.err    { color: var(--red); }

/* Info chip */
.sb-info-chip {
  display: flex; align-items: flex-start; gap: 7px;
  padding: 6px 9px;
  background: var(--accent-dim);
  border: 1px solid var(--border-dim);
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  color: var(--text-mid);
  margin-top: 6px;
  line-height: 1.45;
}
.sb-info-chip-icon { color: var(--accent); margin-top: 1px; flex-shrink: 0; }

/* ── STREAMLIT WIDGET OVERRIDES (sidebar) ────────────────────────────────── */
section[data-testid="stSidebar"] .stRadio > label { display: none !important; }
section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
  font-size: 0.85rem !important;
  color: var(--text-mid) !important;
  font-family: var(--font-sans) !important;
}
section[data-testid="stSidebar"] .stFileUploader label { display: none !important; }
section[data-testid="stSidebar"] .stFileUploader section {
  background: var(--bg-card) !important;
  border: 1px dashed var(--border-dim) !important;
  border-radius: var(--radius-md) !important;
  padding: 0.6rem !important;
}
section[data-testid="stSidebar"] .stFileUploader section:hover {
  border-color: var(--border-mid) !important;
}
section[data-testid="stSidebar"] .stTextInput input {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-high) !important;
  font-size: 0.85rem !important;
  padding: 8px 10px !important;
  font-family: var(--font-mono) !important;
}
section[data-testid="stSidebar"] .stTextInput input:focus {
  border-color: var(--border-bright) !important;
  box-shadow: 0 0 0 3px var(--accent-dim) !important;
  outline: none !important;
}
section[data-testid="stSidebar"] .stTextInput label { display: none !important; }

/* ── MAIN BUTTONS ────────────────────────────────────────────────────────── */
.stButton > button {
  background: var(--bg-elevated) !important;
  color: var(--text-mid) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font-sans) !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.01em !important;
  padding: 0.45rem 1rem !important;
  transition: background 0.18s, border-color 0.18s, color 0.18s, box-shadow 0.18s !important;
}
.stButton > button:hover {
  background: var(--bg-hover) !important;
  border-color: var(--border-mid) !important;
  color: var(--text-high) !important;
  box-shadow: var(--shadow-soft) !important;
}

/* Primary action button variant */
.stButton.primary-btn > button,
.primary-btn .stButton > button {
  background: var(--accent) !important;
  color: #070b14 !important;
  border-color: var(--accent) !important;
  font-weight: 600 !important;
}
.stButton.primary-btn > button:hover,
.primary-btn .stButton > button:hover {
  background: #7dd3fc !important;
  border-color: #7dd3fc !important;
  box-shadow: 0 0 16px var(--accent-glow) !important;
}

/* ── CHAT INPUT ──────────────────────────────────────────────────────────── */
.stChatInput > div {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: var(--radius-lg) !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stChatInput > div:focus-within {
  border-color: var(--border-bright) !important;
  box-shadow: var(--shadow-glow) !important;
}
.stChatInput textarea {
  background: transparent !important;
  color: var(--text-high) !important;
  font-family: var(--font-sans) !important;
  font-size: 0.95rem !important;
  line-height: 1.55 !important;
}
.stChatInput textarea::placeholder { color: var(--text-low) !important; }

/* ── EXPANDER ────────────────────────────────────────────────────────────── */
div[data-testid="stExpander"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-md) !important;
  margin-top: 6px !important;
}
div[data-testid="stExpander"] summary {
  color: var(--text-low) !important;
  font-family: var(--font-sans) !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  padding: 0.6rem 1rem !important;
}
div[data-testid="stExpander"] summary:hover { color: var(--text-mid) !important; }

/* ── ALERTS ──────────────────────────────────────────────────────────────── */
div[data-testid="stAlert"] {
  border-radius: var(--radius-md) !important;
  font-family: var(--font-sans) !important;
  font-size: 0.85rem !important;
}
hr { border-color: var(--border-subtle) !important; margin: 0.75rem 0 !important; }

/* ── PAGE HEADER ─────────────────────────────────────────────────────────── */
.page-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 1.5rem 0 1.25rem;
  margin-bottom: 0.5rem;
  border-bottom: 1px solid var(--border-subtle);
  position: relative;
}
.page-header-left { display: flex; align-items: center; gap: 12px; }
.page-header-icon {
  width: 40px; height: 40px;
  background: linear-gradient(135deg, var(--accent-mid), var(--accent-dim));
  border: 1px solid var(--border-mid);
  border-radius: var(--radius-md);
  display: flex; align-items: center; justify-content: center;
  font-size: 20px;
  box-shadow: var(--shadow-soft);
}
.page-header-title {
  font-size: 1.35rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text-high);
  line-height: 1.1;
}
.page-header-title em { font-style: normal; color: var(--accent); }
.page-header-sub {
  font-size: 0.78rem;
  color: var(--text-low);
  margin-top: 2px;
  letter-spacing: 0.01em;
}
.page-header-right {
  display: flex; align-items: center; gap: 6px;
}
.live-pill {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 10px;
  background: var(--green-dim);
  border: 1px solid var(--green-border);
  border-radius: 100px;
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--green);
  font-family: var(--font-mono);
  letter-spacing: 0.06em;
}
.live-pill .dot { width: 5px; height: 5px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }

/* ── WELCOME / EMPTY STATE ───────────────────────────────────────────────── */
.welcome-wrap {
  display: flex; flex-direction: column; align-items: center;
  padding: 3.5rem 1rem 2rem;
  text-align: center;
}
.welcome-glyph {
  width: 64px; height: 64px;
  background: linear-gradient(135deg, var(--accent-mid), var(--accent-dim));
  border: 1px solid var(--border-mid);
  border-radius: 18px;
  display: flex; align-items: center; justify-content: center;
  font-size: 30px;
  margin-bottom: 1.5rem;
  box-shadow: 0 0 32px var(--accent-glow), var(--shadow-soft);
}
.welcome-title {
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: -0.025em;
  color: var(--text-high);
  margin-bottom: 0.5rem;
  line-height: 1.15;
}
.welcome-title em { font-style: normal; color: var(--accent); }
.welcome-sub {
  font-size: 0.95rem;
  color: var(--text-mid);
  max-width: 400px;
  line-height: 1.6;
  margin-bottom: 2.5rem;
}

/* Capability steps */
.welcome-steps {
  display: flex; gap: 1.5rem; justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 2.5rem;
  width: 100%; max-width: 540px;
}
.welcome-step {
  display: flex; flex-direction: column; align-items: center;
  gap: 6px;
  font-size: 0.8rem;
  color: var(--text-mid);
  width: 100px;
}
.welcome-step-icon {
  width: 38px; height: 38px;
  background: var(--bg-card);
  border: 1px solid var(--border-dim);
  border-radius: var(--radius-md);
  display: flex; align-items: center; justify-content: center;
  font-size: 17px;
  transition: border-color 0.2s, background 0.2s;
}
.welcome-step:hover .welcome-step-icon {
  border-color: var(--border-mid);
  background: var(--bg-hover);
}

/* Suggested prompts */
.welcome-prompts-label {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-low);
  margin-bottom: 0.75rem;
  align-self: flex-start;
  width: 100%; max-width: 560px;
}
.welcome-prompts {
  display: flex; flex-direction: column; gap: 7px;
  width: 100%; max-width: 560px;
}
.welcome-prompt-btn {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 14px;
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  font-size: 0.875rem;
  color: var(--text-mid);
  cursor: pointer;
  transition: background 0.18s, border-color 0.18s, color 0.18s;
  text-align: left;
}
.welcome-prompt-btn:hover {
  background: var(--bg-hover);
  border-color: var(--border-dim);
  color: var(--text-high);
}
.welcome-prompt-icon { font-size: 15px; flex-shrink: 0; }

/* ── CHAT MESSAGES ───────────────────────────────────────────────────────── */
.msg-wrap { display: flex; flex-direction: column; gap: 0; }

/* User bubble */
.msg-user {
  display: flex; justify-content: flex-end;
  margin: 0.75rem 0;
  animation: msgIn 0.22s ease;
}
.msg-user-bubble {
  max-width: 75%;
  background: linear-gradient(135deg, rgba(56,189,248,0.13), rgba(56,189,248,0.06));
  border: 1px solid var(--border-mid);
  border-radius: var(--radius-lg) var(--radius-lg) 4px var(--radius-lg);
  padding: 0.85rem 1.1rem;
  font-size: 0.925rem;
  color: var(--text-high);
  line-height: 1.65;
}

/* AI response */
.msg-ai {
  display: flex; align-items: flex-start; gap: 12px;
  margin: 0.75rem 0;
  animation: msgIn 0.22s ease;
}
.msg-ai-avatar {
  width: 32px; height: 32px;
  background: linear-gradient(135deg, var(--accent-mid), var(--accent-dim));
  border: 1px solid var(--border-mid);
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 15px;
  flex-shrink: 0;
  margin-top: 2px;
  box-shadow: var(--shadow-soft);
}
.msg-ai-body { flex: 1; min-width: 0; }
.msg-ai-meta {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 6px;
}
.msg-ai-name {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text-mid);
  letter-spacing: 0.01em;
}
.msg-ai-content {
  font-size: 0.925rem;
  color: var(--text-high);
  line-height: 1.75;
}
.msg-ai-content p { margin: 0 0 0.75em; }
.msg-ai-content p:last-child { margin-bottom: 0; }
.msg-ai-content strong { color: var(--text-high); font-weight: 600; }
.msg-ai-content em { color: var(--text-mid); }
.msg-ai-content code {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  padding: 1px 5px;
  font-family: var(--font-mono);
  font-size: 0.83em;
  color: var(--accent);
}
.msg-ai-content h1,.msg-ai-content h2,.msg-ai-content h3,.msg-ai-content h4 {
  color: var(--text-high);
  font-weight: 600;
  margin: 1.1em 0 0.45em;
  letter-spacing: -0.01em;
  line-height: 1.3;
}
.msg-ai-content h1 { font-size: 1.2em; }
.msg-ai-content h2 { font-size: 1.1em; }
.msg-ai-content h3 { font-size: 1em; }
.msg-ai-content ul,.msg-ai-content ol {
  padding-left: 1.4em;
  margin: 0.4em 0 0.8em;
}
.msg-ai-content li { margin-bottom: 0.3em; }
.msg-ai-content blockquote {
  border-left: 3px solid var(--border-mid);
  margin: 0.6em 0;
  padding-left: 0.9em;
  color: var(--text-mid);
}

/* Source footer pills */
.msg-sources {
  display: flex; flex-wrap: wrap; align-items: center; gap: 5px;
  margin-top: 10px;
  padding-top: 9px;
  border-top: 1px solid var(--border-subtle);
}
.msg-source-label {
  font-size: 0.7rem;
  color: var(--text-low);
  font-weight: 500;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-right: 2px;
}
.src-pill {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 8px;
  border-radius: 100px;
  font-size: 0.7rem;
  font-weight: 500;
  font-family: var(--font-mono);
  letter-spacing: 0.02em;
  border: 1px solid;
  transition: opacity 0.15s;
}
.src-pill:hover { opacity: 0.8; }
.src-pill.rag    { background: rgba(56,189,248,0.1);   color: var(--accent); border-color: rgba(56,189,248,0.25); }
.src-pill.web    { background: rgba(52,211,153,0.1);   color: var(--green);  border-color: rgba(52,211,153,0.25); }
.src-pill.sum    { background: rgba(167,139,250,0.1);  color: var(--purple); border-color: rgba(167,139,250,0.25); }
.src-pill.err    { background: rgba(248,113,113,0.1);  color: var(--red);    border-color: rgba(248,113,113,0.25); }

/* Source type badge (inline) */
.type-badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 7px;
  border-radius: var(--radius-sm);
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-family: var(--font-mono);
}
.type-badge.rag     { background: rgba(56,189,248,0.1);  color: var(--accent); }
.type-badge.web     { background: rgba(52,211,153,0.1);  color: var(--green);  }
.type-badge.summary { background: rgba(167,139,250,0.1); color: var(--purple); }
.type-badge.err     { background: rgba(248,113,113,0.1); color: var(--red);    }

/* ── SUMMARY CARD ────────────────────────────────────────────────────────── */
.summary-card {
  background: var(--bg-card);
  border: 1px solid var(--border-dim);
  border-radius: var(--radius-lg);
  overflow: hidden;
  margin-bottom: 4px;
}
.summary-header {
  background: linear-gradient(135deg, rgba(167,139,250,0.12), rgba(56,189,248,0.08));
  border-bottom: 1px solid var(--border-dim);
  padding: 1rem 1.25rem;
  display: flex; align-items: center; gap: 10px;
}
.summary-header-icon {
  width: 32px; height: 32px;
  background: var(--purple-dim);
  border: 1px solid var(--purple-border);
  border-radius: var(--radius-sm);
  display: flex; align-items: center; justify-content: center;
  font-size: 16px;
  flex-shrink: 0;
}
.summary-header-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-high);
  letter-spacing: -0.01em;
}
.summary-header-sub {
  font-size: 0.72rem;
  color: var(--text-low);
  margin-top: 1px;
  font-family: var(--font-mono);
}
.summary-body {
  padding: 1.1rem 1.25rem;
  font-size: 0.9rem;
  line-height: 1.75;
  color: var(--text-mid);
}
.summary-body strong { color: var(--text-high); font-weight: 600; }
.summary-body h1,.summary-body h2,.summary-body h3 {
  color: var(--text-high);
  font-weight: 600;
  margin: 1.1em 0 0.4em;
  letter-spacing: -0.01em;
}
.summary-body h2 { font-size: 0.95rem; color: var(--purple); }
.summary-body h3 { font-size: 0.88rem; }
.summary-body ul { padding-left: 1.3em; margin: 0.4em 0 0.8em; }
.summary-body li { margin-bottom: 0.3em; }

/* ── WORKFLOW TRACE ───────────────────────────────────────────────────────── */
.trace-wrap {
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 0.9rem 1rem;
  margin-bottom: 0.5rem;
}
.trace-title {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-low);
  margin-bottom: 0.65rem;
  font-family: var(--font-mono);
}
.trace-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 5px 0;
  border-bottom: 1px solid var(--border-subtle);
}
.trace-row:last-child { border-bottom: none; padding-bottom: 0; }
.trace-left { display: flex; align-items: center; gap: 8px; }
.trace-step-icon { font-size: 13px; width: 20px; text-align: center; flex-shrink: 0; }
.trace-step-name { font-size: 0.83rem; font-weight: 500; color: var(--text-high); }
.trace-step-desc { font-size: 0.75rem; color: var(--text-low); margin-left: 4px; }
.trace-status {
  font-size: 0.68rem;
  font-weight: 600;
  font-family: var(--font-mono);
  letter-spacing: 0.05em;
  padding: 2px 7px;
  border-radius: 100px;
}
.trace-status.running { background: var(--amber-dim);  color: var(--amber);  border: 1px solid var(--amber-border); }
.trace-status.done    { background: var(--green-dim);  color: var(--green);  border: 1px solid var(--green-border); }
.trace-status.pending { background: var(--bg-card);    color: var(--text-low); border: 1px solid var(--border-subtle); }
.trace-status.blocked { background: var(--red-dim);   color: var(--red);    border: 1px solid var(--red-border); }
.trace-status.skipped { background: var(--bg-card);    color: var(--text-low); border: 1px solid var(--border-subtle); opacity: 0.5; }

/* ── ANIMATIONS ──────────────────────────────────────────────────────────── */
@keyframes pulse  { 0%,100%{opacity:1} 50%{opacity:0.35} }
@keyframes msgIn  { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn { from{opacity:0} to{opacity:1} }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def ingest_pdfs_into_vectordb(workspace: Optional[str] = None):
    """(Re-)build ChromaDB from scratch using every PDF in the knowledge-base dir."""
    import shutil

    kb_dir   = get_knowledge_base_dir(workspace)
    persist  = get_persist_directory(workspace)

    documents = []
    if not os.path.exists(kb_dir):
        return 0
    for file_name in os.listdir(kb_dir):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(kb_dir, file_name)
            try:
                loader        = PyPDFLoader(file_path)
                pdf_documents = loader.load()
                for document in pdf_documents:
                    document.metadata["source"]   = file_name
                    document.metadata["filename"] = file_name
                documents.extend(pdf_documents)
            except Exception as e:
                st.warning(f"Error loading {file_path}: {e}")
    if not documents:
        return 0

    if os.path.exists(persist):
        shutil.rmtree(persist)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts         = text_splitter.split_documents(documents)
    embeddings    = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore   = Chroma.from_documents(texts, embeddings, persist_directory=persist)
    vectorstore.persist()
    return len(documents)


@st.cache_resource
def get_vectorstore():
    if not os.path.exists(PERSIST_DIRECTORY):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)


@st.cache_resource
def create_retriever():
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})


# ─────────────────────────────────────────────────────────────────────────────
# FULL-DOCUMENT LOADER (used by summarize_node)
# ─────────────────────────────────────────────────────────────────────────────

def load_all_pdf_pages(workspace: Optional[str] = None) -> dict[str, List[Document]]:
    """Load every page of every PDF directly from disk (bypasses vectorstore).

    Returns a dict mapping filename → list of Document (one per page).
    This gives summarize_node the entire document content, not just top-k chunks.
    """
    kb_dir = get_knowledge_base_dir(workspace)
    pdf_map: dict[str, List[Document]] = {}
    if not os.path.exists(kb_dir):
        return pdf_map
    for file_name in sorted(os.listdir(kb_dir)):
        if not file_name.lower().endswith(".pdf"):
            continue
        file_path = os.path.join(kb_dir, file_name)
        try:
            loader = PyPDFLoader(file_path)
            pages  = loader.load()
            for doc in pages:
                doc.metadata["source"]   = file_name
                doc.metadata["filename"] = file_name
            pdf_map[file_name] = pages
        except Exception as exc:
            st.warning(f"Could not load {file_name} for summarization: {exc}")
    return pdf_map


# ── Summarization strategy constants ─────────────────────────────────────────
# Groq TPM limit for this model is 8 000 tokens (~3 chars ≈ 1 token).
#
# SINGLE-PASS  (small docs ≤ SUMMARY_SINGLEPASS_PAGE_LIMIT pages)
#   All pages are concatenated and sent in ONE call.
#   Max content: SUMMARY_SINGLEPASS_CHARS chars ≈ 1 900 tokens.
#   With ~400-token prompt overhead → ~2 300 tokens total, well under 8k TPM.
#   Typical 10–30 page project report = 1 API call, low latency.
#
# MAP-REDUCE   (large docs > SUMMARY_SINGLEPASS_PAGE_LIMIT pages)
#   Pages split into SUMMARY_BATCH_CHARS batches; each batch → 1 MAP call,
#   then partial summaries merged in a single REDUCE call.
#   Scales to arbitrarily large documents.

SUMMARY_SINGLEPASS_PAGE_LIMIT = 50       # pages; docs at-or-below use single-pass
SUMMARY_SINGLEPASS_CHARS      = 5_500    # max total content chars for single-pass
SUMMARY_BATCH_CHARS           = 3_500    # max chars per map-step batch
SUMMARY_REDUCE_CHARS          = 3_000    # max chars of partial summaries for reduce


def _chunk_pages_for_map(
    pdf_map: dict[str, List[Document]]
) -> tuple[List[tuple[str, str]], List[str]]:
    """Split all PDF pages into small text batches safe for the 8k-TPM model.

    Returns:
        batches   — list of (filename_label, batch_text) tuples
        sources   — human-readable source labels for attribution
    """
    batches: List[tuple[str, str]] = []
    sources: List[str] = []

    for filename, pages in pdf_map.items():
        sources.append(f"PDF: {filename} ({len(pages)} pages)")
        current_batch: List[str] = []
        current_len = 0

        for page in pages:
            page_num = page.metadata.get("page", "?")
            text     = page.page_content.strip()
            if not text:
                continue
            page_str = f"[Page {int(page_num)+1 if isinstance(page_num, int) else page_num}]\n{text}"

            if current_len + len(page_str) > SUMMARY_BATCH_CHARS and current_batch:
                batches.append((filename, "\n\n".join(current_batch)))
                current_batch = []
                current_len   = 0

            current_batch.append(page_str)
            current_len += len(page_str)

        if current_batch:
            batches.append((filename, "\n\n".join(current_batch)))

    return batches, sources


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH STATE
# ─────────────────────────────────────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    question:         str
    history:          List[dict]
    documents:        List[Document]
    sources:          List[str]
    needs_web_search: bool
    source_type:      str
    answer:           str
    trusted_domains:  List[str]
    source_mode:      str   # "hybrid" | "docs_only" | "web_only"
    is_summary:       bool  # NEW: True when summarization intent detected
    preferred_model:  str


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────────────────────

def route_from_start(state: GraphState) -> str:
    """Primary router: send summary-intent queries to summarize_node, others to retrieve."""
    if state.get("is_summary"):
        return "summarize"
    return "retrieve"


def route_after_retrieval(state: GraphState) -> str:
    mode = state.get("source_mode", "hybrid")
    if mode == "docs_only":
        return "generate"
    return "web_search" if state.get("needs_web_search") else "generate"


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _total_page_count(pdf_map: dict[str, List[Document]]) -> int:
    return sum(len(pages) for pages in pdf_map.values())


def _build_singlepass_context(pdf_map: dict[str, List[Document]]) -> str:
    """Concatenate all page text up to SUMMARY_SINGLEPASS_CHARS.

    Pages are sampled evenly when the document is too long to fit in full,
    so the LLM always sees content from the beginning, middle, and end rather
    than just the first N pages.
    """
    all_pages: List[tuple[str, Document]] = []
    for filename, pages in pdf_map.items():
        for doc in pages:
            all_pages.append((filename, doc))

    # Quick check: does everything fit without sampling?
    full_text_parts: List[str] = []
    total = 0
    for filename, doc in all_pages:
        page_num = doc.metadata.get("page", "?")
        text     = doc.page_content.strip()
        if not text:
            continue
        part = f"[{filename} · Page {int(page_num)+1 if isinstance(page_num, int) else page_num}]\n{text}"
        if total + len(part) > SUMMARY_SINGLEPASS_CHARS:
            # Doesn't fit in full — fall through to sampling below
            break
        full_text_parts.append(part)
        total += len(part)
    else:
        # Loop completed without breaking → everything fits
        return "\n\n".join(full_text_parts)

    # Evenly-spaced sampling: pick indices spread across the full page list
    non_empty = [(fn, d) for fn, d in all_pages if d.page_content.strip()]
    if not non_empty:
        return ""

    # Estimate how many pages we can fit
    avg_page_chars = sum(len(d.page_content) for _, d in non_empty) / len(non_empty)
    max_pages = max(1, int(SUMMARY_SINGLEPASS_CHARS / max(avg_page_chars, 1)))
    step = max(1, len(non_empty) // max_pages)
    sampled = non_empty[::step][:max_pages]

    parts: List[str] = []
    total = 0
    for filename, doc in sampled:
        page_num = doc.metadata.get("page", "?")
        text     = doc.page_content.strip()
        part = f"[{filename} · Page {int(page_num)+1 if isinstance(page_num, int) else page_num}]\n{text}"
        if total + len(part) > SUMMARY_SINGLEPASS_CHARS:
            break
        parts.append(part)
        total += len(part)

    note = f"\n\n[Note: document sampled evenly — {len(sampled)} of {len(non_empty)} pages shown]"
    return "\n\n".join(parts) + note


_SUMMARY_STRUCTURED_SECTIONS = (
    "1. **Project / Document Objective** — What problem is solved or goal pursued?\n"
    "2. **Methodology** — What approach, techniques, or processes are described?\n"
    "3. **Technologies / Tools Used** — Software, hardware, frameworks, algorithms?\n"
    "4. **Workflow / Architecture** — System structure or process flow?\n"
    "5. **Results / Findings** — Outcomes, measurements, or findings?\n"
    "6. **Conclusion** — Key takeaways, limitations, or future directions?"
)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARIZE NODE
# ─────────────────────────────────────────────────────────────────────────────

def summarize_node(state: GraphState) -> GraphState:
    """Adaptive summarization: single-pass for small docs, map-reduce for large ones.

    Strategy selection (controlled by SUMMARY_SINGLEPASS_PAGE_LIMIT = 50):
      ≤ 50 pages → SINGLE-PASS: all content in one LLM call (~1 API request).
                   Fast, low latency, ideal for typical 10–30 page project reports.
      > 50 pages → MAP-REDUCE : pages batched into SUMMARY_BATCH_CHARS chunks,
                   each summarised independently, then merged in a REDUCE call.
                   Scales to arbitrarily large documents within the 8k TPM limit.
    """
    history  = state.get("history", [])
    pdf_map  = load_all_pdf_pages()

    if not pdf_map:
        return {
            "answer": (
                "⚠️ **No documents found.**\n\n"
                "Please upload one or more PDFs using the sidebar, "
                "then click **Ingest / Re-ingest PDFs** before requesting a summary."
            ),
            "sources":     [],
            "source_type": "summary_no_docs",
            "documents":   [],
        }

    total_pages  = _total_page_count(pdf_map)
    doc_list     = ", ".join(pdf_map.keys())
    source_labels = [f"PDF: {fn} ({len(pg)} pages)" for fn, pg in pdf_map.items()]
    all_docs     = [doc for pages in pdf_map.values() for doc in pages]
    history_text = _format_history(history)
    preferred_model = state.get("preferred_model") or LLM_MODEL_ID

    use_single_pass = total_pages <= SUMMARY_SINGLEPASS_PAGE_LIMIT
    strategy_label  = "single-pass" if use_single_pass else "map-reduce"
    print(
        f"[summarize_node] {total_pages} total page(s) across {len(pdf_map)} file(s) "
        f"→ using {strategy_label} strategy (threshold={SUMMARY_SINGLEPASS_PAGE_LIMIT})"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SINGLE-PASS PATH  (small documents)
    # ══════════════════════════════════════════════════════════════════════════
    if use_single_pass:
        context = _build_singlepass_context(pdf_map)

        prompt = (
            f"You are an expert document analyst. Document(s): {doc_list}\n\n"
            "Produce a **comprehensive structured summary** covering ALL sections below "
            "(omit only if the document has absolutely no relevant content for that section):\n\n"
            f"{_SUMMARY_STRUCTURED_SECTIONS}\n\n"
            "Guidelines:\n"
            "- Be thorough and specific — name actual technologies, metrics, methods.\n"
            "- Cover the entire document, not just the opening pages.\n"
            "- Use the exact bold headings listed above.\n\n"
            f"Recent conversation:\n{history_text}\n\n"
            f"--- DOCUMENT CONTENT ---\n{context}\n--- END ---\n\n"
            "Comprehensive structured summary:"
        )

        try:
            print("[summarize_node] Single-pass: invoking LLM (1 API call)")
            resp = invoke_with_fallback(prompt, preferred_model=preferred_model)
            return {
                "answer":      resp.content,
                "sources":     source_labels,
                "source_type": "summary",
                "documents":   all_docs,
            }
        except Exception as exc:
            print(f"[summarize_node] Single-pass failed: {exc}")
            return {
                "answer": _friendly_model_error_message(exc),
                "sources":     source_labels,
                "source_type": "summary",
                "documents":   [],
            }

    # ══════════════════════════════════════════════════════════════════════════
    # MAP-REDUCE PATH  (large documents)
    # ══════════════════════════════════════════════════════════════════════════
    batches, _ = _chunk_pages_for_map(pdf_map)
    partial_summaries: List[str] = []

    print(f"[summarize_node] Map-reduce: {len(batches)} batch(es) to process")

    # ── MAP phase ─────────────────────────────────────────────────────────────
    for idx, (filename, batch_text) in enumerate(batches):
        map_prompt = (
            f"You are summarizing a section of the document '{filename}' "
            f"(batch {idx + 1} of {len(batches)}).\n\n"
            "Extract and list the key points from this section. "
            "Focus on: objectives, methods, technologies, workflow steps, "
            "results, and conclusions. Be concise but complete.\n\n"
            f"--- DOCUMENT SECTION ---\n{batch_text}\n--- END SECTION ---\n\n"
            "Key points from this section:"
        )
        try:
            resp    = invoke_with_fallback(map_prompt, preferred_model=preferred_model)
            partial = resp.content.strip()
            partial_summaries.append(f"[From: {filename}, batch {idx + 1}]\n{partial}")
            print(f"[summarize_node] MAP {idx + 1}/{len(batches)} done ({len(partial)} chars)")
        except Exception as exc:
            print(f"[summarize_node] MAP batch {idx + 1} failed: {exc}")
            partial_summaries.append(
                f"[From: {filename}, batch {idx + 1}] (extraction failed: {exc})"
            )

    if not partial_summaries or all("extraction failed" in p for p in partial_summaries):
        return {
            "answer": "The language model is currently unavailable. Please try again later.",
            "sources":     source_labels,
            "source_type": "summary",
            "documents":   [],
        }

    # ── REDUCE phase ──────────────────────────────────────────────────────────
    combined = "\n\n".join(partial_summaries)
    if len(combined) > SUMMARY_REDUCE_CHARS:
        combined = combined[:SUMMARY_REDUCE_CHARS] + "\n\n[...additional sections omitted...]"

    reduce_prompt = (
        f"You are an expert document analyst. Below are key-point extracts "
        f"from ALL sections of the document(s): {doc_list}\n\n"
        "Synthesise these into a single **comprehensive structured summary** "
        "with the following sections "
        "(omit a section only if there is genuinely no relevant information):\n\n"
        f"{_SUMMARY_STRUCTURED_SECTIONS}\n\n"
        "Guidelines:\n"
        "- Cover the ENTIRE document, not just one part.\n"
        "- Use the exact bold headings listed above.\n"
        "- Be thorough and specific — name actual technologies, metrics, methods.\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"--- EXTRACTED KEY POINTS ---\n{combined}\n--- END ---\n\n"
        "Comprehensive structured summary:"
    )

    try:
        print("[summarize_node] REDUCE phase: generating final summary")
        resp = invoke_with_fallback(reduce_prompt, preferred_model=preferred_model)
        return {
            "answer":      resp.content,
            "sources":     source_labels,
            "source_type": "summary",
            "documents":   all_docs,
        }
    except Exception as exc:
        print(f"[summarize_node] REDUCE phase failed: {exc}")
        fallback = (
            "⚠️ **Final synthesis failed** — here are the per-section extracts:\n\n"
            + "\n\n".join(partial_summaries)
        )
        return {
            "answer":      fallback,
            "sources":     source_labels,
            "source_type": "summary",
            "documents":   [],
        }


# ─────────────────────────────────────────────────────────────────────────────
# EXISTING NODES (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_node(state: GraphState) -> GraphState:
    mode = state.get("source_mode", "hybrid")

    if mode == "web_only":
        print("[retrieve_node] source_mode=web_only → skipping RAG, forcing web fallback")
        return {"documents": [], "sources": [], "needs_web_search": True}

    question    = state["question"]
    vectorstore = get_vectorstore()
    retriever   = create_retriever()

    if vectorstore is None:
        print("[retrieve_node] vectorstore is None → web fallback")
        return {"documents": [], "sources": [], "needs_web_search": True}

    relevant_docs: List[Document] = []

    try:
        docs_and_scores = vectorstore.similarity_search_with_relevance_scores(
            question, k=RETRIEVAL_K
        )

        print(f"[retrieve_node] Raw (doc, score) pairs returned ({len(docs_and_scores)} total):")
        for i, (doc, score) in enumerate(docs_and_scores):
            snippet = doc.page_content[:80].replace("\n", " ")
            print(f"  [{i}] score={score:.4f} | {snippet}…")

        if not docs_and_scores:
            print("[retrieve_node] similarity_search returned 0 results → web fallback")
            return {"documents": [], "sources": [], "needs_web_search": True}

        all_scores = [score for _, score in docs_and_scores]
        max_score  = max(all_scores)

        if max_score > 1.0:
            DISTANCE_CEILING = 1.5
            relevant_docs = [doc for doc, score in docs_and_scores if score <= DISTANCE_CEILING]
            print(
                f"[retrieve_node] Scores appear to be raw distances (max={max_score:.4f}). "
                f"Keeping docs with distance ≤ {DISTANCE_CEILING}. "
                f"Surviving: {len(relevant_docs)}/{len(docs_and_scores)}"
            )
        else:
            relevant_docs = [doc for doc, score in docs_and_scores if score >= RETRIEVAL_THRESHOLD]
            print(
                f"[retrieve_node] Scores are similarities (max={max_score:.4f}). "
                f"Keeping docs with score ≥ {RETRIEVAL_THRESHOLD}. "
                f"Surviving: {len(relevant_docs)}/{len(docs_and_scores)}"
            )

    except Exception as exc:
        print(f"[retrieve_node] similarity_search_with_relevance_scores raised: {exc}")
        print("[retrieve_node] Falling back to retriever.invoke() (no score filter)")
        relevant_docs = retriever.invoke(question) if retriever is not None else []
        print(f"[retrieve_node] retriever.invoke() returned {len(relevant_docs)} docs")

    if not relevant_docs:
        print("[retrieve_node] No relevant docs after filtering → web fallback triggered")
        return {"documents": [], "sources": [], "needs_web_search": True}

    print(f"[retrieve_node] {len(relevant_docs)} relevant docs found → using RAG, no web fallback")
    return {
        "documents":        relevant_docs,
        "sources":          _build_source_summary(relevant_docs),
        "needs_web_search": False,
        "source_type":      "rag",
    }


def web_search_node(state: GraphState) -> GraphState:
    question        = state["question"]
    trusted_domains = state.get("trusted_domains", [])

    if not trusted_domains:
        print("[web_search_node] trusted_domains is empty → blocking web search (P1)")
        return {
            "documents":   [],
            "sources":     [],
            "source_type": "web_no_trusted_config",
        }

    tavily_search  = TavilySearch(max_results=WEB_MAX_RESULTS, search_depth="advanced")
    search_results = tavily_search.invoke(question)
    scraped_docs   = []
    source_labels  = []

    if not search_results or "results" not in search_results:
        return {"documents": [], "sources": [], "source_type": "web"}

    raw_results      = search_results["results"][:WEB_MAX_RESULTS]
    filtered_results = [r for r in raw_results if is_trusted(r.get("url", ""), trusted_domains)]

    if not filtered_results:
        print("[web_search_node] All results filtered out by trusted-source policy")
        return {
            "documents":   [],
            "sources":     [],
            "source_type": "web_no_trusted",
        }

    print(
        f"[web_search_node] Trusted-source filter: "
        f"{len(filtered_results)}/{len(raw_results)} results kept"
    )

    for result in filtered_results:
        url     = result.get("url")
        title   = result.get("title")
        snippet = result.get("content") or result.get("snippet") or ""

        if url:
            if title:
                source_labels.append(f"Web: {title} ({url})")
            else:
                source_labels.append(f"Web: {url}")

            try:
                loader = WebBaseLoader(url)
                docs   = loader.load()
                for doc in docs:
                    doc.metadata["source"] = url
                    if title:
                        doc.metadata["title"] = title
                    doc.page_content = doc.page_content[:2000]
                scraped_docs.extend(docs)
                continue
            except Exception as e:
                st.warning(f"Error scraping {url}: {e}")

        if snippet:
            scraped_docs.append(Document(
                page_content=snippet,
                metadata={"source": url or title or "web search result", "title": title or ""}
            ))

    sources = source_labels if source_labels else _build_source_summary(scraped_docs)
    return {"documents": scraped_docs, "sources": sources, "source_type": "web"}


def generate_node(state: GraphState) -> GraphState:
    question    = state["question"]
    documents   = state["documents"]
    history     = state.get("history", [])
    source_type = state.get("source_type", "rag")

    if source_type == "web_no_trusted_config":
        return {
            "answer": (
                "⚠️ **Web search blocked:** no trusted sources are configured.\n\n"
                "Please add one or more trusted domains in the sidebar "
                "(🌐 **Trusted Sources**) before using web search. "
                "If you have uploaded PDFs, try rephrasing your question so "
                "it can be answered from the knowledge base."
            ),
            "sources":     [],
            "source_type": "web_no_trusted_config",
        }

    if source_type == "web_no_trusted":
        return {
            "answer": (
                "⚠️ **No answer generated:** all web search results came from untrusted domains.\n\n"
                "Please add relevant trusted sources in the sidebar "
                "(🌐 **Trusted Sources**) or refine your query."
            ),
            "sources":     [],
            "source_type": "web_no_trusted",
        }

    context = "\n\n".join(
        f"[{_normalize_source_label(doc)}]\n{doc.page_content}"
        for doc in documents[:10]
    )
    history_text = _format_history(history)
    prompt = f"""You are a grounded Q&A assistant.
Use the recent conversation history and the provided context to answer the user's question.
If the answer is not supported by the context, say you cannot find enough information.

Recent conversation:
{history_text}

Context:
{context if context else 'No retrieved context.'}

Question:
{question}

Answer:"""
    preferred_model = state.get("preferred_model") or LLM_MODEL_ID
    try:
        response = invoke_with_fallback(prompt, preferred_model=preferred_model)
        return {
            "answer":      response.content,
            "sources":     state.get("sources", _build_source_summary(documents)),
            "source_type": source_type,
        }
    except Exception as exc:
        return {
            "answer": _friendly_model_error_message(exc),
            "sources":     state.get("sources", _build_source_summary(documents)),
            "source_type": source_type,
        }


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH — uses the local, token-budgeted summarize_node defined ABOVE.
#
# NOTE (fix applied): previously this function did:
#     from summary_node import summarize_node as _summarize_node
#     workflow.add_node("summarize", _summarize_node)
# That silently swapped in a DIFFERENT implementation from an external
# `summary_node.py` module which apparently had no char/page budgeting,
# causing prompts as large as ~25,794 tokens to be sent against an 8,000
# TPM model limit (413 "Request too large" errors).
#
# The fix: register THIS file's `summarize_node` (single-pass ≤5,500 chars,
# map-reduce batching at ≤3,500 chars/batch) directly, and drop the external
# import entirely so there's only one summarization code path.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def build_graph():
    workflow = StateGraph(GraphState)

    # Nodes — all defined locally in this file, single source of truth.
    workflow.add_node("retrieve",   retrieve_node)
    workflow.add_node("summarize",  summarize_node)   # ← local, token-budgeted version
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate",   generate_node)

    # START → conditional branch: summarize vs retrieve
    workflow.add_conditional_edges(
        START, route_from_start,
        {"summarize": "summarize", "retrieve": "retrieve"}
    )

    # summarize_node exits directly to END (generates its own answer)
    workflow.add_edge("summarize", END)

    # Normal RAG path
    workflow.add_conditional_edges(
        "retrieve", route_after_retrieval,
        {"generate": "generate", "web_search": "web_search"}
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()



# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        # ── Brand ─────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="sb-brand">
          <div class="sb-brand-icon">⬡</div>
          <div class="sb-brand-text">
            <div class="sb-brand-name">Neural<em>RAG</em></div>
            <div class="sb-brand-sub">Knowledge Intelligence</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Knowledge Mode ─────────────────────────────────────────────────────
        st.markdown('<div class="sb-section"><div class="sb-label"><span class="sb-label-icon">⚡</span>Query Mode</div>', unsafe_allow_html=True)

        mode_labels = {
            "hybrid":    "Hybrid — RAG + Web",
            "docs_only": "Documents Only",
            "web_only":  "Trusted Web Only",
        }
        mode_keys   = list(mode_labels.keys())
        mode_values = list(mode_labels.values())

        if "source_mode" not in st.session_state:
            st.session_state.source_mode = "hybrid"

        current_idx = mode_keys.index(st.session_state.source_mode)
        selected = st.radio(
            "source_mode_radio",
            options=mode_values,
            index=current_idx,
            label_visibility="collapsed",
        )
        st.session_state.source_mode = mode_keys[mode_values.index(selected)]

        mode_desc = {
            "hybrid":    "Search documents first, fall back to web if no match.",
            "docs_only": "Answer only from uploaded PDFs. No web search.",
            "web_only":  "Skip documents. Query trusted domains only.",
        }
        st.markdown(
            f'<div class="sb-info-chip"><span class="sb-info-chip-icon">ℹ</span>'
            f'<span>{mode_desc[st.session_state.source_mode]}</span></div></div>',
            unsafe_allow_html=True,
        )

        # ── Documents ─────────────────────────────────────────────────────────
        st.markdown('<div class="sb-section"><div class="sb-label"><span class="sb-label-icon">📄</span>Documents</div>', unsafe_allow_html=True)

        if "saved_pdf_names" not in st.session_state:
            st.session_state.saved_pdf_names = set()

        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
            label_visibility="collapsed",
        )
        if uploaded_files:
            new_files = [uf for uf in uploaded_files if uf.name not in st.session_state.saved_pdf_names]
            if new_files:
                saved = save_uploaded_pdfs(new_files)
                if saved:
                    st.session_state.saved_pdf_names.update(saved)
                    st.success(f"✓ {len(saved)} file(s) saved — click Index to embed")

        os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
        pdf_files = sorted(f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.lower().endswith(".pdf"))
        pdf_count = len(pdf_files)

        if pdf_count:
            st.markdown(
                f'<div class="sb-status ok" style="margin:8px 0 6px">'
                f'<span class="dot"></span>{pdf_count} PDF{"s" if pdf_count > 1 else ""} ready</div>',
                unsafe_allow_html=True,
            )
            for pdf in pdf_files:
                col_pdf, col_del = st.columns([5, 1])
                col_pdf.markdown(
                    f'<div class="sb-file-row"><span class="sb-file-icon">◈</span>'
                    f'<span class="sb-file-name">{pdf}</span></div>',
                    unsafe_allow_html=True,
                )
                if col_del.button("✕", key=f"del_pdf_{pdf}", help=f"Remove {pdf}"):
                    try:
                        os.remove(os.path.join(KNOWLEDGE_BASE_DIR, pdf))
                        st.session_state.saved_pdf_names.discard(pdf)
                        st.success(f"Removed — click Index to rebuild")
                        st.rerun()
                    except OSError as e:
                        st.error(f"Could not remove {pdf}: {e}")
        else:
            st.markdown(
                '<div class="sb-status warn" style="margin:8px 0 6px">'
                '<span class="dot"></span>No documents uploaded</div>',
                unsafe_allow_html=True,
            )

        if st.button("⟳  Index Documents", use_container_width=True):
            with st.spinner("Embedding documents…"):
                doc_count = ingest_pdfs_into_vectordb()
                st.cache_resource.clear()
                if doc_count > 0:
                    st.success(f"✓ {doc_count} pages indexed")
                else:
                    st.warning("No documents to index")

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Trusted Sources ────────────────────────────────────────────────────
        st.markdown('<div class="sb-section"><div class="sb-label"><span class="sb-label-icon">🌐</span>Trusted Sources</div>', unsafe_allow_html=True)

        trusted_domains = load_trusted_sources()

        if trusted_domains:
            st.markdown(
                f'<div class="sb-status ok" style="margin-bottom:6px">'
                f'<span class="dot"></span>{len(trusted_domains)} domain(s) trusted</div>',
                unsafe_allow_html=True,
            )
            for i, domain in enumerate(trusted_domains):
                col_d, col_r = st.columns([5, 1])
                col_d.markdown(
                    f'<div class="sb-file-row" style="margin:2px 0">'
                    f'<span class="sb-file-icon" style="color:var(--green)">✓</span>'
                    f'<span class="sb-file-name">{domain}</span></div>',
                    unsafe_allow_html=True,
                )
                if col_r.button("✕", key=f"rm_domain_{i}", help=f"Remove {domain}"):
                    new_list = [d for j, d in enumerate(trusted_domains) if j != i]
                    save_trusted_sources(new_list)
                    st.rerun()
        else:
            st.markdown(
                '<div class="sb-status err" style="margin-bottom:6px">'
                '<span class="dot"></span>Web search blocked</div>',
                unsafe_allow_html=True,
            )

        with st.form(key="add_domain_form", clear_on_submit=True):
            new_domain = st.text_input("Add domain", placeholder="e.g. arxiv.org", label_visibility="collapsed")
            submitted = st.form_submit_button("＋  Add domain", use_container_width=True)
            if submitted and new_domain.strip():
                valid, result = validate_domain(new_domain)
                if not valid:
                    st.error(result)
                elif result in trusted_domains:
                    st.info(f"{result} is already trusted")
                else:
                    trusted_domains.append(result)
                    save_trusted_sources(trusted_domains)
                    st.success(f"Trusted: {result}")
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # ── System Status ──────────────────────────────────────────────────────
        chroma_ok = os.path.exists(PERSIST_DIRECTORY)
        st.markdown(f"""
        <div class="sb-section sb-section-last">
          <div class="sb-label"><span class="sb-label-icon">⬡</span>System</div>
          <div class="sb-stat-row">
            <span class="sb-stat-label">Vector store</span>
            <span class="sb-stat-value {'ok' if chroma_ok else 'err'}">{'● Online' if chroma_ok else '● Offline'}</span>
          </div>
          <div class="sb-stat-row">
            <span class="sb-stat-label">LLM</span>
            <span class="sb-stat-value">gpt-oss-20b</span>
          </div>
          <div class="sb-stat-row">
            <span class="sb-stat-label">Embeddings</span>
            <span class="sb-stat-value">MiniLM-L6-v2</span>
          </div>
          <div class="sb-stat-row">
            <span class="sb-stat-label">Web search</span>
            <span class="sb-stat-value">Tavily</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("messages"):
            st.write("")
            if st.button("✕  Clear conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()


def render_header():
    st.markdown("""
    <div class="page-header">
      <div class="page-header-left">
        <div class="page-header-icon">⬡</div>
        <div>
          <div class="page-header-title">Neural<em>RAG</em></div>
          <div class="page-header-sub">Intelligent document research &amp; retrieval</div>
        </div>
      </div>
      <div class="page-header-right">
        <div class="live-pill"><span class="dot"></span>LIVE</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state():
    """Welcome screen shown before the first message."""
    st.markdown("""
    <div class="welcome-wrap">
      <div class="welcome-glyph">⬡</div>
      <div class="welcome-title">Your <em>Knowledge</em> Assistant</div>
      <div class="welcome-sub">
        Upload documents, define trusted sources, then ask anything.
        NeuralRAG finds the right answer from the right place.
      </div>

      <div class="welcome-steps">
        <div class="welcome-step">
          <div class="welcome-step-icon">📄</div>
          <span>Upload a PDF</span>
        </div>
        <div class="welcome-step">
          <div class="welcome-step-icon">⚡</div>
          <span>Index it</span>
        </div>
        <div class="welcome-step">
          <div class="welcome-step-icon">🌐</div>
          <span>Add trusted sources</span>
        </div>
        <div class="welcome-step">
          <div class="welcome-step-icon">💬</div>
          <span>Ask anything</span>
        </div>
      </div>

      <div class="welcome-prompts-label">Try asking</div>
      <div class="welcome-prompts" id="welcome-prompts">
        <div class="welcome-prompt-btn" onclick="document.querySelector('textarea').value='Summarize my document'; document.querySelector('textarea').dispatchEvent(new Event('input', {bubbles:true}))">
          <span class="welcome-prompt-icon">📋</span>Summarize my document
        </div>
        <div class="welcome-prompt-btn" onclick="document.querySelector('textarea').value='What are the main findings in this PDF?'; document.querySelector('textarea').dispatchEvent(new Event('input', {bubbles:true}))">
          <span class="welcome-prompt-icon">🔍</span>What are the main findings in this PDF?
        </div>
        <div class="welcome-prompt-btn" onclick="document.querySelector('textarea').value='What technologies are used in this project?'; document.querySelector('textarea').dispatchEvent(new Event('input', {bubbles:true}))">
          <span class="welcome-prompt-icon">🛠</span>What technologies are used in this project?
        </div>
        <div class="welcome-prompt-btn" onclick="document.querySelector('textarea').value='Search trusted sources for the latest research'; document.querySelector('textarea').dispatchEvent(new Event('input', {bubbles:true}))">
          <span class="welcome-prompt-icon">🌐</span>Search trusted sources for the latest research
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _badge_html(source_type: str) -> str:
    """Return the inline source-type badge HTML."""
    if source_type == "rag":
        return '<span class="type-badge rag">📄 PDF</span>'
    elif source_type == "web":
        return '<span class="type-badge web">🌐 Web</span>'
    elif source_type == "summary":
        return '<span class="type-badge summary">◉ Summary</span>'
    elif source_type in ("web_no_trusted", "web_no_trusted_config"):
        return '<span class="type-badge err">⚠ Blocked</span>'
    return ""


def render_message(
    role: str,
    content: str,
    source: str = None,
    source_type: str = None,
    domain_list: List[str] = None,
):
    if role == "user":
        st.markdown(
            f'<div class="msg-user"><div class="msg-user-bubble">{content}</div></div>',
            unsafe_allow_html=True,
        )
        return

    # ── Assistant message ──────────────────────────────────────────────────────
    badge = _badge_html(source_type)

    # Source pills
    source_pills_html = ""
    if domain_list:
        pill_cls = "rag" if source_type in ("rag", "summary") else "web"
        pills = "".join(f'<span class="src-pill {pill_cls}">{d}</span>' for d in domain_list)
        source_pills_html = (
            f'<div class="msg-sources">'
            f'<span class="msg-source-label">Source</span>{pills}</div>'
        )

    # For summaries, wrap content in the styled summary card
    if source_type == "summary":
        doc_label = domain_list[0] if domain_list else "Document"
        body_html = (
            f'<div class="summary-card">'
            f'<div class="summary-header">'
            f'<div class="summary-header-icon">◉</div>'
            f'<div>'
            f'<div class="summary-header-title">Document Summary</div>'
            f'<div class="summary-header-sub">{doc_label}</div>'
            f'</div></div>'
            f'<div class="summary-body">{content}</div>'
            f'</div>'
        )
    else:
        body_html = f'<div class="msg-ai-content">{content}</div>'

    st.markdown(f"""
    <div class="msg-ai">
      <div class="msg-ai-avatar">⬡</div>
      <div class="msg-ai-body">
        <div class="msg-ai-meta">
          <span class="msg-ai-name">NeuralRAG</span>
          {badge}
        </div>
        {body_html}
        {source_pills_html}
      </div>
    </div>
    """, unsafe_allow_html=True)

    if source:
        with st.expander("View sources", expanded=False):
            st.code(source, language=None)


def render_workflow_trace(placeholder, steps):
    """Clean inline trace panel — no emoji overload, clear status chips."""
    status_label = {
        "running": "running",
        "done":    "done",
        "pending": "pending",
        "blocked": "blocked",
        "skipped": "skipped",
    }
    step_icons = {
        "Intent Check": "◈",
        "Summarizer":   "◉",
        "Retrieval":    "◈",
        "Web Search":   "⟁",
        "Trust Filter": "🛡",
        "Generator":    "⬡",
    }

    rows_html = ""
    for s in steps:
        icon  = step_icons.get(s["name"], "·")
        cls   = status_label[s["status"]]
        rows_html += (
            f'<div class="trace-row">'
            f'<div class="trace-left">'
            f'<span class="trace-step-icon">{icon}</span>'
            f'<span class="trace-step-name">{s["name"]}</span>'
            f'<span class="trace-step-desc">— {s["desc"]}</span>'
            f'</div>'
            f'<span class="trace-status {cls}">{cls}</span>'
            f'</div>'
        )

    with placeholder.container():
        st.markdown(
            f'<div class="trace-wrap">'
            f'<div class="trace-title">Processing</div>'
            f'{rows_html}'
            f'</div>',
            unsafe_allow_html=True,
        )


def _extract_domain_labels(sources: List[str], source_type: str) -> List[str]:
    labels = []
    for s in sources:
        if source_type in ("rag", "summary"):
            if s.startswith("PDF:"):
                part     = s[4:].strip()
                filename = part.split("(")[0].strip()
                if filename and filename not in labels:
                    labels.append(filename)
        elif source_type in ("web", "web_no_trusted", "web_no_trusted_config"):
            if s.startswith("Web:"):
                part = s[4:].strip()
                url  = part[part.rfind("(") + 1 : part.rfind(")")] if "(" in part else part
                d    = domain_from_url(url)
                if d and d not in labels:
                    labels.append(d)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()
    render_header()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        render_empty_state()
    else:
        for msg in st.session_state.messages:
            render_message(
                role=msg["role"],
                content=msg["content"],
                source=msg.get("source"),
                source_type=msg.get("source_type"),
                domain_list=msg.get("domain_list"),
            )

    st.write("")

    if question := st.chat_input("Ask about your documents or search the web…"):
        st.session_state.messages.append({"role": "user", "content": question})

        trace_placeholder = st.empty()
        source_mode       = st.session_state.get("source_mode", "hybrid")
        summary_intent    = is_summary_intent(question)

        if summary_intent:
            steps = [
                {"name": "Intent Check", "desc": "Summary intent detected",         "status": "done"},
                {"name": "Summarizer",   "desc": "Loading full documents from disk", "status": "running"},
            ]
        elif source_mode == "docs_only":
            steps = [
                {"name": "Intent Check", "desc": "Normal Q&A query",              "status": "done"},
                {"name": "Retrieval",    "desc": "Searching uploaded documents",  "status": "running"},
                {"name": "Generator",    "desc": "Composing grounded answer",     "status": "pending"},
            ]
        else:
            steps = [
                {"name": "Intent Check", "desc": "Normal Q&A query",              "status": "done"},
                {"name": "Retrieval",    "desc": "Searching uploaded documents",  "status": "running"},
                {"name": "Web Search",   "desc": "Fetching current web sources",  "status": "pending"},
                {"name": "Trust Filter", "desc": "Applying domain policy",        "status": "pending"},
                {"name": "Generator",    "desc": "Composing grounded answer",     "status": "pending"},
            ]

        render_workflow_trace(trace_placeholder, steps)
        time.sleep(0.7)

        conversation_history = st.session_state.messages[-(HISTORY_WINDOW + 1):-1]
        trusted_domains      = load_trusted_sources()

        try:
            app    = build_graph()
            result = app.invoke({
                "question":        question,
                "history":         conversation_history,
                "trusted_domains": trusted_domains,
                "source_mode":     source_mode,
                "is_summary":      summary_intent,
                "preferred_model": st.session_state.get("selected_model") or st.session_state.get("preferred_model") or DEFAULT_MODEL,
            })

            source_type = result.get("source_type", "rag")

            if summary_intent:
                steps[1]["status"] = "done"
            else:
                steps[1]["status"] = "done"
                if source_mode != "docs_only":
                    web_ran = source_type in ("web", "web_no_trusted", "web_no_trusted_config")
                    steps[2]["status"] = "done" if web_ran else "skipped"
                    if web_ran:
                        steps[3]["status"] = "blocked" if source_type == "web_no_trusted_config" else "done"
                    generator_idx = 4
                else:
                    generator_idx = 2
                steps[generator_idx]["status"] = "running"
                render_workflow_trace(trace_placeholder, steps)
                time.sleep(0.4)
                steps[generator_idx]["status"] = "done"

            render_workflow_trace(trace_placeholder, steps)
            time.sleep(0.4)

            if result and result.get("answer"):
                answer      = result["answer"]
                sources     = result.get("sources") or _build_source_summary(result.get("documents", []))
                source_text = ("Sources:\n" + "\n".join(f"  {s}" for s in sources)) if sources else None
                domain_list = _extract_domain_labels(sources, source_type)

                st.session_state.messages.append({
                    "role":        "assistant",
                    "content":     answer,
                    "source":      source_text,
                    "source_type": source_type,
                    "domain_list": domain_list,
                })
            else:
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": "Could not generate an answer. Please try again.",
                })

        except Exception as e:
            st.session_state.messages.append({
                "role":    "assistant",
                "content": f"Execution error: {str(e)}",
            })

        trace_placeholder.empty()
        st.rerun()


if __name__ == "__main__":
    main()