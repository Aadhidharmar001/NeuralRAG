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

from utils.config import DEFAULT_MODEL, SUPPORTED_MODELS
from utils.rag_components import invoke_with_fallback


from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION — must be FIRST Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
favicon = Image.open(os.path.join(os.path.dirname(__file__), "favicon.png"))
st.set_page_config(
    page_title="NeuralRAG · Intelligent Research Agent",
    page_icon=favicon,
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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg-base: #04060d;
  --bg-surface: rgba(10, 15, 30, 0.6);
  --bg-sidebar: rgba(6, 8, 18, 0.85);
  --bg-card: rgba(15, 22, 42, 0.45);
  --bg-hover: rgba(56, 189, 248, 0.08);
  --accent-cyan: #00F2FE;
  --accent-violet: #8A4FFF;
  --accent-gradient: linear-gradient(135deg, #00F2FE 0%, #8A4FFF 100%);
  --border-glass: rgba(255, 255, 255, 0.05);
  --border-glow: rgba(0, 242, 254, 0.25);
  --text-high: #F8FAFC;
  --text-mid: #94A3B8;
  --text-low: #64748B;
  --green: #05F2C7;
  --green-dim: rgba(5, 242, 199, 0.08);
  --green-border: rgba(5, 242, 199, 0.25);
  --amber: #fbbf24;
  --amber-dim: rgba(251, 191, 36, 0.08);
  --amber-border: rgba(251, 191, 36, 0.25);
  --red: #FF2E93;
  --red-dim: rgba(255, 46, 147, 0.08);
  --red-border: rgba(255, 46, 147, 0.25);
  --purple: #A78BFA;
  --purple-dim: rgba(167, 139, 250, 0.08);
  --purple-border: rgba(167, 139, 250, 0.25);
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 20px;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
  background: var(--bg-base) !important;
  color: var(--text-high) !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 15px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

body::before {
  content: '';
  position: fixed; inset: 0;
  background-image: 
    radial-gradient(circle at 10% 0%, rgba(138, 79, 255, 0.05) 0%, transparent 35%),
    radial-gradient(circle at 90% 100%, rgba(0, 242, 254, 0.05) 0%, transparent 35%),
    linear-gradient(rgba(255,255,255,0.003) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.003) 1px, transparent 1px);
  background-size: 100% 100%, 100% 100%, 56px 56px, 56px 56px;
  pointer-events: none;
  z-index: 0;
}

/* Streamlit Header Override */
[data-testid='stHeader'] { background: transparent !important; height: 0 !important; }
#MainMenu, footer, [data-testid='stDecoration'] { display: none !important; }
.block-container { padding: 3rem 2.5rem 6rem !important; max-width: 960px !important; margin: 0 auto !important; }

/* Custom Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

/* Sidebar Layout */
section[data-testid="stSidebar"] {
  background: var(--bg-sidebar) !important;
  backdrop-filter: blur(25px) !important;
  border-right: 1px solid var(--border-glass) !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

.sb-brand {
  display: flex; align-items: center; gap: 14px;
  padding: 2rem 1.5rem 1.5rem;
  border-bottom: 1px solid var(--border-glass);
}
.brand-logo-svg {
  filter: drop-shadow(0 0 8px rgba(0, 242, 254, 0.3));
}
.sb-brand-name {
  font-family: 'Outfit', sans-serif;
  font-weight: 700;
  font-size: 1.25rem;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #00F2FE, #8A4FFF);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.sb-brand-sub {
  font-size: 0.72rem;
  color: var(--text-low);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.sb-section { padding: 1.5rem; border-bottom: 1px solid var(--border-glass); }
.sb-section-last { border-bottom: none; }
.sb-label {
  display: flex; align-items: center; gap: 8px;
  font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--text-low); margin-bottom: 1rem;
}
.sb-label-icon { font-size: 13px; color: var(--accent-cyan); }

/* Radio Button Redesign */
div[data-testid="stRadio"] > div { flex-direction: column; gap: 8px; }
div[data-testid="stRadio"] label {
  background: rgba(15, 22, 42, 0.35) !important;
  border: 1px solid var(--border-glass) !important;
  border-radius: var(--radius-sm) !important;
  padding: 10px 14px !important;
  color: var(--text-mid) !important;
  font-family: 'Outfit', sans-serif !important;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
  cursor: pointer !important;
  display: flex !important;
}
div[data-testid="stRadio"] label:hover {
  border-color: rgba(0, 242, 254, 0.25) !important;
  background: rgba(0, 242, 254, 0.03) !important;
  color: var(--text-high) !important;
  transform: translateX(2px);
}
div[data-testid="stRadio"] label:has(input:checked) {
  background: linear-gradient(135deg, rgba(0, 242, 254, 0.1), rgba(138, 79, 255, 0.06)) !important;
  border-color: var(--accent-cyan) !important;
  color: var(--text-high) !important;
  box-shadow: 0 0 15px rgba(0, 242, 254, 0.15) !important;
}
div[data-testid="stRadio"] input[type="radio"] { display: none !important; }
div[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] { margin-left: 0 !important; }

/* File Uploader styling */
section[data-testid="stSidebar"] .stFileUploader section {
  background: rgba(10, 15, 30, 0.4) !important;
  border: 1px dashed rgba(0, 242, 254, 0.15) !important;
  border-radius: var(--radius-md) !important;
  padding: 0.8rem !important;
}
section[data-testid="stSidebar"] .stFileUploader section:hover {
  border-color: var(--accent-cyan) !important;
  background: rgba(0, 242, 254, 0.02) !important;
}

/* File Row & Domain Card lists */
.sb-file-row {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 12px;
  background: rgba(15, 22, 42, 0.35);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-sm);
  margin-bottom: 6px;
  transition: all 0.25s ease;
}
.sb-file-row:hover {
  border-color: rgba(0, 242, 254, 0.25);
  background: rgba(0, 242, 254, 0.03);
  transform: translateX(2px);
}
.sb-file-icon { color: var(--accent-cyan); font-size: 12px; }
.sb-file-name { font-size: 0.78rem; color: var(--text-mid); font-family: 'JetBrains Mono', monospace; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* Status pills */
.sb-status {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: 100px;
  font-size: 0.7rem; font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.03em; margin-bottom: 8px;
}
.sb-status.ok { background: var(--green-dim); color: var(--green); border: 1px solid var(--green-border); }
.sb-status.warn { background: var(--amber-dim); color: var(--amber); border: 1px solid var(--amber-border); }
.sb-status.err { background: var(--red-dim); color: var(--red); border: 1px solid var(--red-border); }
.sb-status .dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; animation: pulse 2s infinite; }

/* Input box & selectbox override */
div[data-baseweb="select"] > div,
input[data-testid="stTextInput"] {
  background: rgba(15, 22, 42, 0.5) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-high) !important;
}
div[data-baseweb="select"] > div:hover,
input[data-testid="stTextInput"]:hover {
  border-color: rgba(0, 242, 254, 0.25) !important;
}

/* System telemetry card */
.sb-stat-row { display: flex; justify-content: space-between; padding: 6px 0; font-size: 0.78rem; }
.sb-stat-label { color: var(--text-low); }
.sb-stat-value { color: var(--text-mid); font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; }
.sb-stat-value.ok { color: var(--green); text-shadow: 0 0 8px rgba(5, 242, 199, 0.3); }
.sb-stat-value.err { color: var(--red); }

/* Buttons styling */
.stButton > button {
  background: rgba(15, 22, 42, 0.6) !important;
  color: var(--text-high) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Outfit', sans-serif !important;
  padding: 0.5rem 1.2rem !important;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0, 242, 254, 0.12), rgba(138, 79, 255, 0.08)) !important;
  border-color: var(--accent-cyan) !important;
  box-shadow: 0 0 15px rgba(0, 242, 254, 0.2) !important;
  transform: translateY(-1px) !important;
}

/* Header locks */
.page-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 1.5rem 0; margin-bottom: 2rem;
  border-bottom: 1px solid var(--border-glass);
}
.page-header-left { display: flex; align-items: center; gap: 14px; }
.page-header-title {
  font-size: 1.5rem; font-weight: 700;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #00F2FE, #8A4FFF);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.page-header-sub { font-size: 0.8rem; color: var(--text-low); margin-top: 2px; }
.live-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; background: rgba(5, 242, 199, 0.08);
  border: 1px solid rgba(5, 242, 199, 0.2); border-radius: 100px;
  font-size: 0.68rem; font-weight: 600; color: var(--green);
  font-family: 'JetBrains Mono', monospace; letter-spacing: 0.08em;
  box-shadow: 0 0 10px rgba(5, 242, 199, 0.15);
}
.live-pill .dot { width: 5px; height: 5px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }

/* Empty state / Welcome screen */
.welcome-wrap { display: flex; flex-direction: column; align-items: center; padding: 2.5rem 0; text-align: center; }
.welcome-logo-container { margin-bottom: 1.5rem; filter: drop-shadow(0 0 15px rgba(0, 242, 254, 0.4)); }
.welcome-title { font-size: 2.25rem; font-weight: 700; letter-spacing: -0.03em; margin-bottom: 0.75rem; }
.welcome-title span { background: linear-gradient(135deg, #00F2FE, #8A4FFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.welcome-sub { font-size: 0.95rem; color: var(--text-mid); max-width: 500px; margin-bottom: 2.5rem; line-height: 1.6; }

.welcome-grid {
  display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;
  width: 100%; max-width: 680px; margin-bottom: 2.5rem;
}
.welcome-card {
  background: rgba(15, 22, 42, 0.35);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-md);
  padding: 1.25rem; text-align: left; cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex; flex-direction: column; gap: 6px;
  position: relative; overflow: hidden;
}
.welcome-card::before {
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(135deg, rgba(0, 242, 254, 0.03), rgba(138, 79, 255, 0.03));
  opacity: 0; transition: opacity 0.3s ease;
}
.welcome-card:hover {
  transform: translateY(-3px);
  border-color: rgba(0, 242, 254, 0.3);
  box-shadow: 0 10px 30px -10px rgba(0, 242, 254, 0.15), 0 0 15px rgba(138, 79, 255, 0.05);
}
.welcome-card:hover::before { opacity: 1; }
.welcome-card-icon { font-size: 20px; margin-bottom: 4px; }
.welcome-card-title { font-size: 0.925rem; font-weight: 600; color: var(--text-high); }
.welcome-card-desc { font-size: 0.78rem; color: var(--text-low); line-height: 1.45; }

/* Stepper Stepper Workflow */
.stepper { display: flex; flex-direction: column; gap: 12px; width: 100%; }
.step { display: flex; align-items: center; gap: 16px; position: relative; padding: 4px 0; }
.step::after {
  content: ''; position: absolute; left: 16px; top: 34px; bottom: -20px;
  width: 2px; background: rgba(255, 255, 255, 0.05); z-index: 1;
}
.step:last-child::after { display: none; }
.step-icon-outer {
  width: 34px; height: 34px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  background: rgba(15, 22, 42, 0.6);
  border: 2px solid rgba(255, 255, 255, 0.05);
  position: relative; z-index: 2;
  transition: all 0.3s ease;
}
.step-icon-inner { width: 10px; height: 10px; border-radius: 50%; background: var(--text-low); }

/* Stepper Status variations */
.step.done .step-icon-outer { border-color: var(--green); background: var(--green-dim); box-shadow: 0 0 10px rgba(5, 242, 199, 0.15); }
.step.done .step-icon-inner { background: var(--green); }
.step.done::after { background: var(--green); opacity: 0.4; }

.step.running .step-icon-outer {
  border-color: var(--accent-cyan); background: rgba(0, 242, 254, 0.05);
  box-shadow: 0 0 12px rgba(0, 242, 254, 0.2);
  animation: pulse-border 1.5s infinite alternate;
}
.step.running .step-icon-inner { background: var(--accent-cyan); animation: pulse 1.5s infinite; }

.step.blocked .step-icon-outer { border-color: var(--red); background: var(--red-dim); }
.step.blocked .step-icon-inner { background: var(--red); }

.step.skipped { opacity: 0.5; }

.step-content { display: flex; flex-direction: column; }
.step-name { font-size: 0.85rem; font-weight: 600; color: var(--text-high); }
.step-desc { font-size: 0.75rem; color: var(--text-low); }

/* Chat bubbles styling */
.msg-wrap { display: flex; flex-direction: column; gap: 0; }
.msg-user { display: flex; justify-content: flex-end; margin: 1rem 0; animation: msgIn 0.3s cubic-bezier(0.4, 0, 0.2, 1) both; }
.msg-user-bubble {
  max-width: 75%;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(79, 70, 229, 0.15));
  border: 1px solid rgba(99, 102, 241, 0.3);
  border-radius: 18px 18px 4px 18px; padding: 1rem 1.25rem;
  font-size: 0.925rem; color: var(--text-high); line-height: 1.65;
  box-shadow: 0 4px 15px rgba(99, 102, 241, 0.05);
}

.msg-ai { display: flex; align-items: flex-start; gap: 16px; margin: 1rem 0; animation: msgIn 0.3s cubic-bezier(0.4, 0, 0.2, 1) both; }
.msg-ai-avatar {
  width: 36px; height: 36px; border-radius: 50%;
  background: linear-gradient(135deg, #00F2FE, #8A4FFF);
  border: 1px solid rgba(255, 255, 255, 0.1);
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; color: white; flex-shrink: 0;
  box-shadow: 0 0 10px rgba(0, 242, 254, 0.2);
}
.msg-ai-body {
  flex: 1; min-width: 0;
  background: rgba(15, 22, 42, 0.35);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-md); padding: 1.25rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
}
.msg-ai-meta { display: flex; align-items: center; gap: 8px; margin-bottom: 0.75rem; }
.msg-ai-name { font-size: 0.8rem; font-weight: 600; color: var(--text-high); }

.msg-ai-content { font-size: 0.925rem; color: var(--text-mid); line-height: 1.75; }
.msg-ai-content p { margin: 0 0 0.8em; }
.msg-ai-content p:last-child { margin-bottom: 0; }
.msg-ai-content strong { color: var(--text-high); font-weight: 600; }
.msg-ai-content em { color: var(--text-mid); }
.msg-ai-content code {
  background: rgba(15, 22, 42, 0.6);
  border: 1px solid var(--border-glass);
  border-radius: 4px; padding: 2px 6px;
  font-family: 'JetBrains Mono', monospace; font-size: 0.82em; color: var(--accent-cyan);
}
.msg-ai-content h1, .msg-ai-content h2, .msg-ai-content h3 { color: var(--text-high); font-weight: 600; margin: 1.2em 0 0.5em; letter-spacing: -0.015em; }
.msg-ai-content h1 { font-size: 1.25rem; }
.msg-ai-content h2 { font-size: 1.15rem; }
.msg-ai-content h3 { font-size: 1rem; }

/* Source footer elements */
.msg-sources { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid var(--border-glass); }
.msg-source-label { font-size: 0.7rem; color: var(--text-low); font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; margin-right: 4px; }
.src-pill {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 10px; border-radius: 100px;
  font-size: 0.7rem; font-weight: 500; font-family: 'JetBrains Mono', monospace;
  border: 1px solid; transition: all 0.2s ease;
}
.src-pill.rag { background: rgba(0, 242, 254, 0.06); color: var(--accent-cyan); border-color: rgba(0, 242, 254, 0.15); }
.src-pill.web { background: rgba(52, 211, 153, 0.06); color: var(--green); border-color: rgba(52, 211, 153, 0.15); }
.src-pill:hover { transform: translateY(-1px); border-color: currentColor; }

/* Chat inputs overrides */
.stChatInput > div {
  background: rgba(15, 22, 42, 0.75) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: var(--radius-lg) !important;
  backdrop-filter: blur(20px) !important;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4) !important;
  transition: all 0.3s ease !important;
}
.stChatInput > div:focus-within {
  border-color: var(--accent-cyan) !important;
  box-shadow: 0 0 20px rgba(0, 242, 254, 0.25), 0 10px 40px rgba(0, 0, 0, 0.4) !important;
}

/* Summary Card styler */
.summary-card { background: rgba(15, 22, 42, 0.4); border: 1px solid var(--border-glass); border-radius: var(--radius-md); overflow: hidden; }
.summary-header {
  background: linear-gradient(135deg, rgba(138, 79, 255, 0.12), rgba(0, 242, 254, 0.08));
  border-bottom: 1px solid var(--border-glass); padding: 1.25rem;
  display: flex; align-items: center; gap: 12px;
}
.summary-header-icon {
  width: 32px; height: 32px; border-radius: 6px;
  background: rgba(138, 79, 255, 0.1); border: 1px solid rgba(138, 79, 255, 0.2);
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; color: var(--purple); flex-shrink: 0;
}
.summary-header-title { font-size: 0.95rem; font-weight: 600; color: var(--text-high); }
.summary-header-sub { font-size: 0.72rem; color: var(--text-low); font-family: 'JetBrains Mono', monospace; margin-top: 1px; }
.summary-body { padding: 1.5rem; font-size: 0.925rem; line-height: 1.75; color: var(--text-mid); }
.summary-body h2 { font-size: 1.05rem; color: var(--purple); margin-top: 1.25rem; }
.summary-body ul { padding-left: 1.4em; }

/* Dynamic Animations */
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
@keyframes pulse-border { 0% { border-color: rgba(0, 242, 254, 0.1); } 100% { border-color: rgba(0, 242, 254, 0.6); } }
@keyframes msgIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes rotate { 100% { transform: rotate(360deg); } }
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


_SUMMARY_GUIDELINES = (
    "1. **High-Level Overview**: Start with a concise overview of the document's main purpose, scope, and target audience.\n"
    "2. **Key Themes & Sections**: Dynamically identify and present the main topics/sections of the document using descriptive markdown headings (e.g., `## Key Findings`, `## System Architecture`, `## Methodology`, etc.) that are specifically relevant to the actual document contents. Do NOT use a generic or rigid template if it does not fit the document.\n"
    "3. **Thorough Details**: Within those sections, use bullet points to outline key details, naming specific tools, frameworks, metrics, and methods mentioned.\n"
    "4. **Key Takeaways**: End with a brief summary of the main takeaways, limitations, or future directions."
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
        f"-> using {strategy_label} strategy (threshold={SUMMARY_SINGLEPASS_PAGE_LIMIT})"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SINGLE-PASS PATH  (small documents)
    # ══════════════════════════════════════════════════════════════════════════
    if use_single_pass:
        context = _build_singlepass_context(pdf_map)

        prompt = (
            f"You are an expert document analyst. Document(s): {doc_list}\n\n"
            "Produce a **comprehensive dynamic summary** tailored to the document structure and content, following these guidelines:\n\n"
            f"{_SUMMARY_GUIDELINES}\n\n"
            "Guidelines:\n"
            "- Be thorough and specific — name actual technologies, metrics, methods.\n"
            "- Cover the entire document, not just the opening pages.\n"
            "- Structure the output naturally with markdown headers, lists, and bold text based on the guidelines above.\n\n"
            f"Recent conversation:\n{history_text}\n\n"
            f"--- DOCUMENT CONTENT ---\n{context}\n--- END ---\n\n"
            "Comprehensive document summary:"
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
        "Synthesise these into a single **comprehensive dynamic summary** tailored to the document structure and content, following these guidelines:\n\n"
        f"{_SUMMARY_GUIDELINES}\n\n"
        "Guidelines:\n"
        "- Cover the ENTIRE document, not just one part.\n"
        "- Be thorough and specific — name actual technologies, metrics, methods.\n"
        "- Structure the output naturally with markdown headers, lists, and bold text based on the guidelines above.\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"--- EXTRACTED KEY POINTS ---\n{combined}\n--- END ---\n\n"
        "Comprehensive document summary:"
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
        print("[retrieve_node] source_mode=web_only -> skipping RAG, forcing web fallback")
        return {"documents": [], "sources": [], "needs_web_search": True}

    question    = state["question"]
    vectorstore = get_vectorstore()
    retriever   = create_retriever()

    if vectorstore is None:
        print("[retrieve_node] vectorstore is None -> web fallback")
        return {"documents": [], "sources": [], "needs_web_search": True}

    relevant_docs: List[Document] = []

    try:
        docs_and_scores = vectorstore.similarity_search_with_relevance_scores(
            question, k=RETRIEVAL_K
        )

        print(f"[retrieve_node] Raw (doc, score) pairs returned ({len(docs_and_scores)} total):")
        for i, (doc, score) in enumerate(docs_and_scores):
            snippet = doc.page_content[:80].replace("\n", " ")
            print(f"  [{i}] score={score:.4f} | {snippet}...")

        if not docs_and_scores:
            print("[retrieve_node] similarity_search returned 0 results -> web fallback")
            return {"documents": [], "sources": [], "needs_web_search": True}

        all_scores = [score for _, score in docs_and_scores]
        max_score  = max(all_scores)

        if max_score > 1.0:
            DISTANCE_CEILING = 1.5
            relevant_docs = [doc for doc, score in docs_and_scores if score <= DISTANCE_CEILING]
            print(
                f"[retrieve_node] Scores appear to be raw distances (max={max_score:.4f}). "
                f"Keeping docs with distance <= {DISTANCE_CEILING}. "
                f"Surviving: {len(relevant_docs)}/{len(docs_and_scores)}"
            )
        else:
            relevant_docs = [doc for doc, score in docs_and_scores if score >= RETRIEVAL_THRESHOLD]
            print(
                f"[retrieve_node] Scores are similarities (max={max_score:.4f}). "
                f"Keeping docs with score >= {RETRIEVAL_THRESHOLD}. "
                f"Surviving: {len(relevant_docs)}/{len(docs_and_scores)}"
            )

    except Exception as exc:
        print(f"[retrieve_node] similarity_search_with_relevance_scores raised: {exc}")
        print("[retrieve_node] Falling back to retriever.invoke() (no score filter)")
        relevant_docs = retriever.invoke(question) if retriever is not None else []
        print(f"[retrieve_node] retriever.invoke() returned {len(relevant_docs)} docs")

    if not relevant_docs:
        print("[retrieve_node] No relevant docs after filtering -> web fallback triggered")
        return {"documents": [], "sources": [], "needs_web_search": True}

    print(f"[retrieve_node] {len(relevant_docs)} relevant docs found -> using RAG, no web fallback")
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
        print("[web_search_node] trusted_domains is empty -> blocking web search (P1)")
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
          <svg class="brand-logo-svg" viewBox="0 0 100 100" width="36" height="36" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#00F2FE" />
                <stop offset="100%" stop-color="#8A4FFF" />
              </linearGradient>
              <filter id="glow">
                <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>
            <polygon points="50,5 90,28 90,72 50,95 10,72 10,28" fill="none" stroke="url(#logoGrad)" stroke-width="6" filter="url(#glow)">
              <animateTransform attributeName="transform" type="rotate" from="0 50 50" to="360 50 50" dur="12s" repeatCount="indefinite"/>
            </polygon>
            <polygon points="50,20 76,35 76,65 50,80 24,65 24,35" fill="none" stroke="url(#logoGrad)" stroke-width="3" opacity="0.7">
              <animateTransform attributeName="transform" type="rotate" from="360 50 50" to="0 50 50" dur="8s" repeatCount="indefinite"/>
            </polygon>
            <circle cx="50" cy="50" r="10" fill="url(#logoGrad)"/>
          </svg>
          <div class="sb-brand-text">
            <div class="sb-brand-name">NeuralRAG</div>
            <div class="sb-brand-sub">Knowledge Intelligence</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Model Selection ───────────────────────────────────────────────────
        st.markdown('<div class="sb-section"><div class="sb-label"><span class="sb-label-icon">⚙️</span>Active Model</div>', unsafe_allow_html=True)
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = DEFAULT_MODEL
        
        selected_model = st.selectbox(
            "model_selector",
            options=SUPPORTED_MODELS,
            index=SUPPORTED_MODELS.index(st.session_state.selected_model) if st.session_state.selected_model in SUPPORTED_MODELS else 0,
            label_visibility="collapsed",
            key="selected_model_dropdown"
        )
        st.session_state.selected_model = selected_model
        st.markdown('</div>', unsafe_allow_html=True)

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
            f'<div style="font-size:0.75rem; color:var(--text-low); margin-top:8px; line-height:1.4;">'
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
            <span class="sb-stat-label">Active LLM</span>
            <span class="sb-stat-value">{st.session_state.selected_model}</span>
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
        <svg viewBox="0 0 100 100" width="30" height="30" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 0 8px rgba(0, 242, 254, 0.4));">
          <defs>
            <linearGradient id="headerLogoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#00F2FE" />
              <stop offset="100%" stop-color="#8A4FFF" />
            </linearGradient>
          </defs>
          <polygon points="50,5 90,28 90,72 50,95 10,72 10,28" fill="none" stroke="url(#headerLogoGrad)" stroke-width="8"/>
          <circle cx="50" cy="50" r="14" fill="url(#headerLogoGrad)"/>
        </svg>
        <div>
          <div class="page-header-title">NeuralRAG</div>
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
      <div class="welcome-logo-container">
        <svg viewBox="0 0 100 100" width="80" height="80" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="welcomeLogoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#00F2FE" />
              <stop offset="100%" stop-color="#8A4FFF" />
            </linearGradient>
            <filter id="welcomeGlow">
              <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="blur"/>
              <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
          <polygon points="50,5 90,28 90,72 50,95 10,72 10,28" fill="none" stroke="url(#welcomeLogoGrad)" stroke-width="6" filter="url(#welcomeGlow)">
            <animateTransform attributeName="transform" type="rotate" from="0 50 50" to="360 50 50" dur="15s" repeatCount="indefinite"/>
          </polygon>
          <polygon points="50,20 76,35 76,65 50,80 24,65 24,35" fill="none" stroke="url(#welcomeLogoGrad)" stroke-width="3" opacity="0.6">
            <animateTransform attributeName="transform" type="rotate" from="360 50 50" to="0 50 50" dur="10s" repeatCount="indefinite"/>
          </polygon>
          <circle cx="50" cy="50" r="12" fill="url(#welcomeLogoGrad)"/>
        </svg>
      </div>
      <div class="welcome-title">Your <span>Knowledge</span> Intelligence</div>
      <div class="welcome-sub">
        Upload documents, define trusted sources, and start exploring. NeuralRAG routes your queries dynamically across vector storage and web services to deliver fully grounded insights.
      </div>

      <div class="welcome-grid">
        <div class="welcome-card" onclick="document.querySelector('textarea').value='Summarize the document'; document.querySelector('textarea').dispatchEvent(new Event('input', {bubbles:true}))">
          <div class="welcome-card-icon">📋</div>
          <div class="welcome-card-title">Summarize Document</div>
          <div class="welcome-card-desc">Generate a dynamic, comprehensive summary tailored specifically to your document.</div>
        </div>
        <div class="welcome-card" onclick="document.querySelector('textarea').value='What is the methodology and results described in the document?'; document.querySelector('textarea').dispatchEvent(new Event('input', {bubbles:true}))">
          <div class="welcome-card-icon">🔬</div>
          <div class="welcome-card-title">Analyze Methodology</div>
          <div class="welcome-card-desc">Drill down into structural techniques, methods, parameters, and findings.</div>
        </div>
        <div class="welcome-card" onclick="document.querySelector('textarea').value='What technologies or tools are used in this project?'; document.querySelector('textarea').dispatchEvent(new Event('input', {bubbles:true}))">
          <div class="welcome-card-icon">🛠️</div>
          <div class="welcome-card-title">Extract Tech Stack</div>
          <div class="welcome-card-desc">List all key software, hardware libraries, database clusters, and frameworks.</div>
        </div>
        <div class="welcome-card" onclick="document.querySelector('textarea').value='Search trusted sources for the latest research related to this document'; document.querySelector('textarea').dispatchEvent(new Event('input', {bubbles:true}))">
          <div class="welcome-card-icon">🌐</div>
          <div class="welcome-card-title">Cross-Reference Web</div>
          <div class="welcome-card-desc">Augment the document facts with real-time searches filtered through your trusted domains.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _badge_html(source_type: str) -> str:
    """Return the inline source-type badge HTML."""
    if source_type == "rag":
        return '<span class="type-badge rag">📄 PDF Knowledge</span>'
    elif source_type == "web":
        return '<span class="type-badge web">🌐 Trusted Web</span>'
    elif source_type == "summary":
        return '<span class="type-badge summary">◉ Document Summary</span>'
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
      <div class="msg-ai-avatar">
        <svg viewBox="0 0 100 100" width="20" height="20" xmlns="http://www.w3.org/2000/svg">
          <polygon points="50,5 90,28 90,72 50,95 10,72 10,28" fill="none" stroke="white" stroke-width="12"/>
          <circle cx="50" cy="50" r="16" fill="white"/>
        </svg>
      </div>
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
        with st.expander("View source context", expanded=False):
            st.code(source, language=None)


def _get_step_status_svg(status: str) -> str:
    if status == "done":
        return """
        <div class="step-icon-outer" style="border-color:var(--green); background:var(--green-dim);">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--green)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
        </div>
        """
    elif status == "running":
        return """
        <div class="step-icon-outer" style="border-color:var(--accent-cyan); background:rgba(0, 242, 254, 0.05); animation: pulse-border 1.5s infinite alternate;">
          <div class="step-icon-inner" style="background:var(--accent-cyan); width: 8px; height: 8px; border-radius: 50%; animation: pulse 1.5s infinite;"></div>
        </div>
        """
    elif status == "blocked":
        return """
        <div class="step-icon-outer" style="border-color:var(--red); background:var(--red-dim);">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
        </div>
        """
    elif status == "skipped":
        return """
        <div class="step-icon-outer" style="border-color:var(--text-low); opacity:0.4;">
          <div style="width:6px; height:2px; background:var(--text-low);"></div>
        </div>
        """
    else: # pending
        return """
        <div class="step-icon-outer" style="border: 2px dashed var(--text-low); background:transparent;">
          <div class="step-icon-inner" style="background:transparent; width:6px; height:6px;"></div>
        </div>
        """


def render_workflow_trace(placeholder, steps):
    rows_html = ""
    for s in steps:
        status_svg = _get_step_status_svg(s["status"])
        rows_html += (
            f'<div class="step {s["status"]}">'
            f'{status_svg}'
            f'<div class="step-content">'
            f'<span class="step-name">{s["name"]}</span>'
            f'<span class="step-desc">{s["desc"]}</span>'
            f'</div>'
            f'</div>'
        )

    with placeholder.container():
        st.markdown(
            f'<div class="trace-wrap">'
            f'<div class="trace-title">Workflow Execution</div>'
            f'<div class="stepper">'
            f'{rows_html}'
            f'</div>'
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
                steps[generator_idx]["status"] = "done"

            render_workflow_trace(trace_placeholder, steps)

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
