import streamlit as st
import os
import re
import json
import time
import random
import markdown as md_lib
from dotenv import load_dotenv
from typing import List, Optional
from typing_extensions import TypedDict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be FIRST
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralRAG · Your Document Companion",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE          = 1000
CHUNK_OVERLAP       = 200
LLM_MODEL_ID        = "openai/gpt-oss-20b"
RETRIEVAL_K         = 3
RETRIEVAL_THRESHOLD = 0.1
HISTORY_WINDOW      = 5
WEB_MAX_RESULTS     = 3

_DOMAIN_RE = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$"
)

# ── Summary intent detection (unchanged) ─────────────────────────────────────
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

# ── Greeting intent detection (NEW) ──────────────────────────────────────────
_GREETING_RE = re.compile(
    r"^\s*("
    r"hi+|hey+|hello+|howdy|sup|what'?s\s+up|yo+|greetings|good\s+(?:morning|afternoon|evening|day)|"
    r"hiya|heya|hi\s+there|hey\s+there|hello\s+there|"
    r"how\s+are\s+you|how'?s\s+it\s+going|how\s+do\s+you\s+do|"
    r"nice\s+to\s+meet\s+you|pleased\s+to\s+meet\s+you|"
    r"thanks|thank\s+you|thx|cheers|ty"
    r")[!?.]*\s*$",
    re.IGNORECASE,
)

_GREETING_RESPONSES = [
    "Hey there! 👋 I'm NeuralRAG — your personal document research assistant.\n\nYou can upload PDFs in the sidebar, index them, and then ask me anything about their contents. I can also search trusted web sources for you.\n\nWhat would you like to explore?",
    "Hello! 😊 Great to see you here.\n\nI'm NeuralRAG — I help you have conversations with your documents. Upload a PDF, index it, and start asking questions. I'll find the answers.\n\nWhat's on your mind?",
    "Hi! 👋 Welcome to NeuralRAG.\n\nI'm here to help you get the most out of your documents — summaries, specific questions, research, all of it. Just upload your PDFs and let's get started.\n\nHow can I help you today?",
    "Hey! Nice to meet you 🤝\n\nI'm NeuralRAG, an AI assistant built around your documents. Ask me to summarize a PDF, find specific information, or search trusted web sources.\n\nWhat would you like to know?",
]

_THANKS_RESPONSES = [
    "You're very welcome! 😊 Let me know if there's anything else I can help with.",
    "Happy to help! Feel free to ask anything else about your documents.",
    "Anytime! 👋 I'm here whenever you need me.",
]


def is_summary_intent(question: str) -> bool:
    return bool(_SUMMARY_INTENT_RE.search(question))


def is_greeting_intent(question: str) -> bool:
    return bool(_GREETING_RE.match(question))


def _get_greeting_response(question: str) -> str:
    q = question.strip().lower().rstrip("!?. ")
    if q in ("thanks", "thank you", "thx", "cheers", "ty"):
        return random.choice(_THANKS_RESPONSES)
    return random.choice(_GREETING_RESPONSES)


# ─────────────────────────────────────────────────────────────────────────────
# PATH HELPERS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def get_knowledge_base_dir(workspace: Optional[str] = None) -> str:
    return os.path.join("knowledge-base", workspace) if workspace else "knowledge-base"

def get_persist_directory(workspace: Optional[str] = None) -> str:
    return os.path.join("chroma_db", workspace) if workspace else "chroma_db"

def get_trusted_sources_file(workspace: Optional[str] = None) -> str:
    if workspace:
        os.makedirs("trusted_sources", exist_ok=True)
        return os.path.join("trusted_sources", f"{workspace}.json")
    return "trusted_sources.json"

KNOWLEDGE_BASE_DIR   = get_knowledge_base_dir()
PERSIST_DIRECTORY    = get_persist_directory()
TRUSTED_SOURCES_FILE = get_trusted_sources_file()


# ─────────────────────────────────────────────────────────────────────────────
# TRUSTED-SOURCES HELPERS (unchanged)
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

def is_trusted(url: str, trusted_domains: List[str]) -> bool:
    if not trusted_domains:
        return False
    d = domain_from_url(url)
    return any(d == td.lower() or d.endswith("." + td.lower()) for td in trusted_domains)

def validate_domain(raw: str) -> tuple[bool, str]:
    clean = (
        raw.strip().lower()
        .replace("https://", "").replace("http://", "").replace("www.", "")
        .split("/")[0].split("?")[0].split(":")[0]
    )
    if not clean:
        return False, "Domain cannot be empty."
    if not _DOMAIN_RE.match(clean):
        return False, f"'{clean}' is not a valid domain. Enter something like **arxiv.org** or **who.int**."
    return True, clean


# ─────────────────────────────────────────────────────────────────────────────
# PDF UPLOAD HELPER (unchanged)
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
# HELPER FUNCTIONS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_source_label(document: Document) -> str:
    source = document.metadata.get("source")
    title  = document.metadata.get("title")
    page   = document.metadata.get("page")
    if source and isinstance(source, str) and source.startswith(("http://", "https://")):
        return f"Web: {title} ({source})" if title else f"Web: {source}"
    filename = os.path.basename(str(source)) if source else "uploaded document"
    if page is not None:
        return f"PDF: {filename} (page {int(page) + 1})"
    return f"PDF: {filename}"

def _build_source_summary(documents: List[Document]) -> List[str]:
    seen = []
    for doc in documents:
        label = _normalize_source_label(doc)
        if label not in seen:
            seen.append(label)
    return seen

def _format_history(history: List[dict]) -> str:
    if not history:
        return "No prior conversation."
    turns = []
    for msg in history[-HISTORY_WINDOW:]:
        role    = msg.get("role", "user").capitalize()
        content = msg.get("content", "").strip()
        if content:
            turns.append(f"{role}: {content}")
    return "\n".join(turns) if turns else "No prior conversation."


# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN → HTML
# ─────────────────────────────────────────────────────────────────────────────

def _md_to_html(text: str) -> str:
    return md_lib.markdown(
        text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — v3: fixed layout + richer product feel
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&family=Inter:wght@400;450;500;600&display=swap');

/* ── DESIGN TOKENS ───────────────────────────────────────────────────────── */
:root {
  --bg-base:        #F8F7F4;
  --bg-sidebar:     #FAFAF8;
  --bg-card:        #FFFFFF;
  --bg-hover:       #F2F1EE;
  --bg-subtle:      #F9F8F6;
  --bg-input:       #FFFFFF;

  --border-subtle:  rgba(0,0,0,0.06);
  --border-light:   rgba(0,0,0,0.09);
  --border-mid:     rgba(0,0,0,0.14);
  --border-focus:   rgba(99,102,241,0.55);

  /* Richer indigo accent */
  --accent:         #6366F1;
  --accent-light:   #EEF2FF;
  --accent-hover:   #4F52DA;
  --accent-dark:    #3730A3;
  --accent-glow:    rgba(99,102,241,0.18);

  /* Semantic */
  --green:          #16A34A;
  --green-bg:       #F0FDF4;
  --green-border:   #86EFAC;
  --amber:          #B45309;
  --amber-bg:       #FFFBEB;
  --amber-border:   #FCD34D;
  --red:            #DC2626;
  --red-bg:         #FEF2F2;
  --red-border:     #FCA5A5;
  --purple:         #7C3AED;
  --purple-bg:      #F5F3FF;
  --purple-border:  #C4B5FD;
  --teal:           #0F766E;
  --teal-bg:        #F0FDFA;
  --teal-border:    #5EEAD4;
  --rose:           #E11D48;
  --rose-bg:        #FFF1F2;
  --rose-border:    #FECDD3;

  /* Text */
  --text-primary:   #18181B;
  --text-secondary: #52525B;
  --text-tertiary:  #A1A1AA;
  --text-inverse:   #FFFFFF;
  --font-display:   'Plus Jakarta Sans', system-ui, sans-serif;
  --font-body:      'Inter', system-ui, sans-serif;
  --font-mono:      'SF Mono','Fira Code','Cascadia Code',monospace;

  /* Shape */
  --r-xs:  6px;  --r-sm: 8px;  --r-md: 12px;
  --r-lg: 16px;  --r-xl: 20px; --r-full: 9999px;

  /* Shadows */
  --sh-xs:    0 1px 2px rgba(0,0,0,0.05);
  --sh-sm:    0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
  --sh-md:    0 4px 8px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.04);
  --sh-lg:    0 8px 24px rgba(0,0,0,0.08), 0 3px 8px rgba(0,0,0,0.04);
  --sh-focus: 0 0 0 3px var(--accent-glow);
  --sh-accent:0 4px 14px rgba(99,102,241,0.28);
}

/* ── BASE RESET ──────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
  background: var(--bg-base) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
  font-size: 15px; line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  /* CRITICAL: no overflow hidden, no fixed height that causes black strips */
  overflow-x: hidden !important;
}

/* ── STREAMLIT CHROME CLEANUP ────────────────────────────────────────────── */
[data-testid="stHeader"]  { background: transparent !important; height: 0 !important; min-height: 0 !important; overflow: visible !important; }
#MainMenu, footer         { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbar"] [data-testid="stToolbarActions"]  { display: none !important; }
[data-testid="stToolbar"] [data-testid="stAppDeployButton"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
div[data-testid="stChatMessage"] { background: transparent !important; padding: 0 !important; }

/* Main content area — padding-bottom gives room for the chat input */
.block-container {
  padding: 0 2.5rem 7rem !important;
  max-width: 880px !important;
  margin: 0 auto !important;
  /* No overflow hidden — that's what creates the black strip */
}

/* ── SCROLLBAR ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.12); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.22); }

/* ── SIDEBAR ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--bg-sidebar) !important;
  border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

.sb-brand {
  display: flex; align-items: center; gap: 11px;
  padding: 1.5rem 1.25rem 1.25rem;
  border-bottom: 1px solid var(--border-subtle);
}
.sb-brand-mark {
  width: 36px; height: 36px;
  background: linear-gradient(135deg, var(--accent), var(--accent-hover));
  border-radius: var(--r-md);
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; flex-shrink: 0;
  box-shadow: var(--sh-accent);
}
.sb-brand-name {
  font-family: var(--font-display); font-weight: 800;
  font-size: 1.025rem; color: var(--text-primary);
  letter-spacing: -0.022em; line-height: 1.1;
}
.sb-brand-name em { font-style: normal; color: var(--accent); }
.sb-brand-tag { font-size: 0.7rem; color: var(--text-tertiary); margin-top: 2px; }

.sb-section { padding: 1rem 1.25rem; border-bottom: 1px solid var(--border-subtle); }
.sb-section-last { border-bottom: none; }

.sb-head {
  font-family: var(--font-display); font-size: 0.68rem; font-weight: 700;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--text-tertiary); margin-bottom: 0.8rem;
}

.sb-doc {
  display: flex; align-items: center; gap: 9px;
  padding: 8px 10px; background: var(--bg-card);
  border: 1px solid var(--border-subtle); border-radius: var(--r-md);
  margin-bottom: 4px; box-shadow: var(--sh-xs);
  transition: border-color 0.15s, box-shadow 0.15s;
}
.sb-doc:hover { border-color: var(--border-light); box-shadow: var(--sh-sm); }
.sb-doc-icon { font-size: 13px; flex-shrink: 0; opacity: 0.65; }
.sb-doc-name { font-size: 0.79rem; color: var(--text-secondary); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.sb-domain {
  display: flex; align-items: center; gap: 8px; padding: 7px 10px;
  background: var(--green-bg); border: 1px solid var(--green-border);
  border-radius: var(--r-md); margin-bottom: 4px;
  font-size: 0.79rem; color: var(--green); font-weight: 500;
}

.sb-pill {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 10px; border-radius: var(--r-full);
  font-size: 0.7rem; font-weight: 600; margin-bottom: 8px;
  font-family: var(--font-display);
}
.sb-pill .dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; animation: pulse 2.2s ease-in-out infinite; flex-shrink: 0; }
.sb-pill.ok   { background: var(--green-bg); color: var(--green); border: 1px solid var(--green-border); }
.sb-pill.warn { background: var(--amber-bg); color: var(--amber); border: 1px solid var(--amber-border); }
.sb-pill.err  { background: var(--red-bg);   color: var(--red);   border: 1px solid var(--red-border); }

.sb-callout {
  display: flex; gap: 8px; padding: 9px 11px;
  background: var(--accent-light); border: 1px solid rgba(99,102,241,0.18);
  border-radius: var(--r-md); margin-top: 8px;
  font-size: 0.74rem; color: var(--text-secondary); line-height: 1.5;
}
.sb-callout-icon { color: var(--accent); flex-shrink: 0; }

.sb-stat { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; font-size: 0.77rem; }
.sb-stat-label { color: var(--text-tertiary); }
.sb-stat-val   { color: var(--text-secondary); font-weight: 500; }
.sb-stat-val.ok  { color: var(--green); }
.sb-stat-val.err { color: var(--red); }

/* ── SIDEBAR WIDGET OVERRIDES ────────────────────────────────────────────── */
section[data-testid="stSidebar"] .stRadio > label { display: none !important; }
section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
  font-size: 0.875rem !important; color: var(--text-secondary) !important; font-family: var(--font-body) !important;
}
section[data-testid="stSidebar"] .stFileUploader label { display: none !important; }
section[data-testid="stSidebar"] .stFileUploader section {
  background: var(--bg-card) !important; border: 2px dashed var(--border-light) !important;
  border-radius: var(--r-lg) !important; padding: 0.8rem !important; transition: border-color 0.15s, background 0.15s !important;
}
section[data-testid="stSidebar"] .stFileUploader section:hover {
  border-color: var(--accent) !important; background: var(--accent-light) !important;
}
section[data-testid="stSidebar"] .stTextInput input {
  background: var(--bg-card) !important; border: 1px solid var(--border-light) !important;
  border-radius: var(--r-md) !important; color: var(--text-primary) !important;
  font-size: 0.875rem !important; padding: 8px 12px !important;
  font-family: var(--font-body) !important; box-shadow: var(--sh-xs) !important;
}
section[data-testid="stSidebar"] .stTextInput input:focus {
  border-color: var(--border-focus) !important; box-shadow: var(--sh-focus) !important; outline: none !important;
}
section[data-testid="stSidebar"] .stTextInput label { display: none !important; }
section[data-testid="stSidebar"] [role="radio"] { accent-color: var(--accent) !important; }
section[data-testid="stSidebar"] .stFormSubmitButton > button {
  background: var(--bg-card) !important; color: var(--text-secondary) !important;
  border: 1px solid var(--border-light) !important; border-radius: var(--r-md) !important;
  font-family: var(--font-body) !important; font-size: 0.83rem !important;
  font-weight: 500 !important; padding: 0.45rem 1rem !important;
  box-shadow: var(--sh-xs) !important; width: 100% !important;
  transition: all 0.15s !important;
}
section[data-testid="stSidebar"] .stFormSubmitButton > button:hover {
  background: var(--bg-hover) !important; border-color: var(--border-mid) !important; color: var(--text-primary) !important;
}

/* ── MAIN BUTTONS ────────────────────────────────────────────────────────── */
.stButton > button {
  background: var(--bg-card) !important; color: var(--text-secondary) !important;
  border: 1px solid var(--border-light) !important; border-radius: var(--r-md) !important;
  font-family: var(--font-body) !important; font-size: 0.85rem !important; font-weight: 500 !important;
  padding: 0.5rem 1rem !important; box-shadow: var(--sh-xs) !important; transition: all 0.15s !important;
}
.stButton > button:hover { background: var(--bg-hover) !important; border-color: var(--border-mid) !important; color: var(--text-primary) !important; box-shadow: var(--sh-sm) !important; }
.stButton > button:active { background: var(--accent-light) !important; border-color: var(--accent) !important; color: var(--accent) !important; }

/* ── SPINNER ─────────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── ALERTS ──────────────────────────────────────────────────────────────── */
div[data-testid="stAlert"] { border-radius: var(--r-md) !important; font-family: var(--font-body) !important; font-size: 0.875rem !important; }

/* ── CHAT INPUT — FIXED: no sticky, no ::before that bleeds black ─────────── */
.stChatInput > div {
  background: var(--bg-input) !important;
  border: 1.5px solid var(--border-light) !important;
  border-radius: var(--r-xl) !important;
  box-shadow: var(--sh-md) !important;
  transition: border-color 0.18s, box-shadow 0.18s !important;
}
.stChatInput > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: var(--sh-focus), var(--sh-md) !important;
}
.stChatInput textarea {
  background: transparent !important; color: var(--text-primary) !important;
  font-family: var(--font-body) !important; font-size: 0.95rem !important; line-height: 1.55 !important;
}
.stChatInput textarea::placeholder { color: var(--text-tertiary) !important; }

/* ── EXPANDER ────────────────────────────────────────────────────────────── */
div[data-testid="stExpander"] {
  background: var(--bg-card) !important; border: 1px solid var(--border-subtle) !important;
  border-radius: var(--r-md) !important; margin-top: 6px !important; box-shadow: var(--sh-xs) !important;
}
div[data-testid="stExpander"] summary {
  color: var(--text-tertiary) !important; font-family: var(--font-body) !important;
  font-size: 0.78rem !important; font-weight: 500 !important; padding: 0.6rem 1rem !important;
}
div[data-testid="stExpander"] summary:hover { color: var(--text-secondary) !important; }

/* ── PAGE HEADER ─────────────────────────────────────────────────────────── */
.page-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 1.75rem 0 1.375rem;
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 0.5rem;
}
.ph-left  { display: flex; align-items: center; gap: 14px; }
.ph-right { display: flex; align-items: center; gap: 8px; }
.ph-mark {
  width: 46px; height: 46px;
  background: linear-gradient(135deg, var(--accent), var(--accent-hover));
  border-radius: var(--r-lg); display: flex; align-items: center; justify-content: center;
  font-size: 22px; box-shadow: var(--sh-accent);
}
.ph-title {
  font-family: var(--font-display); font-size: 1.45rem; font-weight: 800;
  letter-spacing: -0.03em; color: var(--text-primary); line-height: 1.1;
}
.ph-title em { font-style: normal; color: var(--accent); }
.ph-sub { font-size: 0.79rem; color: var(--text-tertiary); margin-top: 3px; }
.mode-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 14px; background: var(--accent-light);
  border: 1px solid rgba(99,102,241,0.22); border-radius: var(--r-full);
  font-size: 0.72rem; font-weight: 600; color: var(--accent);
  font-family: var(--font-display); letter-spacing: 0.01em;
}
.mode-chip .dot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent); animation: pulse 2s ease-in-out infinite; flex-shrink: 0; }

/* ── WELCOME SCREEN ──────────────────────────────────────────────────────── */
.welcome {
  display: flex; flex-direction: column; align-items: center;
  padding: 3.5rem 1rem 2.5rem; text-align: center;
  position: relative; overflow: hidden;
  animation: fadeUp 0.45s ease;
}

/* Decorative background orbs */
.welcome::before, .welcome::after {
  content: '';
  position: absolute; border-radius: 50%;
  pointer-events: none; z-index: 0;
  animation: float 6s ease-in-out infinite;
}
.welcome::before {
  width: 360px; height: 360px;
  background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
  top: -80px; left: 50%; transform: translateX(-50%);
}
.welcome::after {
  width: 260px; height: 260px;
  background: radial-gradient(circle, rgba(124,58,237,0.06) 0%, transparent 70%);
  top: 60px; left: 50%; transform: translateX(-30%);
  animation-delay: -3s;
}
.welcome > * { position: relative; z-index: 1; }

.welcome-avatar {
  width: 72px; height: 72px;
  background: linear-gradient(135deg, var(--accent), var(--accent-hover));
  border-radius: 22px; display: flex; align-items: center; justify-content: center;
  font-size: 34px; margin-bottom: 1.5rem;
  box-shadow: var(--sh-accent), 0 0 0 6px rgba(99,102,241,0.1);
  animation: popIn 0.5s cubic-bezier(0.34,1.56,0.64,1);
}
.welcome-eyebrow {
  font-family: var(--font-display); font-size: 0.72rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 0.625rem;
  background: var(--accent-light); padding: 4px 12px;
  border-radius: var(--r-full); border: 1px solid rgba(99,102,241,0.2);
  display: inline-block;
}
.welcome-title {
  font-family: var(--font-display); font-size: 2.25rem; font-weight: 800;
  letter-spacing: -0.035em; color: var(--text-primary);
  margin-bottom: 0.875rem; line-height: 1.15;
}
.welcome-title .accent-word {
  background: linear-gradient(135deg, var(--accent) 0%, var(--purple) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.welcome-sub {
  font-size: 1.05rem; color: var(--text-secondary);
  max-width: 420px; line-height: 1.75; margin-bottom: 3rem;
}

/* Feature steps */
.steps {
  display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;
  margin-bottom: 3rem; width: 100%; max-width: 600px;
}
.step {
  display: flex; flex-direction: column; align-items: center; gap: 10px;
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: var(--r-lg); padding: 1.25rem 1rem; width: 128px;
  box-shadow: var(--sh-sm); cursor: default;
  transition: all 0.22s cubic-bezier(0.34,1.4,0.64,1);
}
.step:hover {
  box-shadow: var(--sh-lg); border-color: rgba(99,102,241,0.25);
  transform: translateY(-4px) scale(1.02);
  background: linear-gradient(to bottom, var(--bg-card), var(--accent-light));
}
.step-num {
  width: 24px; height: 24px; background: var(--accent-light); color: var(--accent);
  border-radius: var(--r-full); font-size: 0.72rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center; font-family: var(--font-display);
  transition: background 0.2s, color 0.2s;
}
.step:hover .step-num { background: var(--accent); color: white; }
.step-icon { font-size: 1.5rem; transition: transform 0.2s; }
.step:hover .step-icon { transform: scale(1.15); }
.step-label { font-size: 0.77rem; font-weight: 600; color: var(--text-secondary); text-align: center; line-height: 1.3; font-family: var(--font-display); }

/* Prompt chips */
.prompts-label {
  font-family: var(--font-display); font-size: 0.68rem; font-weight: 700;
  letter-spacing: 0.09em; text-transform: uppercase; color: var(--text-tertiary);
  margin-bottom: 0.875rem; align-self: flex-start; width: 100%; max-width: 600px;
}
.prompts { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; width: 100%; max-width: 600px; }
.prompt-chip {
  display: flex; align-items: flex-start; justify-content: space-between; gap: 10px;
  padding: 13px 16px; background: var(--bg-card);
  border: 1px solid var(--border-subtle); border-radius: var(--r-lg);
  font-size: 0.875rem; color: var(--text-secondary); cursor: pointer;
  transition: all 0.18s ease; text-align: left; box-shadow: var(--sh-xs);
  line-height: 1.45;
}
.prompt-chip:hover {
  background: linear-gradient(135deg, var(--accent-light), #F5F3FF);
  border-color: rgba(99,102,241,0.3);
  box-shadow: var(--sh-sm), 0 0 0 1px rgba(99,102,241,0.1);
  color: var(--accent-dark);
  transform: translateY(-1px);
}
.prompt-chip-inner { display: flex; align-items: flex-start; gap: 10px; }
.prompt-icon { font-size: 1rem; flex-shrink: 0; margin-top: 2px; }
.prompt-arrow {
  font-size: 0.9rem; color: var(--text-tertiary); flex-shrink: 0;
  margin-top: 2px; opacity: 0;
  transition: opacity 0.15s, transform 0.15s, color 0.15s;
}
.prompt-chip:hover .prompt-arrow { opacity: 1; transform: translateX(2px); color: var(--accent); }

/* ── CONVERSATION ────────────────────────────────────────────────────────── */
.turn-group { margin: 0.5rem 0 1.75rem; animation: fadeUp 0.2s ease; }

/* User bubble */
.msg-user { display: flex; justify-content: flex-end; margin-bottom: 0.375rem; }
.msg-user-bubble {
  max-width: 68%;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
  color: white; border-radius: var(--r-xl) var(--r-xl) 5px var(--r-xl);
  padding: 0.875rem 1.125rem; font-size: 0.925rem; line-height: 1.65;
  box-shadow: var(--sh-accent); word-break: break-word;
}

/* AI response */
.msg-ai { display: flex; align-items: flex-start; gap: 12px; }
.msg-ai-avatar {
  width: 34px; height: 34px; background: linear-gradient(135deg, var(--accent), var(--accent-hover));
  border-radius: var(--r-md); display: flex; align-items: center; justify-content: center;
  font-size: 16px; flex-shrink: 0; margin-top: 2px;
  box-shadow: var(--sh-sm), 0 0 0 1px rgba(99,102,241,0.15);
}
.msg-ai-body { flex: 1; min-width: 0; }
.msg-ai-meta { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.msg-ai-name { font-family: var(--font-display); font-size: 0.8rem; font-weight: 700; color: var(--text-primary); }

/* Response card — subtle left accent border */
.ai-card {
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-left: 3px solid var(--accent);
  border-radius: 0 var(--r-xl) var(--r-xl) var(--r-xl);
  padding: 1.125rem 1.25rem; box-shadow: var(--sh-sm);
  transition: box-shadow 0.2s;
}
.ai-card:hover { box-shadow: var(--sh-md); }

/* Greeting card — warmer styling */
.greeting-card {
  background: linear-gradient(135deg, var(--accent-light) 0%, #F5F3FF 100%);
  border: 1px solid rgba(99,102,241,0.2);
  border-left: 3px solid var(--accent);
  border-radius: 0 var(--r-xl) var(--r-xl) var(--r-xl);
  padding: 1.125rem 1.25rem; box-shadow: var(--sh-sm);
}

/* Markdown prose inside cards */
.ai-prose { font-size: 0.925rem; color: var(--text-primary); line-height: 1.8; }
.ai-prose p { margin: 0 0 0.8em; }
.ai-prose p:last-child { margin-bottom: 0; }
.ai-prose strong { font-weight: 600; }
.ai-prose em { color: var(--text-secondary); font-style: italic; }
.ai-prose a { color: var(--accent); text-decoration: none; border-bottom: 1px solid rgba(99,102,241,0.3); }
.ai-prose a:hover { border-bottom-color: var(--accent); }
.ai-prose code {
  background: var(--bg-subtle); border: 1px solid var(--border-subtle); border-radius: var(--r-xs);
  padding: 2px 6px; font-family: var(--font-mono); font-size: 0.82em; color: var(--accent);
}
.ai-prose pre {
  background: var(--bg-subtle); border: 1px solid var(--border-subtle);
  border-radius: var(--r-md); padding: 1rem; overflow-x: auto; margin: 0.75em 0 1em;
}
.ai-prose pre code { background: none; border: none; padding: 0; font-size: 0.85em; color: var(--text-secondary); }
.ai-prose h1 { font-family: var(--font-display); font-size: 1.25em; font-weight: 800; color: var(--text-primary); margin: 1.5em 0 0.5em; letter-spacing: -0.02em; }
.ai-prose h2 { font-family: var(--font-display); font-size: 1.1em; font-weight: 700; color: var(--text-primary); margin: 1.3em 0 0.45em; letter-spacing: -0.015em; }
.ai-prose h3 { font-family: var(--font-display); font-size: 0.98em; font-weight: 700; color: var(--text-secondary); margin: 1.1em 0 0.4em; }
.ai-prose ul, .ai-prose ol { padding-left: 1.5em; margin: 0.35em 0 0.85em; }
.ai-prose li { margin-bottom: 0.3em; }
.ai-prose li > p { margin: 0; }
.ai-prose blockquote { border-left: 3px solid var(--border-mid); margin: 0.6em 0; padding: 0.4em 0 0.4em 1em; color: var(--text-secondary); }
.ai-prose table { border-collapse: collapse; width: 100%; margin: 0.75em 0 1em; font-size: 0.88em; }
.ai-prose th { background: var(--bg-subtle); font-family: var(--font-display); font-weight: 600; text-align: left; padding: 8px 12px; border: 1px solid var(--border-light); }
.ai-prose td { padding: 7px 12px; border: 1px solid var(--border-subtle); }
.ai-prose tr:nth-child(even) td { background: var(--bg-subtle); }
.ai-prose hr { border: none; border-top: 1px solid var(--border-subtle); margin: 1.2em 0; }

/* Source pills */
.msg-sources {
  display: flex; flex-wrap: wrap; align-items: center; gap: 5px;
  margin-top: 12px; padding-top: 11px; border-top: 1px solid var(--border-subtle);
}
.src-label { font-size: 0.68rem; color: var(--text-tertiary); font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; margin-right: 3px; }
.src-pill {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 10px; border-radius: var(--r-full); font-size: 0.71rem; font-weight: 500; border: 1px solid;
}
.src-pill.rag { background: var(--accent-light); color: var(--accent); border-color: rgba(99,102,241,0.22); }
.src-pill.web { background: var(--teal-bg);     color: var(--teal);   border-color: var(--teal-border); }
.src-pill.sum { background: var(--purple-bg);   color: var(--purple); border-color: var(--purple-border); }
.src-pill.err { background: var(--red-bg);      color: var(--red);    border-color: var(--red-border); }

/* Type badge */
.type-badge {
  display: inline-flex; align-items: center; gap: 4px; padding: 3px 8px;
  border-radius: var(--r-xs); font-size: 0.68rem; font-weight: 600;
  letter-spacing: 0.04em; text-transform: uppercase; font-family: var(--font-display);
}
.type-badge.rag      { background: var(--accent-light); color: var(--accent); }
.type-badge.web      { background: var(--teal-bg);      color: var(--teal); }
.type-badge.summary  { background: var(--purple-bg);    color: var(--purple); }
.type-badge.greeting { background: var(--accent-light); color: var(--accent); }
.type-badge.err      { background: var(--red-bg);       color: var(--red); }

/* ── SUMMARY REPORT CARD ─────────────────────────────────────────────────── */
.summary-card {
  background: var(--bg-card); border: 1px solid var(--purple-border);
  border-left: 3px solid var(--purple);
  border-radius: 0 var(--r-xl) var(--r-xl) var(--r-xl);
  overflow: hidden; box-shadow: var(--sh-md);
}
.summary-header {
  background: linear-gradient(135deg, #EDE9FE 0%, var(--accent-light) 100%);
  border-bottom: 1px solid var(--purple-border);
  padding: 1.125rem 1.375rem; display: flex; align-items: center; gap: 12px;
}
.summary-header-icon {
  width: 38px; height: 38px; background: white;
  border: 1px solid var(--purple-border); border-radius: var(--r-md);
  display: flex; align-items: center; justify-content: center;
  font-size: 19px; flex-shrink: 0; box-shadow: var(--sh-xs);
}
.summary-header-title { font-family: var(--font-display); font-size: 0.95rem; font-weight: 700; color: var(--text-primary); }
.summary-header-sub   { font-size: 0.73rem; color: var(--text-tertiary); margin-top: 2px; }
.summary-report-badge {
  margin-left: auto; padding: 4px 11px;
  background: var(--purple-bg); border: 1px solid var(--purple-border);
  border-radius: var(--r-full); font-size: 0.69rem; font-weight: 700;
  color: var(--purple); font-family: var(--font-display); letter-spacing: 0.05em; text-transform: uppercase;
}
.summary-body { padding: 1.375rem 1.5rem; font-size: 0.915rem; line-height: 1.8; color: var(--text-primary); }
.summary-body p { margin: 0 0 0.75em; }
.summary-body p:last-child { margin-bottom: 0; }
.summary-body strong { font-weight: 600; }
.summary-body h1 {
  font-family: var(--font-display); font-size: 1.1em; font-weight: 800;
  color: var(--text-primary); margin: 1.5em 0 0.5em; letter-spacing: -0.02em;
  padding-bottom: 0.4em; border-bottom: 1px solid var(--border-subtle);
}
.summary-body h2 {
  font-family: var(--font-display); font-size: 0.95em; font-weight: 700;
  color: var(--purple); margin: 1.4em 0 0.45em; letter-spacing: -0.01em;
  display: flex; align-items: center; gap: 8px;
}
.summary-body h2::before {
  content: ''; display: inline-block; width: 3px; height: 1em;
  background: var(--purple); border-radius: 2px; flex-shrink: 0;
}
.summary-body h3 { font-family: var(--font-display); font-size: 0.88em; font-weight: 700; color: var(--text-secondary); margin: 1.1em 0 0.4em; }
.summary-body ul, .summary-body ol { padding-left: 1.5em; margin: 0.4em 0 0.85em; }
.summary-body li { margin-bottom: 0.35em; }
.summary-body code { background: var(--bg-subtle); border: 1px solid var(--border-subtle); border-radius: var(--r-xs); padding: 2px 5px; font-family: var(--font-mono); font-size: 0.82em; color: var(--purple); }
.summary-body blockquote { border-left: 3px solid var(--purple-border); margin: 0.6em 0; padding: 0.4em 0 0.4em 1em; color: var(--text-secondary); }

/* ── WORKFLOW TRACE CARD ─────────────────────────────────────────────────── */
.trace-card {
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: var(--r-lg); padding: 1rem 1.25rem; margin-bottom: 1rem;
  box-shadow: var(--sh-sm); animation: fadeUp 0.2s ease;
}
.trace-title {
  font-family: var(--font-display); font-size: 0.68rem; font-weight: 700;
  letter-spacing: 0.09em; text-transform: uppercase; color: var(--text-tertiary); margin-bottom: 0.75rem;
}
.trace-row { display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--border-subtle); }
.trace-row:last-child { border-bottom: none; padding-bottom: 0; }
.trace-left { display: flex; align-items: center; gap: 9px; }
.trace-icon { font-size: 13px; width: 20px; text-align: center; flex-shrink: 0; }
.trace-name { font-size: 0.845rem; font-weight: 600; color: var(--text-primary); font-family: var(--font-display); }
.trace-desc { font-size: 0.75rem; color: var(--text-tertiary); margin-left: 2px; }
.trace-pill {
  font-size: 0.67rem; font-weight: 700; padding: 2px 9px;
  border-radius: var(--r-full); letter-spacing: 0.04em; font-family: var(--font-display);
}
.trace-pill.running { background: var(--amber-bg);  color: var(--amber);  border: 1px solid var(--amber-border); }
.trace-pill.done    { background: var(--green-bg);  color: var(--green);  border: 1px solid var(--green-border); }
.trace-pill.pending { background: var(--bg-hover);  color: var(--text-tertiary); border: 1px solid var(--border-light); }
.trace-pill.blocked { background: var(--red-bg);    color: var(--red);    border: 1px solid var(--red-border); }
.trace-pill.skipped { background: var(--bg-hover);  color: var(--text-tertiary); border: 1px solid var(--border-light); opacity: 0.55; }

/* ── ANIMATIONS ──────────────────────────────────────────────────────────── */
@keyframes pulse   { 0%,100%{opacity:1} 50%{opacity:0.35} }
@keyframes fadeUp  { from{opacity:0;transform:translateY(7px)} to{opacity:1;transform:translateY(0)} }
@keyframes float   { 0%,100%{transform:translateX(-50%) translateY(0)} 50%{transform:translateX(-50%) translateY(-12px)} }
@keyframes popIn   { from{opacity:0;transform:scale(0.7)} to{opacity:1;transform:scale(1)} }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ALL BACKEND FUNCTIONS — COMPLETELY UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

def ingest_pdfs_into_vectordb(workspace: Optional[str] = None):
    import shutil
    kb_dir  = get_knowledge_base_dir(workspace)
    persist = get_persist_directory(workspace)
    documents = []
    if not os.path.exists(kb_dir):
        return 0
    for file_name in os.listdir(kb_dir):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(kb_dir, file_name)
            try:
                loader = PyPDFLoader(file_path)
                pdf_docs = loader.load()
                for doc in pdf_docs:
                    doc.metadata["source"]   = file_name
                    doc.metadata["filename"] = file_name
                documents.extend(pdf_docs)
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


def load_all_pdf_pages(workspace: Optional[str] = None) -> dict[str, List[Document]]:
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


SUMMARY_SINGLEPASS_PAGE_LIMIT = 50
SUMMARY_SINGLEPASS_CHARS      = 5_500
SUMMARY_BATCH_CHARS           = 3_500
SUMMARY_REDUCE_CHARS          = 3_000


def _chunk_pages_for_map(pdf_map: dict[str, List[Document]]) -> tuple[List[tuple[str, str]], List[str]]:
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


# ── Graph state — extended with is_greeting ───────────────────────────────────
class GraphState(TypedDict, total=False):
    question:         str
    history:          List[dict]
    documents:        List[Document]
    sources:          List[str]
    needs_web_search: bool
    source_type:      str
    answer:           str
    trusted_domains:  List[str]
    source_mode:      str
    is_summary:       bool
    is_greeting:      bool   # NEW


# ── Routing — greeting check added ───────────────────────────────────────────
def route_from_start(state: GraphState) -> str:
    if state.get("is_greeting"):
        return "greet"
    if state.get("is_summary"):
        return "summarize"
    return "retrieve"


def route_after_retrieval(state: GraphState) -> str:
    mode = state.get("source_mode", "hybrid")
    if mode == "docs_only":
        return "generate"
    return "web_search" if state.get("needs_web_search") else "generate"


# ── Greeting node (NEW) ───────────────────────────────────────────────────────
def greeting_node(state: GraphState) -> GraphState:
    """Short-circuit: return a friendly greeting without touching RAG or web search."""
    question = state.get("question", "")
    return {
        "answer":      _get_greeting_response(question),
        "sources":     [],
        "source_type": "greeting",
        "documents":   [],
    }


# ── Summarize node (unchanged logic) ─────────────────────────────────────────
def _total_page_count(pdf_map: dict[str, List[Document]]) -> int:
    return sum(len(pages) for pages in pdf_map.values())


def _build_singlepass_context(pdf_map: dict[str, List[Document]]) -> str:
    all_pages: List[tuple[str, Document]] = []
    for filename, pages in pdf_map.items():
        for doc in pages:
            all_pages.append((filename, doc))
    full_text_parts: List[str] = []
    total = 0
    for filename, doc in all_pages:
        page_num = doc.metadata.get("page", "?")
        text = doc.page_content.strip()
        if not text:
            continue
        part = f"[{filename} · Page {int(page_num)+1 if isinstance(page_num, int) else page_num}]\n{text}"
        if total + len(part) > SUMMARY_SINGLEPASS_CHARS:
            break
        full_text_parts.append(part)
        total += len(part)
    else:
        return "\n\n".join(full_text_parts)
    non_empty = [(fn, d) for fn, d in all_pages if d.page_content.strip()]
    if not non_empty:
        return ""
    avg_page_chars = sum(len(d.page_content) for _, d in non_empty) / len(non_empty)
    max_pages = max(1, int(SUMMARY_SINGLEPASS_CHARS / max(avg_page_chars, 1)))
    step = max(1, len(non_empty) // max_pages)
    sampled = non_empty[::step][:max_pages]
    parts: List[str] = []
    total = 0
    for filename, doc in sampled:
        page_num = doc.metadata.get("page", "?")
        text = doc.page_content.strip()
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


def summarize_node(state: GraphState) -> GraphState:
    history  = state.get("history", [])
    pdf_map  = load_all_pdf_pages()
    if not pdf_map:
        return {
            "answer": (
                "No documents found.\n\n"
                "Please upload one or more PDFs using the sidebar, "
                "then click **Index Documents** before requesting a summary."
            ),
            "sources":     [],
            "source_type": "summary_no_docs",
            "documents":   [],
        }
    total_pages   = _total_page_count(pdf_map)
    doc_list      = ", ".join(pdf_map.keys())
    source_labels = [f"PDF: {fn} ({len(pg)} pages)" for fn, pg in pdf_map.items()]
    all_docs      = [doc for pages in pdf_map.values() for doc in pages]
    llm           = ChatGroq(temperature=0, model_name=LLM_MODEL_ID)
    history_text  = _format_history(history)
    use_single_pass = total_pages <= SUMMARY_SINGLEPASS_PAGE_LIMIT

    if use_single_pass:
        context = _build_singlepass_context(pdf_map)
        prompt = (
            f"You are an expert document analyst. Document(s): {doc_list}\n\n"
            "Produce a **comprehensive structured summary** covering ALL sections below "
            "(omit only if absolutely no relevant content):\n\n"
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
            resp = llm.invoke(prompt)
            return {"answer": resp.content, "sources": source_labels, "source_type": "summary", "documents": all_docs}
        except Exception as exc:
            return {"answer": f"Summarization failed: {exc}", "sources": source_labels, "source_type": "summary", "documents": []}

    batches, _ = _chunk_pages_for_map(pdf_map)
    partial_summaries: List[str] = []
    for idx, (filename, batch_text) in enumerate(batches):
        map_prompt = (
            f"You are summarizing a section of '{filename}' (batch {idx+1}/{len(batches)}).\n\n"
            "Extract and list key points: objectives, methods, technologies, workflow steps, results, conclusions.\n\n"
            f"--- SECTION ---\n{batch_text}\n--- END ---\n\nKey points:"
        )
        try:
            resp = llm.invoke(map_prompt)
            partial_summaries.append(f"[From: {filename}, batch {idx+1}]\n{resp.content.strip()}")
        except Exception as exc:
            partial_summaries.append(f"[From: {filename}, batch {idx+1}] (failed: {exc})")

    if not partial_summaries or all("failed" in p for p in partial_summaries):
        return {"answer": "All map batches failed — check your Groq API key.", "sources": source_labels, "source_type": "summary", "documents": []}

    combined = "\n\n".join(partial_summaries)
    if len(combined) > SUMMARY_REDUCE_CHARS:
        combined = combined[:SUMMARY_REDUCE_CHARS] + "\n\n[...omitted...]"

    reduce_prompt = (
        f"Synthesise these extracts from {doc_list} into a structured summary:\n\n"
        f"{_SUMMARY_STRUCTURED_SECTIONS}\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"--- EXTRACTS ---\n{combined}\n--- END ---\n\nComprehensive structured summary:"
    )
    try:
        resp = llm.invoke(reduce_prompt)
        return {"answer": resp.content, "sources": source_labels, "source_type": "summary", "documents": all_docs}
    except Exception as exc:
        fallback = "Final synthesis failed — per-section extracts:\n\n" + "\n\n".join(partial_summaries)
        return {"answer": fallback, "sources": source_labels, "source_type": "summary", "documents": []}


def retrieve_node(state: GraphState) -> GraphState:
    mode = state.get("source_mode", "hybrid")
    if mode == "web_only":
        return {"documents": [], "sources": [], "needs_web_search": True}
    question    = state["question"]
    vectorstore = get_vectorstore()
    retriever   = create_retriever()
    if vectorstore is None:
        return {"documents": [], "sources": [], "needs_web_search": True}
    relevant_docs: List[Document] = []
    try:
        docs_and_scores = vectorstore.similarity_search_with_relevance_scores(question, k=RETRIEVAL_K)
        if not docs_and_scores:
            return {"documents": [], "sources": [], "needs_web_search": True}
        all_scores = [s for _, s in docs_and_scores]
        max_score  = max(all_scores)
        if max_score > 1.0:
            relevant_docs = [d for d, s in docs_and_scores if s <= 1.5]
        else:
            relevant_docs = [d for d, s in docs_and_scores if s >= RETRIEVAL_THRESHOLD]
    except Exception:
        relevant_docs = retriever.invoke(question) if retriever is not None else []
    if not relevant_docs:
        return {"documents": [], "sources": [], "needs_web_search": True}
    return {"documents": relevant_docs, "sources": _build_source_summary(relevant_docs), "needs_web_search": False, "source_type": "rag"}


def web_search_node(state: GraphState) -> GraphState:
    question        = state["question"]
    trusted_domains = state.get("trusted_domains", [])
    if not trusted_domains:
        return {"documents": [], "sources": [], "source_type": "web_no_trusted_config"}
    tavily_search  = TavilySearch(max_results=WEB_MAX_RESULTS, search_depth="advanced")
    search_results = tavily_search.invoke(question)
    scraped_docs   = []
    source_labels  = []
    if not search_results or "results" not in search_results:
        return {"documents": [], "sources": [], "source_type": "web"}
    raw_results      = search_results["results"][:WEB_MAX_RESULTS]
    filtered_results = [r for r in raw_results if is_trusted(r.get("url", ""), trusted_domains)]
    if not filtered_results:
        return {"documents": [], "sources": [], "source_type": "web_no_trusted"}
    for result in filtered_results:
        url     = result.get("url")
        title   = result.get("title")
        snippet = result.get("content") or result.get("snippet") or ""
        if url:
            source_labels.append(f"Web: {title} ({url})" if title else f"Web: {url}")
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
            scraped_docs.append(Document(page_content=snippet, metadata={"source": url or title or "web", "title": title or ""}))
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
                "**Web search is blocked** — no trusted sources are configured yet.\n\n"
                "Add one or more trusted domains in the sidebar under **Trusted Sources**. "
                "If you have uploaded PDFs, try rephrasing your question so it can be answered from your documents."
            ),
            "sources":     [],
            "source_type": "web_no_trusted_config",
        }
    if source_type == "web_no_trusted":
        return {
            "answer": (
                "**No answer generated** — all web results came from untrusted domains.\n\n"
                "Add relevant trusted sources in the sidebar or refine your query."
            ),
            "sources":     [],
            "source_type": "web_no_trusted",
        }
    context      = "\n\n".join(f"[{_normalize_source_label(doc)}]\n{doc.page_content}" for doc in documents[:10])
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
    llm      = ChatGroq(temperature=0, model_name=LLM_MODEL_ID)
    response = llm.invoke(prompt)
    return {"answer": response.content, "sources": state.get("sources", _build_source_summary(documents)), "source_type": source_type}


@st.cache_resource
def build_graph():
    workflow = StateGraph(GraphState)
    # Nodes
    workflow.add_node("greet",      greeting_node)    # NEW
    workflow.add_node("retrieve",   retrieve_node)
    workflow.add_node("summarize",  summarize_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate",   generate_node)
    # Edges
    workflow.add_conditional_edges(
        START, route_from_start,
        {"greet": "greet", "summarize": "summarize", "retrieve": "retrieve"}
    )
    workflow.add_edge("greet",    END)
    workflow.add_edge("summarize", END)
    workflow.add_conditional_edges(
        "retrieve", route_after_retrieval,
        {"generate": "generate", "web_search": "web_search"}
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate",   END)
    return workflow.compile()


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sb-brand">
          <div class="sb-brand-mark">📚</div>
          <div>
            <div class="sb-brand-name">Neural<em>RAG</em></div>
            <div class="sb-brand-tag">Your document companion</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Query Mode
        st.markdown('<div class="sb-section"><div class="sb-head">⚡ Query Mode</div>', unsafe_allow_html=True)
        mode_map    = {"hybrid": "Hybrid — Documents + Web", "docs_only": "Documents Only", "web_only": "Trusted Web Only"}
        mode_keys   = list(mode_map.keys())
        mode_values = list(mode_map.values())
        if "source_mode" not in st.session_state:
            st.session_state.source_mode = "hybrid"
        idx      = mode_keys.index(st.session_state.source_mode)
        selected = st.radio("mode", options=mode_values, index=idx, label_visibility="collapsed")
        st.session_state.source_mode = mode_keys[mode_values.index(selected)]
        mode_desc = {
            "hybrid":    "Search your documents first, fall back to trusted web sources if needed.",
            "docs_only": "Only answer from uploaded PDFs. No web search.",
            "web_only":  "Skip documents entirely. Query your trusted domains only.",
        }
        st.markdown(
            f'<div class="sb-callout"><span class="sb-callout-icon">ℹ</span>'
            f'<span>{mode_desc[st.session_state.source_mode]}</span></div></div>',
            unsafe_allow_html=True,
        )

        # Documents
        st.markdown('<div class="sb-section"><div class="sb-head">📄 Documents</div>', unsafe_allow_html=True)
        if "saved_pdf_names" not in st.session_state:
            st.session_state.saved_pdf_names = set()
        uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, key="pdf_uploader", label_visibility="collapsed")
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
            st.markdown(f'<div class="sb-pill ok" style="margin:8px 0 6px"><span class="dot"></span>{pdf_count} document{"s" if pdf_count>1 else ""} ready</div>', unsafe_allow_html=True)
            for pdf in pdf_files:
                c1, c2 = st.columns([5, 1])
                c1.markdown(f'<div class="sb-doc"><span class="sb-doc-icon">📄</span><span class="sb-doc-name">{pdf}</span></div>', unsafe_allow_html=True)
                if c2.button("✕", key=f"del_{pdf}", help=f"Remove {pdf}"):
                    try:
                        os.remove(os.path.join(KNOWLEDGE_BASE_DIR, pdf))
                        st.session_state.saved_pdf_names.discard(pdf)
                        st.success("Removed — click Index to rebuild")
                        st.rerun()
                    except OSError as e:
                        st.error(f"Could not remove: {e}")
        else:
            st.markdown('<div class="sb-pill warn" style="margin:8px 0 6px"><span class="dot"></span>No documents yet</div>', unsafe_allow_html=True)
        if st.button("↻  Index Documents", use_container_width=True):
            with st.spinner("Embedding your documents…"):
                doc_count = ingest_pdfs_into_vectordb()
                st.cache_resource.clear()
                st.success(f"✓ {doc_count} pages indexed") if doc_count > 0 else st.warning("No documents to index")
        st.markdown('</div>', unsafe_allow_html=True)

        # Trusted Sources
        st.markdown('<div class="sb-section"><div class="sb-head">🌐 Trusted Sources</div>', unsafe_allow_html=True)
        trusted_domains = load_trusted_sources()
        if trusted_domains:
            st.markdown(f'<div class="sb-pill ok" style="margin-bottom:6px"><span class="dot"></span>{len(trusted_domains)} domain{"s" if len(trusted_domains)>1 else ""} trusted</div>', unsafe_allow_html=True)
            for i, domain in enumerate(trusted_domains):
                cd, cr = st.columns([5, 1])
                cd.markdown(f'<div class="sb-domain"><span>✓</span>{domain}</div>', unsafe_allow_html=True)
                if cr.button("✕", key=f"rm_{i}", help=f"Remove {domain}"):
                    save_trusted_sources([d for j, d in enumerate(trusted_domains) if j != i])
                    st.rerun()
        else:
            st.markdown('<div class="sb-pill err" style="margin-bottom:6px"><span class="dot"></span>Web search blocked</div>', unsafe_allow_html=True)
        with st.form(key="add_domain_form", clear_on_submit=True):
            new_domain = st.text_input("Domain", placeholder="e.g. arxiv.org", label_visibility="collapsed")
            submitted  = st.form_submit_button("+ Add trusted domain", use_container_width=True)
            if submitted and new_domain.strip():
                valid, result = validate_domain(new_domain)
                if not valid:
                    st.error(result)
                elif result in trusted_domains:
                    st.info(f"{result} is already trusted")
                else:
                    trusted_domains.append(result)
                    save_trusted_sources(trusted_domains)
                    st.success(f"✓ Added: {result}")
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # System
        chroma_ok = os.path.exists(PERSIST_DIRECTORY)
        st.markdown(f"""
        <div class="sb-section sb-section-last">
          <div class="sb-head">⚙ System</div>
          <div class="sb-stat"><span class="sb-stat-label">Vector store</span><span class="sb-stat-val {'ok' if chroma_ok else 'err'}">{'● Ready' if chroma_ok else '● Not indexed'}</span></div>
          <div class="sb-stat"><span class="sb-stat-label">Model</span><span class="sb-stat-val">gpt-oss-20b</span></div>
          <div class="sb-stat"><span class="sb-stat-label">Embeddings</span><span class="sb-stat-val">MiniLM-L6-v2</span></div>
          <div class="sb-stat"><span class="sb-stat-label">Web search</span><span class="sb-stat-val">Tavily</span></div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("messages"):
            st.write("")
            if st.button("Clear conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()


def render_header():
    mode       = st.session_state.get("source_mode", "hybrid")
    mode_label = {"hybrid": "Hybrid mode", "docs_only": "Documents only", "web_only": "Web only"}.get(mode, "Hybrid mode")
    st.markdown(f"""
    <div class="page-header">
      <div class="ph-left">
        <div class="ph-mark">📚</div>
        <div>
          <div class="ph-title">Neural<em>RAG</em></div>
          <div class="ph-sub">Turn your documents into conversations</div>
        </div>
      </div>
      <div class="ph-right">
        <div class="mode-chip"><span class="dot"></span>{mode_label}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state():
    st.markdown("""
    <div class="welcome">
      <div class="welcome-avatar">📚</div>
      <div class="welcome-eyebrow">AI Document Research</div>
      <div class="welcome-title">
        What would you like to<br>
        <span class="accent-word">learn today?</span>
      </div>
      <div class="welcome-sub">
        Upload your documents, add trusted sources, and start a conversation.
        NeuralRAG finds the right answer from the right place — instantly.
      </div>

      <div class="steps">
        <div class="step">
          <div class="step-num">1</div>
          <div class="step-icon">📄</div>
          <div class="step-label">Upload a PDF</div>
        </div>
        <div class="step">
          <div class="step-num">2</div>
          <div class="step-icon">⚡</div>
          <div class="step-label">Index it</div>
        </div>
        <div class="step">
          <div class="step-num">3</div>
          <div class="step-icon">🌐</div>
          <div class="step-label">Add trusted sources</div>
        </div>
        <div class="step">
          <div class="step-num">4</div>
          <div class="step-icon">💬</div>
          <div class="step-label">Ask anything</div>
        </div>
      </div>

      <div class="prompts-label">Try asking</div>
      <div class="prompts">
        <div class="prompt-chip" onclick="(function(){var t=document.querySelector('.stChatInput textarea');t.value='Summarize my document';t.dispatchEvent(new Event('input',{bubbles:true}));t.focus();})()">
          <div class="prompt-chip-inner">
            <span class="prompt-icon">📋</span>
            <span>Summarize my document</span>
          </div>
          <span class="prompt-arrow">→</span>
        </div>
        <div class="prompt-chip" onclick="(function(){var t=document.querySelector('.stChatInput textarea');t.value='What are the main findings in this PDF?';t.dispatchEvent(new Event('input',{bubbles:true}));t.focus();})()">
          <div class="prompt-chip-inner">
            <span class="prompt-icon">🔍</span>
            <span>What are the main findings?</span>
          </div>
          <span class="prompt-arrow">→</span>
        </div>
        <div class="prompt-chip" onclick="(function(){var t=document.querySelector('.stChatInput textarea');t.value='What technologies are used in this project?';t.dispatchEvent(new Event('input',{bubbles:true}));t.focus();})()">
          <div class="prompt-chip-inner">
            <span class="prompt-icon">🛠</span>
            <span>What technologies are used?</span>
          </div>
          <span class="prompt-arrow">→</span>
        </div>
        <div class="prompt-chip" onclick="(function(){var t=document.querySelector('.stChatInput textarea');t.value='Search trusted sources for the latest research';t.dispatchEvent(new Event('input',{bubbles:true}));t.focus();})()">
          <div class="prompt-chip-inner">
            <span class="prompt-icon">🌐</span>
            <span>Search trusted web sources</span>
          </div>
          <span class="prompt-arrow">→</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _badge_html(source_type: str) -> str:
    badges = {
        "rag":                '<span class="type-badge rag">📄 PDF</span>',
        "web":                '<span class="type-badge web">🌐 Web</span>',
        "summary":            '<span class="type-badge summary">✦ Summary</span>',
        "greeting":           '<span class="type-badge greeting">👋 Hello</span>',
        "web_no_trusted":     '<span class="type-badge err">⚠ Blocked</span>',
        "web_no_trusted_config": '<span class="type-badge err">⚠ Blocked</span>',
    }
    return badges.get(source_type, "")


def _source_pills_html(domain_list: List[str], source_type: str) -> str:
    if not domain_list:
        return ""
    pill_cls = {"summary": "sum", "rag": "rag"}.get(source_type, "web")
    pills = "".join(f'<span class="src-pill {pill_cls}">{d}</span>' for d in domain_list)
    return f'<div class="msg-sources"><span class="src-label">Sources</span>{pills}</div>'


def render_message(
    role: str,
    content: str,
    source: str = None,
    source_type: str = None,
    domain_list: List[str] = None,
):
    # User
    if role == "user":
        safe = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f'<div class="turn-group"><div class="msg-user">'
            f'<div class="msg-user-bubble">{safe}</div></div></div>',
            unsafe_allow_html=True,
        )
        return

    # Assistant
    badge        = _badge_html(source_type)
    pills_html   = _source_pills_html(domain_list or [], source_type)
    content_html = _md_to_html(content)

    if source_type == "summary":
        doc_label  = domain_list[0] if domain_list else "Document"
        inner_html = (
            f'<div class="summary-card">'
            f'<div class="summary-header">'
            f'  <div class="summary-header-icon">📋</div>'
            f'  <div><div class="summary-header-title">Document Summary</div>'
            f'  <div class="summary-header-sub">{doc_label}</div></div>'
            f'  <div class="summary-report-badge">Report</div>'
            f'</div>'
            f'<div class="summary-body">{content_html}</div>'
            f'{pills_html}'
            f'</div>'
        )
    elif source_type == "greeting":
        inner_html = (
            f'<div class="greeting-card">'
            f'<div class="ai-prose">{content_html}</div>'
            f'</div>'
        )
    else:
        inner_html = (
            f'<div class="ai-card">'
            f'<div class="ai-prose">{content_html}</div>'
            f'{pills_html}'
            f'</div>'
        )

    st.markdown(
        f'<div class="turn-group">'
        f'<div class="msg-ai">'
        f'  <div class="msg-ai-avatar">📚</div>'
        f'  <div class="msg-ai-body">'
        f'    <div class="msg-ai-meta"><span class="msg-ai-name">NeuralRAG</span>{badge}</div>'
        f'    {inner_html}'
        f'  </div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    if source:
        with st.expander("View sources", expanded=False):
            st.code(source, language=None)


def render_workflow_trace(placeholder, steps):
    step_icons = {"Intent Check": "🔍", "Greeting": "👋", "Summarizer": "📋",
                  "Retrieval": "📄", "Web Search": "🌐", "Trust Filter": "🛡", "Generator": "✨"}
    rows = "".join(
        f'<div class="trace-row">'
        f'<div class="trace-left">'
        f'  <span class="trace-icon">{step_icons.get(s["name"],"·")}</span>'
        f'  <span class="trace-name">{s["name"]}</span>'
        f'  <span class="trace-desc">— {s["desc"]}</span>'
        f'</div>'
        f'<span class="trace-pill {s["status"]}">{s["status"]}</span>'
        f'</div>'
        for s in steps
    )
    with placeholder.container():
        st.markdown(
            f'<div class="trace-card"><div class="trace-title">Processing your question</div>{rows}</div>',
            unsafe_allow_html=True,
        )


def _extract_domain_labels(sources: List[str], source_type: str) -> List[str]:
    labels = []
    for s in sources:
        if source_type in ("rag", "summary"):
            if s.startswith("PDF:"):
                filename = s[4:].strip().split("(")[0].strip()
                if filename and filename not in labels:
                    labels.append(filename)
        elif source_type in ("web", "web_no_trusted", "web_no_trusted_config"):
            if s.startswith("Web:"):
                part = s[4:].strip()
                url  = part[part.rfind("(")+1:part.rfind(")")] if "(" in part else part
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

    if question := st.chat_input("Ask about your documents, or search the web…"):
        st.session_state.messages.append({"role": "user", "content": question})

        trace_placeholder = st.empty()
        source_mode       = st.session_state.get("source_mode", "hybrid")
        summary_intent    = is_summary_intent(question)
        greeting_intent   = is_greeting_intent(question)

        # Build trace steps
        if greeting_intent:
            steps = [
                {"name": "Intent Check", "desc": "Greeting detected",    "status": "done"},
                {"name": "Greeting",     "desc": "Preparing response",   "status": "running"},
            ]
        elif summary_intent:
            steps = [
                {"name": "Intent Check", "desc": "Summary intent detected",          "status": "done"},
                {"name": "Summarizer",   "desc": "Loading full documents from disk",  "status": "running"},
            ]
        elif source_mode == "docs_only":
            steps = [
                {"name": "Intent Check", "desc": "Standard Q&A query",          "status": "done"},
                {"name": "Retrieval",    "desc": "Searching your documents",     "status": "running"},
                {"name": "Generator",    "desc": "Composing grounded answer",    "status": "pending"},
            ]
        else:
            steps = [
                {"name": "Intent Check", "desc": "Standard Q&A query",          "status": "done"},
                {"name": "Retrieval",    "desc": "Searching your documents",     "status": "running"},
                {"name": "Web Search",   "desc": "Fetching current web sources", "status": "pending"},
                {"name": "Trust Filter", "desc": "Applying domain policy",       "status": "pending"},
                {"name": "Generator",    "desc": "Composing grounded answer",    "status": "pending"},
            ]

        render_workflow_trace(trace_placeholder, steps)
        time.sleep(0.5)

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
                "is_greeting":     greeting_intent,
            })

            source_type = result.get("source_type", "rag")

            # Update trace to final state
            if greeting_intent:
                steps[1]["status"] = "done"
            elif summary_intent:
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
                time.sleep(0.35)
                steps[generator_idx]["status"] = "done"

            render_workflow_trace(trace_placeholder, steps)
            time.sleep(0.35)

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
                "content": f"Something went wrong: {str(e)}",
            })

        trace_placeholder.empty()
        st.rerun()


if __name__ == "__main__":
    main()