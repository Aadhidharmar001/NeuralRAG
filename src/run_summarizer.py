import os
import json
from pathlib import Path

# Load .env from repo root (langgraph-node/.env)
ROOT = Path(__file__).parent.parent
ENV_FILE = ROOT / '.env'
if ENV_FILE.exists():
    for line in ENV_FILE.read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k,v = line.split('=',1)
        k=k.strip(); v=v.strip()
        if k and v:
            os.environ.setdefault(k, v)

# Ensure GROQ_API_KEY is set
if 'GROQ_API_KEY' not in os.environ:
    print('GROQ_API_KEY not found in environment or .env')
    raise SystemExit(1)

# Call summarizer and save result
from summary_node import summarize_node
try:
    res = summarize_node({})
except Exception as e:
    res = {"answer": f"❌ Summarization failed with exception: {e}", "sources": [], "source_type": "summary", "documents": []}

OUT = Path(__file__).parent / 'summary_result.json'
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
print('Wrote', OUT)
