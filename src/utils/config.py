import os

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3-70b-8192")

SUPPORTED_MODELS = [
    DEFAULT_MODEL,
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]
