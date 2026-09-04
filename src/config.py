"""Closed corpus: seven Groww SBI Direct Growth scheme pages only."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_PATH = DATA_DIR / "corpus" / "schemes.json"

# Phase 3 — embeddings
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
VECTORS_PATH = EMBEDDINGS_DIR / "vectors.npy"
CHUNK_IDS_PATH = EMBEDDINGS_DIR / "chunk_ids.json"

# Phase 4 — ChromaDB persistent vector store
VECTOR_DB_DIR = DATA_DIR / "vectordb"
CHROMA_PERSIST_DIR = VECTOR_DB_DIR / "chroma"
COLLECTION_NAME = "sbi_schemes"

# Legacy path (existing ingest.py) — kept for backwards compatibility
CHROMA_DIR = ROOT / "chroma_db"
LEGACY_COLLECTION_NAME = "sbi_groww_schemes"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Bundled local copy of the embedding model (repo-local, ~87MB) so the app
# never downloads from HuggingFace at runtime (Render free blocks/slows it).
MODELS_DIR = DATA_DIR / "models"
LOCAL_MODEL_DIR = MODELS_DIR / "all-MiniLM-L6-v2"
MISTRAL_MODEL = "mistral-small-latest"
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_OVERLAP = 40
TOP_K = 6

# Groww brand color tokens (exact hex, see DOCS/architecture + UI spec)
GROWW_GREEN = "#04b488"          # primary accent / buttons / links
GROWW_GREEN_HOVER = "#00a87d"    # accent hover
GROWW_TEXT = "#44475b"           # primary text
GROWW_MUTED = "#7c7e8c"          # secondary / muted text
GROWW_TERTIARY = "#a1a3ad"       # tertiary text (sources, small captions)
GROWW_BORDER = "#e9e9eb"         # borders
GROWW_BG = "#ffffff"             # page background
GROWW_CARD = "#ffffff"           # white card surface
GROWW_SECTION = "#f8f8f8"        # card / section background (muted)
GROWW_SUBTLE = "#e9faf3"         # subtle accent bg (banners, active chips)
GROWW_NEGATIVE = "#ed5533"       # negative / error
GROWW_NEGATIVE_SUBTLE = "#fae9e5"   # negative subtle background
GROWW_WARNING = "#ffb61b"        # warning
GROWW_WARNING_SUBTLE = "#fff5e0"    # warning subtle background
# Backwards-compatible aliases (older names still referenced by tests/docs)
GROWW_DARK = GROWW_TEXT

DISCLAIMER = (
    "Facts-only, not investment advice. Verify on Groww before acting. "
    "Do not enter PAN, Aadhaar, account numbers, OTPs, email, or phone number."
)

WELCOME = (
    "Ask factual questions about seven SBI Direct Growth schemes listed on Groww. "
    "I cite one public Groww page per answer."
)

EXAMPLE_QUESTIONS = [
    "What is the expense ratio of SBI Small Cap Fund Direct Growth?",
    "What is the ELSS lock-in for SBI ELSS Tax Saver Fund?",
    "What is the minimum SIP for SBI Gold Fund Direct Growth?",
]
