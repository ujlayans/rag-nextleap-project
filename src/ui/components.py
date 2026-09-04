"""Reusable Streamlit UI elements: styling, hero banner, examples, disclaimer.

Pure styling/layout — RAG retrieval pipeline is untouched.
"""

from __future__ import annotations

import streamlit as st

from src.config import (
    CHROMA_PERSIST_DIR,
    DISCLAIMER,
    EXAMPLE_QUESTIONS,
    GROWW_GREEN_HOVER,
    RETRIEVAL_INDEX_PATH,
)

SCHEME_LIST = [
    "SBI Small Cap Fund Direct Growth",
    "SBI Mid Cap Direct Plan Growth",
    "SBI Large Cap Direct Plan Growth",
    "SBI Nifty Next 50 Index Fund Direct Growth",
    "SBI Gold Direct Plan Growth",
    "SBI ELSS Tax Saver Fund Direct Growth",
    "SBI Flexicap Fund Direct Growth",
]

FONT_IMPORT = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
"""

# GrowwSans is loaded via @font-face when reachable; otherwise we gracefully
# fall back to Inter / Plus Jakarta Sans. We intentionally prefer the safer
# Google-Fonts fallback (no external asset we don't control).
GROWWSANS_FACE = """
@font-face {
    font-family: 'GrowwSans';
    src: url('https://resources.groww.in/web-assets/font/GrowwSans-Variable.woff2') format('woff2');
    font-weight: 100 900;
    font-display: swap;
}
"""

_FONT_STACK = "'GrowwSans', 'Plus Jakarta Sans', 'Inter', 'Helvetica Neue', Arial, sans-serif"


def get_theme_palette() -> dict:
    """Single source of truth for light/dark colors (page + bubbles)."""
    if st.get_option("theme.base") == "dark":
        return {
            "page_bg": "#1a1a1a", "card_bg": "#242424", "card_border": "#3a3a3c",
            "text_primary": "#f0f0f0", "text_secondary": "#a1a3ad",
            "text_tertiary": "#8a8a8e", "accent": "#04b488",
            "accent_subtle_bg": "#0d3329", "chip_bg": "#242424",
            "chip_border": "#3a3a3c", "input_bg": "#242424", "input_border": "#3a3a3c",
            "on_accent": "#ffffff",
            "user_bubble_bg": "#2c2c2e", "user_bubble_text": "#f0f0f0",
            "assist_bubble_bg": "#04b488", "assist_bubble_text": "#ffffff",
        }
    return {
        "page_bg": "#ffffff", "card_bg": "#ffffff", "card_border": "#e9e9eb",
        "text_primary": "#121212", "text_secondary": "#7c7e8c",
        "text_tertiary": "#a1a3ad", "accent": "#04b488",
        "accent_subtle_bg": "#e9faf3", "chip_bg": "#ffffff",
        "chip_border": "#e9e9eb", "input_bg": "#ffffff", "input_border": "#e9e9eb",
        "on_accent": "#ffffff",
        "user_bubble_bg": "#e8e8ea", "user_bubble_text": "#1a1a1a",
        "assist_bubble_bg": "#04b488", "assist_bubble_text": "#ffffff",
    }


def apply_groww_style() -> None:
    """Inject Groww-styled CSS into the Streamlit app (theme-aware)."""
    st.set_page_config(page_title="SBI Facts Assistant", page_icon="◈", layout="centered")
    P = get_theme_palette()
    st.markdown(
        f"""
<style>
    {FONT_IMPORT}
    {GROWWSANS_FACE}

    :root {{
        --gw-green: {P['accent']};
        --gw-green-hover: {GROWW_GREEN_HOVER};
        --gw-text: {P['text_primary']};
        --gw-muted: {P['text_secondary']};
        --gw-tertiary: {P['text_tertiary']};
        --gw-border: {P['card_border']};
        --gw-bg: {P['page_bg']};
        --gw-card: {P['card_bg']};
        --gw-section: {P['card_bg']};
        --gw-subtle: {P['accent_subtle_bg']};
    }}

    .stApp {{
        font-family: {_FONT_STACK};
        background: {P['page_bg']};
    }}

    /* ---- Compact 540px centered panel (page container) ---- */
    .block-container {{
        max-width: 540px;
        margin: 40px auto 0;
        border: 1px solid {P['card_border']};
        border-radius: 16px;
        background: {P['card_bg']};
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        padding: 1.3rem 1.3rem 1rem;
    }}

    /* Scrollable chat history (never clipped; own overflow-y scroll) */
    .gw-chat-scroll {{
        max-height: 420px;
        overflow-y: auto;
        overflow-x: hidden;
        padding-right: 4px;
        scrollbar-width: thin;
    }}

    /* ---- Header card ---- */
    .groww-hero {{
        border-radius: 12px;
        margin-bottom: 0.75rem;
    }}
    .groww-title {{
        font-size: 20px !important;
        font-weight: 600;
        color: {P['text_primary']};
        margin: 0;
        line-height: 1.3;
        font-family: {_FONT_STACK};
    }}
    .groww-sub {{
        color: {P['text_secondary']};
        margin: 0;
        font-size: 12px;
    }}

    .groww-title-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    }}
    .gw-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #e9faf3;
        border-radius: 12px;
        padding: 3px 9px;
        font-size: 11px;
        font-weight: 500;
        color: #04b488;
        white-space: nowrap;
        flex-shrink: 0;
    }}
    .gw-badge svg {{ width: 13px; height: 13px; }}

    /* "Schemes covered" expander */
    [data-testid="stExpander"] {{
        border: 1px solid {P['card_border']};
        border-radius: 12px;
        background: {P['card_bg']};
        margin-bottom: 0.75rem;
    }}
    [data-testid="stExpander"] summary {{
        font-family: {_FONT_STACK};
        color: {P['text_primary']};
        font-size: 12px;
        font-weight: 400;
    }}
    [data-testid="stExpander"] summary span {{
        color: #44475b;
        font-size: 12px;
        font-weight: 400;
    }}
    [data-testid="stExpander"] summary svg {{ color: #a1a3ad; }}
    [data-testid="stExpander"] [data-testid="stMarkdown"] ul,
    [data-testid="stExpander"] [data-testid="stMarkdown"] ul li,
    [data-testid="stExpander"] [data-testid="stMarkdown"] p {{
        color: #44475b;
        font-size: 12px;
        font-weight: 400;
    }}

    /* ---- Footer disclaimer (plain, no box, no underline) ---- */
    .gw-foot-disclaimer {{
        display: flex;
        align-items: flex-start;
        gap: 6px;
        margin: 6px 0 2px;
        padding-left: 4px;
        font-size: 11px;
        color: {P['text_tertiary']};
        line-height: 1.4;
        border: none;
        background: transparent;
        box-shadow: none;
    }}
    .gw-foot-disclaimer svg {{
        width: 12px;
        height: 12px;
        flex-shrink: 0;
        margin-top: 2px;
    }}

    /* ---- Example question pill chips (native st.button, one row) ---- */
    [data-testid="stButton"] button {{
        display: inline-block;
        width: auto;
        max-width: 100%;
        border: 1px solid {P['card_border']};
        border-radius: 14px;
        padding: 4px 10px;
        background: {P['card_bg']};
        font-size: 11px;
        font-family: {_FONT_STACK};
        color: #7c7e8c;
        cursor: pointer;
        box-shadow: none;
        transition: border-color 0.15s ease, background 0.15s ease, color 0.15s ease;
    }}
    /* Streamlit wraps the button label in its own element that has its own
       font-size; force the chip text down to 11px at the label level. */
    [data-testid="stButton"] button p,
    [data-testid="stButton"] button span,
    [data-testid="stButton"] button div {{
        font-size: 11px !important;
        color: #7c7e8c !important;
        margin: 0 !important;
    }}
    [data-testid="stButton"] button:hover {{
        border-color: {P['accent']};
        background: {P['accent_subtle_bg']};
        color: {P['accent']};
    }}

    /* Make the "Try an example" caption compact */
    [data-testid="stCaptionContainer"] {{
        font-size: 11px !important;
        color: {P['text_secondary']} !important;
        padding: 0 0 2px !important;
        margin: 0 !important;
    }}
    [data-testid="stCaptionContainer"] p {{
        font-size: 11px !important;
        color: {P['text_secondary']} !important;
        margin: 0 !important;
    }}

    /* ---- Inline chat bar (custom form inside the panel) ---- */
    [data-testid="stForm"] {{
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
    }}
    [data-testid="stForm"] [data-testid="stHorizontalBlock"] {{
        gap: 10px;
        align-items: center;
    }}
    [data-testid="stForm"] [data-testid="stColumn"] {{
        padding: 0 !important;
    }}

    /* Text input: remove Streamlit's wrapper chrome, style as a Groww pill */
    [data-testid="stTextInput"] {{
        margin: 0 !important;
        padding: 0 !important;
    }}
    [data-testid="stTextInput"] [data-testid="stBaseInput"] {{
        background: {P['card_bg']} !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }}
    [data-testid="stTextInput"] input {{
        border: 1px solid {P['card_border']};
        border-radius: 24px;
        background: {P['card_bg']};
        padding: 8px 16px;
        min-height: 36px;
        font-family: {_FONT_STACK};
        font-size: 13px;
        color: {P['text_primary']};
        box-shadow: none !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }}
    [data-testid="stTextInput"] input::placeholder {{
        color: {P['text_tertiary']};
        opacity: 1;
    }}
    /* Green focus ring on the input (true source of focus) */
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextInput"] input:focus-visible {{
        border-color: #04b488 !important;
        box-shadow: 0 0 0 3px rgba(4, 180, 136, 0.18) !important;
        outline: none !important;
        outline-offset: 0 !important;
    }}

    /* Send button: green circle with an arrow glyph (no SVG to fight) */
    [data-testid="stFormSubmitButton"] {{
        display: flex;
        align-items: center;
    }}
    [data-testid="stFormSubmitButton"] button {{
        width: 34px;
        height: 34px;
        min-width: 34px;
        min-height: 34px;
        border-radius: 50% !important;
        background: #04b488 !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-family: {_FONT_STACK};
        font-size: 18px;
        line-height: 1;
        box-shadow: none !important;
        cursor: pointer;
        flex: 0 0 auto;
    }}
    [data-testid="stFormSubmitButton"] button:hover {{
        background: {GROWW_GREEN_HOVER} !important;
    }}
    /* Force the arrow glyph white and centered at all wrapper levels */
    [data-testid="stFormSubmitButton"] button p,
    [data-testid="stFormSubmitButton"] button span,
    [data-testid="stFormSubmitButton"] button div {{
        color: #ffffff !important;
        font-size: 18px !important;
        line-height: 1 !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    header[data-testid="stHeader"] {{ background: {P['page_bg']}; }}

    /* Mobile/small screens: full width, no floating box */
    @media (max-width: 620px) {{
        .block-container {{
            margin: 0;
            max-width: 100%;
            border: none;
            border-radius: 0;
            box-shadow: none;
            padding: 1.1rem 1rem 1rem;
        }}
    }}
</style>
""",
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render the top header (title + badge with tooltip + subtitle)."""
    st.markdown(
        f"""
<div class="groww-hero">
  <div class="groww-title-row">
    <p class="groww-title">Groww: SBI Fund Assistant</p>
    <span class="gw-badge" title="Every answer includes a source link from Groww's public pages">
      <svg viewBox='0 0 24 24' fill='none'>
        <path d='M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z' fill='#04b488'/>
        <path d='M10 14.17l-2.5-2.5L6 13.17l4 4 8-8-1.41-1.41L10 14.17z' fill='#ffffff'/>
      </svg>
      Source-verified
    </span>
  </div>
  <p class="groww-sub">Ask questions about the 7 SBI mutual fund schemes.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    """Render the small, unboxed disclaimer line below the chat input."""
    st.markdown(
        f"""
<div class="gw-foot-disclaimer">
  <svg viewBox='0 0 24 24' fill='#a1a3ad'>
    <path d='M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z'/>
  </svg>
  <span>Facts-only, not investment advice. Verify on Groww before acting. Do not enter PAN, Aadhaar, account numbers, OTPs, email, or phone number.</span>
</div>
""",
        unsafe_allow_html=True,
    )


def warn_if_store_empty() -> None:
    """Surface a warning if the runtime retrieval index hasn't been built."""
    if not RETRIEVAL_INDEX_PATH.exists():
        st.warning(
            "Search index is missing (data/retrieval/index.npz not found). "
            "From the project folder run the ingestion pipeline to build it."
        )


def render_examples() -> None:
    """Full-text example chips, smaller/lighter, with "Try an example" label."""
    st.caption("Try an example")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with cols[i]:
            if st.button(q, key=f"ex_{i}"):
                st.session_state["pending_question"] = q


def render_scheme_list() -> None:
    """Render the "Schemes covered" expander inside the header."""
    with st.expander("Schemes covered"):
        st.markdown("\n".join(f"- {name}" for name in SCHEME_LIST))
