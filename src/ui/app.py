"""Streamlit frontend entry point for the SBI Facts Assistant.

Calls the Phase 5 retrieval pipeline (``pipeline.run``) per the architecture.
Thin UI: all styling/hero/examples live in ``src.ui.components``.

Run:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import CHROMA_PERSIST_DIR, WELCOME
from src.retrieval import run_pipeline
from src.ui.components import (
    apply_groww_style,
    get_theme_palette,
    render_disclaimer,
    render_examples,
    render_hero,
    render_scheme_list,
    warn_if_store_empty,
)


CHATBOT_AVATAR = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='72' height='72'>"
    "<circle cx='36' cy='36' r='36' fill='%2304b488'/>"
    "<rect x='17' y='21' width='38' height='27' rx='10' fill='%23ffffff'/>"
    "<circle cx='28' cy='34.5' r='4' fill='%2304b488'/>"
    "<circle cx='36' cy='34.5' r='4' fill='%2304b488'/>"
    "<circle cx='44' cy='34.5' r='4' fill='%2304b488'/>"
    "<path d='M23 46 L29 40 h9 l-13 13 z' fill='%23ffffff'/>"
    "</svg>"
)


def _user_bubble_html(text: str, P: dict) -> str:
    return (
        f'<div style="display:flex; justify-content:flex-end; margin:6px 0; '
        f'font-size:13px;">'
        f'<div style="display:inline-block; max-width:75%; '
        f'background:{P["user_bubble_bg"]}; color:{P["user_bubble_text"]}; '
        f'border-radius:14px 14px 4px 14px; padding:9px 13px; '
        f'line-height:1.5; word-break:break-word;">{text}</div></div>'
    )


def _assistant_bubble_html(body: str, citation: str | None, P: dict) -> str:
    src = ""
    if citation:
        src = (
            f'<div style="margin-top:8px; font-size:10px; '
            f'color:{P["assist_bubble_text"]}; opacity:.8; '
            f'word-break:break-all;">Source: {citation}</div>'
        )
    return (
        f'<div style="display:flex; align-items:flex-start; margin:6px 0; '
        f'font-size:13px;">'
        f'<img src="{CHATBOT_AVATAR}" alt="assistant" '
        f'style="width:30px; height:30px; border-radius:50%; object-fit:cover; '
        f'margin-right:9px; flex-shrink:0;" />'
        f'<div style="display:inline-block; max-width:75%; '
        f'background:{P["assist_bubble_bg"]}; color:{P["assist_bubble_text"]}; '
        f'border-radius:4px 14px 14px 14px; padding:9px 13px; '
        f'line-height:1.55; word-break:break-word;">{body}{src}</div></div>'
    )


_SCOPE_REPLY = WELCOME

_SMALLTALK_RULES = [
    (
        re.compile(
            r"^(h+i+|hel+o+|hey+)(\s+there)?$|"
            r"^(good\s+)?(morn(ing)?|afternoon|even(ing)?)$",
            re.IGNORECASE,
        ),
        "Hi! How may I help you? Ask me about expense ratio, SIP minimum, "
        "exit load, lock-in, riskometer, NAV, AUM, fund manager, or benchmark "
        "for any of the seven SBI schemes.",
    ),
    (
        re.compile(
            r"^(thank you|thanks|thankyou|thx|thank u|tysm|thanks a lot)$",
            re.IGNORECASE,
        ),
        "You're welcome! Let me know if you have more questions.",
    ),
    (
        re.compile(r"^(bye|goodbye|good bye|byee|see you|see ya|cya)$", re.IGNORECASE),
        "Goodbye! Feel free to come back anytime.",
    ),
    (
        re.compile(
            r"^(who are you|who are u|what are you|what can you do|"
            r"what do you do|what can you tell me|what do you know|"
            r"what is this|help)$",
            re.IGNORECASE,
        ),
        _SCOPE_REPLY,
    ),
]


def check_smalltalk(query: str) -> str | None:
    """Canned reply for bare greetings/small talk, else None.

    Pure regex (no LLM, no retrieval). Only triggers when the WHOLE query is
    basically just the greeting -- never when it wraps a real question.
    """
    if not query:
        return None
    text = re.sub(r"[\s.,!?]+$", "", query.strip().lower())
    for pattern, reply in _SMALLTALK_RULES:
        if pattern.fullmatch(text):
            return reply
    return None


def _submit_chat(st_messages: list, prompt: str) -> None:
    st_messages.append({"role": "user", "content": prompt})

    # Fail fast instead of hanging on a missing index (e.g. Render deploy that
    # did not run the ingest chain in the build). Locally the store always exists.
    if not (CHROMA_PERSIST_DIR / "chroma.sqlite3").exists():
        st_messages.append(
            {
                "role": "assistant",
                "content": (
                    "The search index is missing in this deployment "
                    f"({CHROMA_PERSIST_DIR.name}/chroma not found). On Render the "
                    "build command must run the ingest chain (data loading -> chunking -> "
                    "embedding -> vector store) before starting the app."
                ),
            }
        )
        return

    reply = check_smalltalk(prompt)
    if reply is not None:
        body, citation = reply, None
    else:
        with st.spinner("Looking up Groww snapshot…"):
            result = run_pipeline(prompt)
        body, citation = result.text, result.citation

    st_messages.append({"role": "assistant", "content": body, "citation": citation})


def main() -> None:
    apply_groww_style()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    P = get_theme_palette()

    # Everything (header, dropdown, examples, chat history) lives inside ONE
    # real Streamlit container. Each element is appended to that container, so
    # the panel styling applied to the container wraps them visually as a unit.
    with st.container():
        render_hero()
        render_scheme_list()
        warn_if_store_empty()

        # Example questions: directly visible, stacked vertically (no dropdown).
        render_examples()

        # Chat history: rendered as a single self-contained HTML block inside
        # its own scrollable region (never split across markdown calls).
        history_html = "".join(
            _user_bubble_html(msg["content"], P)
            if msg["role"] == "user"
            else _assistant_bubble_html(msg["content"], msg.get("citation"), P)
            for msg in st.session_state.messages
        )
        if history_html:
            st.markdown(
                f'<div class="gw-chat-scroll">{history_html}</div>',
                unsafe_allow_html=True,
            )

        # Inline chat bar: a custom form INSIDE the panel, so there is no
        # separate viewport-anchored footer and no seam/gap to fix.
        # Enter or the green circular send button both submit the form.
        with st.form("chat_bar", clear_on_submit=True):
            text_col, send_col = st.columns([6, 1], vertical_alignment="center")
            with text_col:
                typed_text = st.text_input(
                    "Ask a scheme fact…",
                    key="chat_text",
                    placeholder="Ask a scheme fact…",
                    label_visibility="collapsed",
                )
            with send_col:
                submitted = st.form_submit_button("➤")

    # A chip click (st.button) submits its question; otherwise the form text.
    pending = st.session_state.pop("pending_question", None)
    typed_text = (typed_text or "").strip()
    if submitted and typed_text:
        effective = typed_text
    elif pending:
        effective = pending
    else:
        effective = None

    if effective:
        _submit_chat(st.session_state.messages, effective)
        st.rerun()

    render_disclaimer()


if __name__ == "__main__":
    main()
