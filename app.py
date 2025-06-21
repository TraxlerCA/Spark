# app.py
#
# Streamlit interface for your local RAG-enabled assistant.
# Keeps the CLI’s score filtering, shows a concise “sources” footer,
# and avoids the duplicate-bubble issue by streaming in a plain placeholder.

# --------------------------------------------------------------------------- #
# imports
# --------------------------------------------------------------------------- #
from __future__ import annotations
import json
import logging
import sys
from datetime import datetime
from typing import Any, List
from core import process_query

import streamlit as st
import chromadb

from config import settings
from history_utils import (
    format_prompt, ROLE_USER, ROLE_ASSISTANT, ChatMessage,
    DEFAULT_SYSTEM_PROMPT
)

from llm_client import stream_llm_response
from main import ABS_MIN_SCORE, REL_WINDOW, extract_sources  # Keep these for now

# --------------------------------------------------------------------------- #
# optional RAG integration
# --------------------------------------------------------------------------- #
try:
    from rag.retriever import retrieve, format_context
    RAG_AVAILABLE = True
except (ImportError, FileNotFoundError):
    RAG_AVAILABLE = False

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
SESSION_KEY_HISTORY = "history"
SESSION_KEY_MODEL = "model"
SESSION_KEY_SYSTEM_PROMPT = "system_prompt"
SESSION_KEY_USE_RAG = "use_rag"
MODEL_OPTIONS: list[str] = ["gemma3:4b", "deepseek-r1:latest"]

# --------------------------------------------------------------------------- #
# logger setup
# --------------------------------------------------------------------------- #
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
@st.cache_resource
def check_vector_store_exists() -> bool:
    if not settings.persist_dir.exists():
        return False
    try:
        client = chromadb.PersistentClient(path=str(settings.persist_dir))
        collection = client.get_collection(settings.collection)
        return collection.count() > 0
    except Exception:
        return False

def init_session_state() -> None:
    if SESSION_KEY_HISTORY not in st.session_state:
        st.session_state[SESSION_KEY_HISTORY] = []
    if SESSION_KEY_MODEL not in st.session_state:
        st.session_state[SESSION_KEY_MODEL] = settings.ollama_model
    if SESSION_KEY_SYSTEM_PROMPT not in st.session_state:
        st.session_state[SESSION_KEY_SYSTEM_PROMPT] = DEFAULT_SYSTEM_PROMPT
    if SESSION_KEY_USE_RAG not in st.session_state:
        st.session_state[SESSION_KEY_USE_RAG] = RAG_AVAILABLE

# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    with st.sidebar:
        st.title("Settings")
        if RAG_AVAILABLE:
            st.checkbox("⚡ use RAG context", key=SESSION_KEY_USE_RAG)
        else:
            st.warning("RAG not available. Ingest data first.", icon="⚠️")

        st.selectbox("choose model", MODEL_OPTIONS, key=SESSION_KEY_MODEL)
        st.text_area("system prompt", key=SESSION_KEY_SYSTEM_PROMPT, height=100)

        if st.button("🔄 reset chat"):
            st.session_state[SESSION_KEY_HISTORY] = []
            st.rerun()

        payload = json.dumps(st.session_state[SESSION_KEY_HISTORY], indent=2)
        st.download_button("💾 download chat", payload, file_name="chat_history.json")

# --------------------------------------------------------------------------- #
# main chat logic
# --------------------------------------------------------------------------- #
def run_chat() -> None:
    st.header("💬 Chat with your AI assistant")

    # Display history
    for turn in st.session_state[SESSION_KEY_HISTORY]:
        role = "user" if turn["role"] == ROLE_USER else "assistant"
        st.chat_message(role).markdown(turn["content"], unsafe_allow_html=True)

    # User input
    if not (user_input := st.chat_input("Ask a question...")):
        return

    content = user_input.strip()
    ts_user = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_message_content = f"{content}\n\n<sub>{ts_user}</sub>"

    st.session_state[SESSION_KEY_HISTORY].append(
        ChatMessage(role=ROLE_USER, content=user_message_content)
    )
    st.chat_message("user").markdown(user_message_content, unsafe_allow_html=True)

    # Call the core processing logic
    with st.spinner("assistant is thinking…"):
        response_generator, sources = process_query(
            user_prompt=content,
            history=st.session_state[SESSION_KEY_HISTORY][:-1],  # Exclude current prompt
            use_rag=st.session_state[SESSION_KEY_USE_RAG],
            system_prompt=st.session_state[SESSION_KEY_SYSTEM_PROMPT],
            model=st.session_state[SESSION_KEY_MODEL],
        )

    # Stream answer in a placeholder
    placeholder = st.empty()
    full_response = ""
    for chunk in response_generator:
        full_response += chunk
        placeholder.markdown(f"{full_response} ▌")

    # Add provenance footer
    if sources:
        provenance = f"\n\n<sub>📚 sources: {', '.join(sorted(sources))}</sub>"
    else:
        provenance = "\n\n<sub>📚 sources: none (model knowledge)</sub>"

    full_response_with_footer = full_response + provenance
    placeholder.markdown(full_response_with_footer, unsafe_allow_html=True)

    # Save assistant turn
    ts_assistant = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    assistant_message_content = f"{full_response}\n\n<sub>{ts_assistant}{provenance}</sub>"

    # Display the final assistant message
    placeholder.markdown(assistant_message_content, unsafe_allow_html=True)

    # Store just the string in history (so it can be JSON-serialized)
    st.session_state[SESSION_KEY_HISTORY].append(
        ChatMessage(role=ROLE_ASSISTANT, content=assistant_message_content)
    )

# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="local AI assistant", layout="wide")

    if not check_vector_store_exists():
        st.warning(
            "vector store is not initialised or empty. "
            "run `python ingest.py` first.",
            icon="⚠️",
        )
        st.stop()

    init_session_state()
    render_sidebar()
    run_chat()


if __name__ == "__main__":
    main()
