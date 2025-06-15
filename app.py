# app.py
#
# Description: Streamlit chat interface with history, model & system-prompt settings,
#              markdown rendering, timestamps, spinner, prompt‐length indicator,
#              automatic scroll, and chat export for a local Ollama LLM.
#

from __future__ import annotations
import logging
import json
from typing import Any, Dict, List
from datetime import datetime

import streamlit as st
from pydantic_settings import BaseSettings
from pydantic import Field, ValidationError

from history_utils import format_prompt, ROLE_USER, ROLE_ASSISTANT
from main import stream_llm_response, OllamaSettings

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
SESSION_KEY_HISTORY = "history"
SESSION_KEY_MODEL = "model"
SESSION_KEY_SYSTEM_PROMPT = "system_prompt"
MODEL_OPTIONS = ["gemma3:4b", "deepseek-r1:latest"]

# --------------------------------------------------------------------------- #
# logger setup
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
class AppSettings(BaseSettings):
    model: str = Field("gemma3:4b", description="default model name")

    class Config:
        env_prefix = "APP_"
        env_file = ".env"
        env_file_encoding = "utf-8"

class ChatError(Exception):
    """Error during chat processing."""

@st.cache_data
def get_settings() -> AppSettings:
    """Load application settings from environment variables."""
    try:
        return AppSettings()
    except ValidationError as e:
        logger.error(json.dumps({"event": "invalid_settings", "errors": e.errors()}))
        raise ChatError("invalid application settings") from e

# --------------------------------------------------------------------------- #
# session‐state initialization
# --------------------------------------------------------------------------- #
def init_session_state() -> None:
    """Initialize session state for history, model, and system prompt."""
    if SESSION_KEY_HISTORY not in st.session_state:
        st.session_state[SESSION_KEY_HISTORY] = []  # type: List[Dict[str, Any]]
    if SESSION_KEY_MODEL not in st.session_state:
        st.session_state[SESSION_KEY_MODEL] = get_settings().model
    if SESSION_KEY_SYSTEM_PROMPT not in st.session_state:
        st.session_state[SESSION_KEY_SYSTEM_PROMPT] = (
            "You are a helpful AI assistant. Use the conversation history to answer user questions."
        )

# --------------------------------------------------------------------------- #
# sidebar UI
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    """Render sidebar for model selection, system‐prompt editing, reset & export."""
    with st.sidebar:
        st.title("Settings")

        # Model selector
        idx = (
            MODEL_OPTIONS.index(st.session_state[SESSION_KEY_MODEL])
            if st.session_state[SESSION_KEY_MODEL] in MODEL_OPTIONS
            else 0
        )
        st.selectbox(
            "Choose model",
            options=MODEL_OPTIONS,
            index=idx,
            key=SESSION_KEY_MODEL,
        )

        # System‐prompt editor
        st.subheader("System prompt")
        st.session_state[SESSION_KEY_SYSTEM_PROMPT] = st.text_area(
            "Edit system prompt:",
            value=st.session_state[SESSION_KEY_SYSTEM_PROMPT],
            height=100,
        )

        # Reset chat button
        if st.button("🔄 Reset chat"):
            st.session_state[SESSION_KEY_HISTORY] = []
            logger.info(json.dumps({"event": "chat_reset"}))

        # Export chat
        payload = json.dumps(st.session_state[SESSION_KEY_HISTORY], indent=2)
        st.download_button(
            "💾 Download chat JSON",
            data=payload,
            file_name="chat_history.json",
            mime="application/json",
        )

# --------------------------------------------------------------------------- #
# input sanitization
# --------------------------------------------------------------------------- #
def sanitize_input(user_text: str) -> str:
    """Sanitize user input by stripping whitespace."""
    return user_text.strip()

# --------------------------------------------------------------------------- #
# main chat logic
# --------------------------------------------------------------------------- #
def run_chat() -> None:
    """Render chat area, handle input, stream responses, and update history."""
    st.header("💬 Chat with Spark (local)")

    # Display history with timestamps and markdown
    for turn in st.session_state[SESSION_KEY_HISTORY]:
        role = "user" if turn["role"] == ROLE_USER else "assistant"
        content = turn["content"]
        ts = turn.get("timestamp")
        if ts:
            formatted = f"{content}\n\n<sub>{ts}</sub>"
        else:
            formatted = content
        st.chat_message(role).markdown(formatted, unsafe_allow_html=True)

    # Get user input
    user_input = st.chat_input("You:")
    if not user_input:
        return

    # Record user message
    content = sanitize_input(user_input)
    ts_user = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state[SESSION_KEY_HISTORY].append({
        "role": ROLE_USER,
        "content": content,
        "timestamp": ts_user,
    })
    st.chat_message("user").markdown(f"{content}\n\n<sub>{ts_user}</sub>", unsafe_allow_html=True)

    # Build prompt with custom system prompt
    prompt = format_prompt(
        history=st.session_state[SESSION_KEY_HISTORY],
        next_user_message=content,
        system_prompt=st.session_state[SESSION_KEY_SYSTEM_PROMPT],
    )

    # Show prompt‐length indicator
    st.caption(f"Prompt length: {len(prompt)} characters")

    # Configure Ollama settings
    settings = OllamaSettings()
    settings.model = st.session_state[SESSION_KEY_MODEL]

    # Stream assistant response with spinner and markdown updates
    with st.chat_message("assistant"):
        placeholder = st.empty()
        response_text = ""
        try:
            with st.spinner("Spark thinks..."):
                for chunk in stream_llm_response(prompt, settings):
                    response_text += chunk
                    placeholder.markdown(response_text, unsafe_allow_html=True)
        except Exception as e:
            error_info: Dict[str, Any] = {"event": "stream_error", "error": str(e)}
            logger.error(json.dumps(error_info))
            st.error("Error getting response, please try again later")
            raise ChatError("failed to get response") from e
        else:
            ts_assistant = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state[SESSION_KEY_HISTORY].append({
                "role": ROLE_ASSISTANT,
                "content": response_text,
                "timestamp": ts_assistant,
            })

    # Automatically rerun to scroll to bottom
    st.rerun()

# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    """Main entry point for Streamlit app."""
    init_session_state()
    render_sidebar()
    run_chat()

if __name__ == "__main__":
    main()
