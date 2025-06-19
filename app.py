# app.py
#
# Description: Streamlit chat interface with history, model & system-prompt settings,
#              markdown rendering, timestamps, spinner, prompt-length indicator,
#              automatic scroll, and chat export for a local Ollama LLM.
#

# --------------------------------------------------------------------------- #
# imports
# --------------------------------------------------------------------------- #
from __future__ import annotations  # enable postponed evaluation of annotations
import logging                     # for structured logging
import json                        # to serialize logs and chat history
from typing import Any, Dict, List  # for type hints on data structures
from datetime import datetime      # for timestamping chat messages

import streamlit as st             # for building the Streamlit web app
from pydantic_settings import BaseSettings  # for environment-based configuration
from pydantic import Field, ValidationError  # for settings fields and validation

from history_utils import format_prompt, ROLE_USER, ROLE_ASSISTANT  # for formatting dialogue history
from main import stream_llm_response, OllamaSettings  # for streaming LLM responses and model settings

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
SESSION_KEY_HISTORY = "history"         # session_state key for storing chat history
SESSION_KEY_MODEL = "model"             # session_state key for the selected model
SESSION_KEY_SYSTEM_PROMPT = "system_prompt"  # session_state key for the system prompt
MODEL_OPTIONS = ["gemma3:4b", "deepseek-r1:latest"]  # available LLM model options

# --------------------------------------------------------------------------- #
# logger setup
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)     # create module-level logger
handler = logging.StreamHandler()        # log to standard output stream
formatter = logging.Formatter("%(message)s")  # simple message-only format
handler.setFormatter(formatter)          # apply formatter to handler
logger.addHandler(handler)               # attach handler to logger
logger.setLevel(logging.INFO)            # set default log level

# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
class AppSettings(BaseSettings):
    """
    load application settings from environment variables.

    Args:
        model (str): default model name to use for chat.

    Returns:
        AppSettings: populated settings instance.
    """
    model: str = Field(
        "gemma3:4b", 
        description="default model name"
    )  # default LLM model

    class Config:
        # prefix for environment vars (e.g., APP_MODEL)
        env_prefix = "APP_"
        env_file = ".env"                    # file to load environment variables from
        env_file_encoding = "utf-8"          # encoding for the env file

class ChatError(Exception):
    """exception type for errors during chat processing."""

@st.cache_data  # cache settings to avoid reloading on each rerun
def get_settings() -> AppSettings:
    """
    load application settings from environment variables with validation.

    Returns:
        AppSettings: validated settings object.

    Raises:
        ChatError: if settings validation fails.
    """
    try:
        return AppSettings()
    except ValidationError as e:
        # log invalid settings error details
        logger.error(json.dumps({
            "event": "invalid_settings", 
            "errors": e.errors()
        }))
        raise ChatError("invalid application settings") from e

# --------------------------------------------------------------------------- #
# session-state initialization
# --------------------------------------------------------------------------- #
def init_session_state() -> None:
    """
    initialize session state keys for history, model, and system prompt.

    Ensures:
        - history list exists
        - model defaults to settings.model
        - system_prompt has a default instructional message
    """
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
    """
    render the application sidebar with settings controls.

    Components:
      - model selector for choosing the LLM
      - text area for editing the system prompt
      - button to reset chat history
      - download button to export chat as JSON
    """
    with st.sidebar:
        st.title("Settings")  # sidebar title

        # model selector: choose from predefined MODEL_OPTIONS
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

        # system prompt editor
        st.subheader("System prompt")
        st.session_state[SESSION_KEY_SYSTEM_PROMPT] = st.text_area(
            "Edit system prompt:",
            value=st.session_state[SESSION_KEY_SYSTEM_PROMPT],
            height=100,
        )

        # reset chat button: clear history and log event
        if st.button("🔄 Reset chat"):
            st.session_state[SESSION_KEY_HISTORY] = []
            logger.info(json.dumps({"event": "chat_reset"}))

        # export chat: prepare JSON payload for download
        payload = json.dumps(
            st.session_state[SESSION_KEY_HISTORY], 
            indent=2
        )
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
    """
    sanitize user input by trimming leading and trailing whitespace.

    Args:
        user_text (str): raw text input from the user.

    Returns:
        str: sanitized text.
    """
    return user_text.strip()

# --------------------------------------------------------------------------- #
# main chat logic
# --------------------------------------------------------------------------- #
def run_chat() -> None:
    """
    render chat interface, process user input, stream assistant responses, and update session history.

    Workflow:
      1. display existing chat history with timestamps
      2. receive new user message
      3. append user message to history
      4. build LLM prompt including custom system prompt
      5. show prompt length indicator
      6. configure and invoke Ollama LLM for response
      7. stream response chunks live to the UI
      8. append assistant response and rerun for scrolling
    """
    st.header("💬 Chat with Spark (local)")  # main header

    # display previous chat turns
    for turn in st.session_state[SESSION_KEY_HISTORY]:
        role = "user" if turn["role"] == ROLE_USER else "assistant"
        content = turn["content"]
        ts = turn.get("timestamp")
        if ts:
            formatted = f"{content}\n\n<sub>{ts}</sub>"
        else:
            formatted = content
        st.chat_message(role).markdown(formatted, unsafe_allow_html=True)

    # receive user input
    user_input = st.chat_input("You:")
    if not user_input:
        return

    # sanitize and record user message with timestamp
    content = sanitize_input(user_input)
    ts_user = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state[SESSION_KEY_HISTORY].append({
        "role": ROLE_USER,
        "content": content,
        "timestamp": ts_user,
    })
    st.chat_message("user").markdown(
        f"{content}\n\n<sub>{ts_user}</sub>",
        unsafe_allow_html=True,
    )

    # construct prompt for LLM using history and system prompt
    prompt = format_prompt(
        history=st.session_state[SESSION_KEY_HISTORY],
        next_user_message=content,
        system_prompt=st.session_state[SESSION_KEY_SYSTEM_PROMPT],
    )

    # show prompt-length indicator for user feedback
    st.caption(f"Prompt length: {len(prompt)} characters")

    # configure Ollama settings with selected model
    settings = OllamaSettings()
    settings.model = st.session_state[SESSION_KEY_MODEL]

    # stream assistant response with live updates
    with st.chat_message("assistant"):
        placeholder = st.empty()  # placeholder for streaming text
        response_text = ""
        try:
            with st.spinner("Spark thinks..."):
                # iterate and render each chunk from LLM
                for chunk in stream_llm_response(prompt, settings):
                    response_text += chunk
                    placeholder.markdown(response_text, unsafe_allow_html=True)
        except Exception as e:
            error_info: Dict[str, Any] = {
                "event": "stream_error",
                "error": str(e),
            }
            # log stream error details
            logger.error(json.dumps(error_info))
            # inform user of failure
            st.error("Error getting response, please try again later")
            raise ChatError("failed to get response") from e
        else:
            # record assistant response with timestamp
            ts_assistant = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state[SESSION_KEY_HISTORY].append({
                "role": ROLE_ASSISTANT,
                "content": response_text,
                "timestamp": ts_assistant,
            })

    # automatically rerun app to scroll chat to bottom
    st.rerun()

# --------------------------------------------------------------------------- #
# application entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """
    entry point for Streamlit app; initialize state, render sidebar, and start chat.
    """
    init_session_state()
    render_sidebar()
    run_chat()

if __name__ == "__main__":
    main()