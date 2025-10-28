# src/app/app.py

import streamlit as st
import json
import os
# Use absolute import so running as a module works consistently
from src.app import rag_core

# Ensure page config is the first Streamlit call
st.set_page_config(layout="wide")

# --- Configuration & Setup ---
PERSONA_FILE_PATH = os.path.join(os.path.dirname(__file__), "persona.json") # More robust path

@st.cache_data(show_spinner=False)
def load_personas() -> dict:
    """Loads professor persona data from the JSON file."""
    if not os.path.exists(PERSONA_FILE_PATH):
        st.error(f"Persona file not found at {PERSONA_FILE_PATH}")
        return {}
    try:
        with open(PERSONA_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading persona file: {e}")
        return {}

PERSONAS = load_personas()
AVAILABLE_CLASSES = list(PERSONAS.keys())

# --- Streamlit App UI ---
st.title("🎓 Professor AI Tutor")

if not AVAILABLE_CLASSES:
    st.warning("No personas loaded. Please check the persona.json file.")
else:
    # --- Sidebar for Class Selection ---
    with st.sidebar:
        st.header("Select Course")
        selected_class = st.selectbox(
            "Which course agent would you like to chat with?",
            options=AVAILABLE_CLASSES,
            index=0,
            key="class_selector" # Add key for potential state management
        )
        selected_persona = PERSONAS.get(selected_class, {})
        st.subheader(f"Professor: {selected_persona.get('professor_name', 'N/A')}")
        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun() # Rerun to clear display

    # --- Initialize Session State & Clear on Class Change ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Track previously selected class to clear history on change
    if "prev_selected_class" not in st.session_state:
        st.session_state.prev_selected_class = selected_class
    elif st.session_state.prev_selected_class != selected_class:
        # Auto-clear chat history when switching classes to avoid cross-class context leakage
        st.session_state.messages = []
        st.session_state.prev_selected_class = selected_class
        st.rerun()

    # --- Display Previous Chat Messages ---
    st.header(f"Chat with the {selected_class} Agent")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Handle New User Input ---
    # Use st.chat_input for better chat UI
    if user_question := st.chat_input(f"Ask about {selected_class}..."):
        # Add user message to session state and display it
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # --- Get AI Response using RAG Core ---
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # Placeholder for streaming/waiting
            message_placeholder.markdown("Thinking...")

            try:
                # <<< Pass chat history to the RAG core >>>
                # Note: This is now a simple wrapper for your RAG core, which handles all retrieval and LLM calls.
                answer = rag_core.get_rag_response(
                    question=user_question,
                    class_name=selected_class,
                    persona=selected_persona,
                    chat_history=st.session_state.messages[:-1] # Pass history *before* current question
                )
                message_placeholder.markdown(answer)

                # Add AI response to session state
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                # Surface errors directly in the chat flow as the assistant
                error_summary = "⚠️ Sorry, an error occurred while generating a response."
                error_details = f"Details: {e}"
                chat_error_message = f"{error_summary}\n\n{error_details}"
                message_placeholder.markdown(chat_error_message)
                st.session_state.messages.append({"role": "assistant", "content": chat_error_message})