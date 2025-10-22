# src/app/app.py

import streamlit as st
import json
import os
# Use absolute import relative to src if running as module, or adjust as needed
from . import rag_core

# --- Configuration & Setup ---
PERSONA_FILE_PATH = os.path.join(os.path.dirname(__file__), "persona.json") # More robust path

def load_personas():
    """Loads professor persona data from the JSON file."""
    if not os.path.exists(PERSONA_FILE_PATH):
        st.error(f"Persona file not found at {PERSONA_FILE_PATH}")
        return {}
    try:
        with open(PERSONA_FILE_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading persona file: {e}")
        return {}

PERSONAS = load_personas()
AVAILABLE_CLASSES = list(PERSONAS.keys())

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
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

    # --- Initialize Chat History in Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

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
                answer, sources = rag_core.get_rag_response(
                    question=user_question,
                    class_name=selected_class,
                    persona=selected_persona,
                    chat_history=st.session_state.messages[:-1] # Pass history *before* current question
                )
                message_placeholder.markdown(answer)

                # Add AI response to session state
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Display sources (optional, could be in expander below message)
                if sources:
                    with st.expander("Show sources used"):
                        for source in sources:
                            st.write(f"- {source}")

            except Exception as e:
                error_message = f"Sorry, an error occurred: {e}"
                message_placeholder.error(error_message)
                # Optionally add error to chat history too
                # st.session_state.messages.append({"role": "assistant", "content": error_message})