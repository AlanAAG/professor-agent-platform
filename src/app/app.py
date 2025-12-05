# src/app/app.py

import streamlit as st
# Use absolute import so running as a module works consistently
from src.app import rag_core
from src.harvester.config import COHORTS

# Ensure page config is the first Streamlit call
st.set_page_config(layout="wide")

# --- Configuration & Setup ---
AVAILABLE_COHORTS = list(COHORTS.keys())
DEFAULT_COHORT = "2029"

# --- Streamlit App UI ---
st.title("🎓 Professor AI Tutor")

# --- Sidebar for Cohort and Class Selection ---
with st.sidebar:
    st.header("Settings")

    # Cohort Selection
    selected_cohort = st.selectbox(
        "Select Cohort:",
        options=AVAILABLE_COHORTS,
        index=AVAILABLE_COHORTS.index(DEFAULT_COHORT) if DEFAULT_COHORT in AVAILABLE_COHORTS else 0,
        key="cohort_selector"
    )

    # Load personas for the selected cohort
    PERSONAS = rag_core.get_cohort_personas(selected_cohort)
    AVAILABLE_CLASSES = list(PERSONAS.keys())

    if not AVAILABLE_CLASSES:
        st.warning(f"No personas loaded for cohort {selected_cohort}.")
    else:
        st.header("Preferred Focus")
        selected_class = st.selectbox(
            "Optionally bias the assistant toward a subject area:",
            options=AVAILABLE_CLASSES,
            index=0,
            key="class_selector" # Add key for potential state management
        )
        selected_persona = PERSONAS.get(selected_class, {})
        st.caption(f"Suggested professor: {selected_persona.get('professor_name', 'N/A')}")

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun() # Rerun to clear display

# --- Initialize Session State & Clear on Class/Cohort Change ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Track previously selected class/cohort to clear history on change
if "prev_selected_class" not in st.session_state:
    st.session_state.prev_selected_class = selected_class
if "prev_selected_cohort" not in st.session_state:
    st.session_state.prev_selected_cohort = selected_cohort

if st.session_state.prev_selected_class != selected_class or st.session_state.prev_selected_cohort != selected_cohort:
    # Auto-clear chat history when switching classes or cohorts to avoid context leakage
    st.session_state.messages = []
    st.session_state.prev_selected_class = selected_class
    st.session_state.prev_selected_cohort = selected_cohort
    st.rerun()

# --- Display Previous Chat Messages ---
st.header("Chat with Professor AI")
st.caption(f"Current focus hint: {selected_class} (Cohort: {selected_cohort})")

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
                chat_history=st.session_state.messages[:-1],  # Pass history *before* current question
                class_hint=selected_class,
                cohort_id=selected_cohort,
            )

            # Determine subject key (handling fallbacks if class isn't in PERSONAS)
            subject_key = selected_class if selected_class in PERSONAS else rag_core.DEFAULT_PERSONA_KEY
            if subject_key not in PERSONAS:
                 # Last resort if default key also missing
                 subject_key = next(iter(PERSONAS), "N/A")

            persona_display = PERSONAS.get(subject_key, {})
            persona_name = persona_display.get("professor_name", "Professor")

            message_placeholder.markdown(answer)
            message_placeholder.caption(f"Persona: {persona_name} ({subject_key})")

            # Add AI response to session state
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            # Surface errors directly in the chat flow as the assistant
            error_summary = "⚠️ Sorry, an error occurred while generating a response."
            error_details = f"Details: {e}"
            chat_error_message = f"{error_summary}\n\n{error_details}"
            message_placeholder.markdown(chat_error_message)
            st.session_state.messages.append({"role": "assistant", "content": chat_error_message})