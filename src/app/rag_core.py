# src/app/rag_core.py

import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document # For type hinting
from langchain_core.messages import HumanMessage, AIMessage # Added message types
import logging
import cohere # Import Cohere library

# --- Setup Logging ---
# Configure logging for better tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Attempt to import necessary components from refinery ---
# This assumes refinery.embedding correctly initializes and exposes 'vector_store'
embedding = None
try:
    # Try absolute import first (execution via `-m src.app.app`)
    from src.refinery import embedding as _embedding
    embedding = _embedding
except Exception as e_abs:
    logging.warning(f"RAG Core: Absolute import failed: {e_abs}. Trying relative import.")
    try:
        # Fallback to relative import (execution from project root)
        from ..refinery import embedding as _embedding_rel  # type: ignore
        embedding = _embedding_rel
    except Exception as e_rel:
        logging.error(f"RAG Core: Failed to import refinery.embedding via both paths: abs={e_abs}; rel={e_rel}")
        embedding = None

if embedding is not None:
    if not hasattr(embedding, 'vector_store') or getattr(embedding, 'vector_store', None) is None:
        logging.error("RAG Core: vector_store not initialized in embedding module.")
        embedding = None
    else:
        logging.info("RAG Core: Successfully imported vector_store from refinery.embedding.")

# --- Load Environment Variables ---
# Loads variables from .env file for local development
load_dotenv()

# --- Initialize LLM (using LangChain for consistency) ---
llm = None
try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    # Initialize the Gemini model via LangChain
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # Cost-effective model with large context
        google_api_key=gemini_api_key,
        convert_system_message_to_human=True, # Helps with certain prompt structures
        temperature=0.5 # Lower temperature for more factual, less creative responses
    )
    logging.info("RAG Core: Gemini model initialized.")
except Exception as e:
    logging.error(f"RAG Core: Error initializing Gemini model: {e}")

# --- Initialize Cohere Client (for Re-ranking) ---
co = None
try:
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    if cohere_api_key:
        co = cohere.Client(cohere_api_key)
        logging.info("RAG Core: Cohere client initialized for re-ranking.")
    else:
        # Log a warning if the key is missing, re-ranking will be skipped
        logging.warning("RAG Core: COHERE_API_KEY not found. Re-ranking will be skipped.")
except Exception as e:
    logging.error(f"RAG Core: Error initializing Cohere client: {e}")

# --- Constants ---
INITIAL_RETRIEVAL_K = 20 # Number of chunks to fetch initially from vector store
FINAL_CONTEXT_K = 7    # Number of chunks to send to LLM after re-ranking (for specific questions)

# --- Helper Function: Build Prompt (for standard RAG) ---
def _build_rag_prompt(question: str, context_docs: list[Document], persona: dict, class_name: str, chat_history: list) -> str:
    """Builds the final prompt incorporating persona, chat history, and structured context."""

    # --- Persona Injection ---
    intro = f"You are role-playing as {persona.get('professor_name', 'the professor')} for the class: {class_name}."
    style = persona.get('style_prompt', "Maintain a helpful, accurate, and professional tone appropriate for a professor.")
    profile = persona.get('profile_summary', '')
    if profile:
        intro += f" ({profile})" # Add profile summary if available

    # --- Format Chat History ---
    history_string = ""
    if chat_history:
        history_string += "PREVIOUS CONVERSATION HISTORY (for context):\n"
        # Include a limited number of recent turns to manage token count
        for msg in chat_history[-6:]: # Example: Use last 6 messages (3 turns)
            role = "Student" if msg["role"] == "user" else persona.get('professor_name', 'Professor')
            history_string += f'{role}: {msg["content"]}\n'
        history_string += "---\n"

    # --- Format Retrieved Context ---
    context_string = ""
    source_set = set() # To track unique sources used in the answer
    if context_docs:
        for i, doc in enumerate(context_docs):
            source_file = doc.metadata.get('source_file', 'Unknown Source')
            page_num = doc.metadata.get('page_number')
            source_info = f"Source: {source_file}" + (f" (Page {page_num})" if page_num else "")
            source_set.add(source_info) # Collect unique source information

            context_string += f"--- Context Chunk {i+1} ({source_info}) ---\n"
            context_string += doc.page_content # Assumes image descriptions are merged by refinery

            # Include relevant links if present in metadata
            links = doc.metadata.get('links')
            if links:
                context_string += "\n[Relevant Links Found in Document:]\n" + "\n".join(f"- {link}" for link in links)

            context_string += "\n\n"
    else: # Handle case where no documents were retrieved
        context_string = "No specific course materials were found to be relevant to this question.\n\n"

    # --- Core Prompt Template (Includes History and Context) ---
    template = f"""
{intro}
{style}

{history_string}
A student has asked the following current question about your {class_name} class:
"{question}"

Based *strictly* on the provided course materials below AND the previous conversation history (if any), answer the student's current question.
Synthesize information across different context chunks and conversation history if needed.
Quote or reference specific source documents or page numbers when possible (e.g., "According to Lecture5_Slides.pdf, page 3...").
If the answer cannot be determined from the provided materials and history, explicitly state that the information wasn't found. Do not make up information or use external knowledge outside of the provided context.

--- COURSE MATERIALS ---
{context_string.strip()}
---

YOUR ANSWER (as {persona.get('professor_name', 'Professor')}):
"""
    return template

# --- Re-ranking Function (Using Cohere API) ---
def _rerank_documents(query: str, documents: list[Document]) -> list[Document]:
    """Re-ranks documents using Cohere Rerank API for better relevance."""
    if not co or not documents: # Check if Cohere client exists and there are docs
        if not documents:
            logging.info("   No documents to re-rank.")
            return [] # Return empty list if no docs were passed
        if not co:
            logging.warning("   Cohere client not configured. Skipping re-ranking, using original vector similarity order.")
        return documents # Return original list if Cohere isn't set up

    logging.info(f"   Re-ranking {len(documents)} documents with Cohere...")
    try:
        doc_texts = [doc.page_content for doc in documents]
        # Call Cohere API
        rerank_results = co.rerank(
            query=query,
            documents=doc_texts,
            top_n=len(documents), # Re-rank all retrieved docs to get a fully sorted list
            model='rerank-english-v3.0' # Use appropriate Cohere model (check docs for updates/multilingual)
        )

        # Create a new list sorted according to Cohere's relevance scores
        reranked_docs = []
        for result in rerank_results.results:
            original_doc = documents[result.index]
            # Optional: Add score to metadata for debugging
            # original_doc.metadata['rerank_score'] = result.relevance_score
            reranked_docs.append(original_doc)

        logging.info("   Cohere Re-ranking complete.")
        return reranked_docs

    except Exception as e:
        logging.error(f"   Error during Cohere re-ranking: {e}. Falling back to original order.")
        return documents # Fallback to original order on API error



# --- Main RAG Function (Orchestrator) ---
def get_rag_response(question: str, class_name: str, persona: dict, chat_history: list | None = None) -> tuple[str, list[str]]:
    """
    Handles query: condenses based on history, retrieves context via vector search,
    re-ranks, and generates a response. Returns answer string and list of unique source identifiers.
    """
    # --- Initial Checks ---
    if not llm:
        error_msg = "LLM model not initialized. Check configuration and API keys."
        logging.error(f"RAG Core: ERROR - {error_msg}")
        return error_msg, []
    if not embedding or not hasattr(embedding, 'vector_store') or embedding.vector_store is None:
        error_msg = "Vector store not initialized or available. Check Supabase connection and refinery.embedding."
        logging.error(f"RAG Core: ERROR - {error_msg}")
        return error_msg, []

    chat_history = chat_history or []
    logging.info(f"RAG Core: Received query for class '{class_name}': '{question}'")
    logging.info(f"   Chat history length: {len(chat_history)}")

    # --- 1. Condense Query using Chat History ---
    condensed_question = _condense_query_with_history(question, chat_history)
    # --- Standard RAG Flow (default for all queries) ---
    final_context_docs: list[Document] = []
    sources_list: list[str] = []
    try:
        # a. Initial Retrieval (using CONDENSED question)
        initial_docs = embedding.vector_store.similarity_search(
            query=condensed_question, # Use condensed query for better semantic match
            k=INITIAL_RETRIEVAL_K,
            filter={"class_name": class_name} # CRUCIAL filter
        )
        logging.info(f"   Initial retrieval returned {len(initial_docs)} chunks.")
        if not initial_docs:
            # Provide a helpful message if nothing is found initially
            return "I couldn't find any documents related to your question in the course materials.", []

        # b. Re-ranking Step (using CONDENSED question)
        reranked_docs = _rerank_documents(condensed_question, initial_docs) # Calls Cohere implementation

        # c. Select Final Context based on re-ranking
        final_context_docs = reranked_docs[:FINAL_CONTEXT_K] # Take top N after re-ranking

        if final_context_docs:
            logging.info(f"   Selected {len(final_context_docs)} chunks for final context after re-ranking.")
            # Prepare unique sources list for this specific answer
            sources_set = set()
            for doc in final_context_docs:
                source_file = doc.metadata.get('source_file', 'N/A')
                page_num = doc.metadata.get('page_number')
                source_info = f"{source_file}" + (f", Page {page_num}" if page_num else "")
                sources_set.add(source_info)
            sources_list = sorted(list(sources_set))
        else:
            # If re-ranking removed all documents
            logging.warning("   No relevant documents remained after re-ranking.")
            return "I found some initial potential matches, but none seemed highly relevant after closer review. Could you please rephrase or ask about a different aspect?", []

    except Exception as e:
        logging.error(f"   Error during Supabase retrieval/re-ranking: {e}")
        # Provide a user-friendly error message
        return f"Sorry, I encountered an error trying to find information for your question. Please try again later.", []

    # --- Build the Final Prompt (Includes history and final context) ---
    # Pass the ORIGINAL question for the final prompt context, but condensed was used for retrieval
    final_prompt = _build_rag_prompt(question, final_context_docs, persona, class_name, chat_history)

    # --- Generate Standard Response ---
    try:
        logging.info("   Generating final answer with LLM...")
        # Simple chain definition for generation
        chain = ChatPromptTemplate.from_template("{rag_prompt_text}") | llm | StrOutputParser()
        # Invoke the chain with the constructed prompt
        final_answer = chain.invoke({"rag_prompt_text": final_prompt})
        logging.info("   LLM generation complete.")
        # Return the generated answer and the list of sources used
        return final_answer.strip(), sources_list
    except Exception as e:
        logging.error(f"   Error during LLM generation: {e}")
        # Provide error message but still return sources if available
        return f"Sorry, I encountered an error while generating the response: {e}", sources_list


# --- NEW: Query Condensing Function ---
def _condense_query_with_history(question: str, chat_history: list) -> str:
    """Uses LLM to rewrite the user's question into a standalone query based on history."""
    # Only condense if there's actual history and a configured LLM
    if not llm or not chat_history:
        return question # Return original question if no history or LLM problem

    logging.info("   Condensing query with chat history...")

    # Format history specifically for the condense prompt
    condense_history_str = ""
    for msg in chat_history[-4:]: # Use limited recent history (e.g., last 4 messages)
        role = "Human" if msg["role"] == "user" else "AI"
        condense_history_str += f'{role}: {msg["content"]}\n'

    # Prompt instructing the LLM to create a standalone question
    condense_prompt_template = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that incorporates relevant context from the history. If the follow-up question is already standalone, return it as is.

Chat History:
{condense_history_str}
Follow Up Input: {question}
Standalone question:"""

    try:
        # Define and invoke the condensing chain
        chain = ChatPromptTemplate.from_template("{prompt_text}") | llm | StrOutputParser()
        standalone_question = chain.invoke({"prompt_text": condense_prompt_template})
        # Log original and condensed for debugging
        logging.info(f"   Original query: '{question}'")
        logging.info(f"   Condensed query: '{standalone_question.strip()}'")
        return standalone_question.strip()
    except Exception as e:
        logging.error(f"   Error during query condensing: {e}. Using original question.")
        return question # Fallback to original question on error


# --- Local Test Block ---
if __name__ == "__main__":
    logging.info("\n--- Running local test for rag_core.py ---")
    # !!! --- IMPORTANT: Make sure Supabase has data for this class --- !!!
    TEST_CLASS = "Statistics" # <<< CHANGE TO A CLASS WITH DATA IN SUPABASE & persona.json
    TEST_QUESTION = "What did the professor say about the law of large numbers in the Oct 17 lecture?" # Specific
    # TEST_QUESTION = "Create a study guide covering everything so far"
    # TEST_QUESTION = "Tell me more about that." # Follow-up (needs dummy history below)

    # --- Dummy History for testing follow-up ---
    # test_chat_history = [
    #     {"role": "user", "content": "What is the law of large numbers?"},
    #     {"role": "assistant", "content": "The law of large numbers states that as you repeat an experiment many times, the average result will get closer to the expected value."}
    # ]
    test_chat_history = [] # Default to empty history

    try:
        current_dir = os.path.dirname(__file__)
        # Construct path robustly, assuming rag_core.py is inside src/app/
        persona_path = os.path.join(current_dir, "persona.json")
        if not os.path.exists(persona_path): raise FileNotFoundError(f"Persona file not found: {persona_path}")

        with open(persona_path, "r") as f: personas = json.load(f)
        test_persona = personas.get(TEST_CLASS)
        if not test_persona: raise ValueError(f"Persona key '{TEST_CLASS}' not found in {persona_path}")

        logging.info(f"Testing with Persona: {test_persona.get('professor_name')}")

        # Ensure RAG components are loaded before calling
        if llm and embedding and hasattr(embedding, 'vector_store') and embedding.vector_store is not None:
            # Pass the dummy history to the test call
            answer, sources = get_rag_response(TEST_QUESTION, TEST_CLASS, test_persona, test_chat_history) # <<< Pass history

            print("\n--- TEST RESULT ---")
            print("Answer:\n", answer)
            print("\nSources:\n", "\n".join(f"- {s}" for s in sources))
        else:
            logging.error("RAG components not initialized. Cannot run test.")

    except FileNotFoundError as e:
        logging.error(f"Error loading test data: {e}")
    except ValueError as e:
        logging.error(f"Error loading test data: {e}")
    except ImportError:
        logging.error("Error: Could not import dependencies. Ensure refinery modules are correct.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during testing: {e}", exc_info=True)