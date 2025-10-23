# src/app/rag_core.py

import os
import json
import re # Needed for parsing topic list
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
MAP_REDUCE_RETRIEVAL_K = 10 # Number of chunks to retrieve per topic in Map step

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

# --- Function to Identify Topics using LLM ---
def _identify_topics_with_llm(syllabus_text: str, class_name: str) -> list[str]:
    """Uses an LLM to extract key topics or units from syllabus text."""
    logging.info("   Attempting to identify topics using LLM...")
    if not llm or not syllabus_text:
        logging.warning("   LLM or syllabus text missing for topic identification.")
        return []

    topic_prompt = f"""
Analyze the following syllabus or course description text for the class '{class_name}'.
Identify the main topics, units, or modules covered in the course.
List them clearly as a numbered list, with each topic on a new line, starting from 1.
Be concise and capture the core subject of each topic.

Syllabus Text:
---
{syllabus_text}
---

Main Topics (numbered list):
"""
    try:
        # Simple LangChain chain for the LLM call
        chain = ChatPromptTemplate.from_template("{prompt_text}") | llm | StrOutputParser()
        response = chain.invoke({"prompt_text": topic_prompt})

        # Parse the numbered list from the LLM response using regex
        topics = re.findall(r"^\s*\d+\.\s*(.+)", response, re.MULTILINE)
        topics = [topic.strip() for topic in topics if topic.strip()] # Clean whitespace

        if topics:
            logging.info(f"   LLM identified {len(topics)} topics: {topics}")
            return topics
        else:
            logging.warning("   LLM did not return topics in the expected numbered list format.")
            return [] # Return empty list if parsing fails
    except Exception as e:
        logging.error(f"   Error during LLM topic identification: {e}")
        return []

# --- Function for Map-Reduce ---
def _handle_map_reduce_query(question: str, class_name: str, persona: dict, chat_history: list) -> tuple[str, list[str]]:
    """Handles broad queries (e.g., study guides) using Map-Reduce."""
    logging.info("   Intent: Map-Reduce Query (Study Guide / Broad Summary)")
    topics = []
    sources_used = set() # Track all sources used across map steps
    try:
        # --- 1. Retrieve Syllabus Text ---
        logging.info("   Retrieving syllabus text for topic identification...")
        if not embedding or not hasattr(embedding, 'vector_store') or embedding.vector_store is None:
            raise RuntimeError("Vector store is not available for syllabus retrieval.")

        # Retrieve chunks likely containing syllabus/topic info
        syllabus_docs = embedding.vector_store.similarity_search(
            query="course structure syllabus overview topics schedule modules objectives",
            k=15, # Fetch a decent number of potential syllabus chunks
            filter={"class_name": class_name, "content_type": "syllabus"}
        )

        if not syllabus_docs:
            logging.warning("   No syllabus documents found in database. Cannot reliably identify topics for Map-Reduce.")
            # --- Fallback Logic: Simple Broad Summary (if no syllabus) ---
            logging.warning("   Falling back to simple broad retrieval and summary.")
            initial_docs = embedding.vector_store.similarity_search(
                query=question, # Use original question for broad retrieval
                k=INITIAL_RETRIEVAL_K * 2, # Retrieve more documents
                filter={"class_name": class_name}
            )
            # Re-rank even for fallback to get somewhat relevant context
            reranked_fallback_docs = _rerank_documents(question, initial_docs)
            final_context_docs = reranked_fallback_docs[:FINAL_CONTEXT_K * 2] # Use more context than usual

            if not final_context_docs: return "I couldn't find any relevant information to summarize.", []

            # Build context and prompt for simple summary
            context_text = "\n\n---\n\n".join([doc.page_content for doc in final_context_docs])
            fallback_prompt = f"""
                You are role-playing as {persona.get('professor_name', 'the professor')} for the class: {class_name}. {persona.get('style_prompt', '')}
                A student asked for a broad summary or study guide: "{question}"
                Based *only* on the extensive course materials provided below, generate a comprehensive summary covering the main points mentioned. Structure it clearly. If possible, organize by potential themes found in the text.

                --- COURSE MATERIALS ---
                {context_text}
                ---

                YOUR COMPREHENSIVE SUMMARY (as {persona.get('professor_name', 'Professor')}):
                """
            # Generate the fallback summary
            chain = ChatPromptTemplate.from_template("{fallback_prompt_text}") | llm | StrOutputParser()
            final_summary = chain.invoke({"fallback_prompt_text": fallback_prompt})
            # Collect sources for fallback
            for doc in final_context_docs:
                source_file = doc.metadata.get('source_file', 'N/A')
                page_num = doc.metadata.get('page_number')
                source_info = f"{source_file}" + (f", Page {page_num}" if page_num else "")
                sources_used.add(source_info)
            return final_summary.strip(), sorted(list(sources_used))
            # --- End Fallback Logic ---

        # Combine syllabus text and collect syllabus source(s)
        full_syllabus_text = "\n\n".join([doc.page_content for doc in syllabus_docs])
        for doc in syllabus_docs:
            sources_used.add(f"{doc.metadata.get('source_file', 'N/A')}") # Add syllabus file as source

        # --- 2. Identify Topics using LLM ---
        topics = _identify_topics_with_llm(full_syllabus_text, class_name)

        if not topics:
            logging.warning("   LLM topic identification failed or returned no topics. Cannot proceed with Map-Reduce.")
            # Return error message, indicating topics couldn't be found
            return "I couldn't identify the main topics from the syllabus to create a structured study guide. You could try asking for a summary of specific lectures or concepts.", sorted(list(sources_used))

    except Exception as e:
        logging.error(f"   Error retrieving syllabus or identifying topics: {e}")
        return f"Sorry, I encountered an error trying to structure the study guide: {e}", []

    # --- 3. Map Step: Summarize each identified topic ---
    topic_summaries = []
    # Define LangChain chain for summarizing individual topics
    map_chain = ChatPromptTemplate.from_template(
        "Summarize the key points, concepts, definitions, and important examples related to the specific topic '{topic}' based *only* on the following context retrieved from {class_name} course materials. Be concise yet thorough for this topic.\n\nContext:\n{context}\n\nKey Points Summary for {topic}:"
        ) | llm | StrOutputParser()

    logging.info(f"   Starting Map step for {len(topics)} identified topics...")
    for i, topic_name in enumerate(topics):
        logging.info(f"      Mapping topic {i+1}/{len(topics)}: '{topic_name}'")
        try:
            # Retrieve chunks specifically relevant to this topic name
            topic_docs = embedding.vector_store.similarity_search(
                query=f"Detailed explanation, examples, formulas, and key concepts related to {topic_name} in {class_name}", # More specific query
                k=MAP_REDUCE_RETRIEVAL_K,
                filter={"class_name": class_name} # Search all relevant content types
            )
            # Re-rank the retrieved docs for the topic summary for better focus
            reranked_topic_docs = _rerank_documents(topic_name, topic_docs)

            if not reranked_topic_docs:
                logging.warning(f"      No relevant documents found for topic: '{topic_name}' after re-ranking. Skipping summary.")
                continue # Skip if no relevant docs found for this topic

            # Use a subset of re-ranked docs for context to manage token limits
            topic_context = "\n\n---\n\n".join([doc.page_content for doc in reranked_topic_docs[:max(5, MAP_REDUCE_RETRIEVAL_K // 2)]]) # Use top half or 5

            # Generate summary for the topic
            summary = map_chain.invoke({"topic": topic_name, "context": topic_context, "class_name": class_name})
            topic_summaries.append(f"## {topic_name}\n\n{summary.strip()}") # Add topic name as heading
            logging.info(f"      Generated summary for topic {i+1}.")

            # Collect sources used in this specific map step
            for doc in reranked_topic_docs[:max(5, MAP_REDUCE_RETRIEVAL_K // 2)]: # Only list sources actually used
                source_file = doc.metadata.get('source_file', 'N/A')
                page_num = doc.metadata.get('page_number')
                source_info = f"{source_file}" + (f", Page {page_num}" if page_num else "")
                sources_used.add(source_info)

        except Exception as e:
            logging.error(f"      Error summarizing topic '{topic_name}': {e}")
            topic_summaries.append(f"## {topic_name}\n\n[An error occurred while summarizing this topic.]")

    # --- 4. Reduce Step: Combine topic summaries ---
    if not topic_summaries:
        logging.warning("   Map step produced no valid summaries.")
        # Return message indicating failure but list syllabus source if found
        return "I couldn't generate a study guide as no relevant topic summaries could be created from the materials.", sorted(list(sources_used))

    logging.info("   Starting Reduce step to combine topic summaries...")
    combined_summaries = "\n\n".join(topic_summaries) # Join summaries with double newline

    # Define prompt for the final combination step
    reduce_prompt = f"""
You are role-playing as {persona.get('professor_name', 'the professor')} for the class: {class_name}. {persona.get('style_prompt', '')}
A student asked for a study guide or summary covering the main topics ("{question}").

Combine the following individual topic summaries into a single, coherent, well-structured study guide for the {class_name} course.
Base the guide *only* on the provided summaries. Ensure a logical flow using the headings provided for each topic.
Add a brief introductory sentence setting the context of the course and a brief concluding sentence.

--- INDIVIDUAL TOPIC SUMMARIES ---
{combined_summaries}
---

YOUR FINAL STUDY GUIDE (as {persona.get('professor_name', 'Professor')}):
"""
    try:
        # Generate the final study guide
        reduce_chain = ChatPromptTemplate.from_template("{reduce_prompt_text}") | llm | StrOutputParser()
        final_answer = reduce_chain.invoke({"reduce_prompt_text": reduce_prompt})
        logging.info("   Reduce step complete.")
        # Return the final guide and all unique sources used across map steps
        return final_answer.strip(), sorted(list(sources_used))
    except Exception as e:
        logging.error(f"   Error during Reduce step: {e}")
        # Return error message but include sources identified
        return f"Sorry, I could generate summaries for individual topics but failed to combine them into a final guide: {e}", sorted(list(sources_used))


# --- Main RAG Function (Orchestrator) ---
def get_rag_response(question: str, class_name: str, persona: dict, chat_history: list | None = None) -> tuple[str, list[str]]:
    """
    Handles query, condenses based on history, detects intent, retrieves context
    (using Map-Reduce for broad queries), re-ranks (for specific queries),
    generates response. Returns answer string and list of unique source identifiers.
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

    # --- 1. Condense Query using Chat History (NEW STEP) ---
    condensed_question = _condense_query_with_history(question, chat_history)

    # --- 2. Intent Detection ---
    # Determine if the user is asking for a broad summary/study guide
    map_reduce_keywords = ["summary", "overview", "study guide", "all topics", "whole course", "everything so far", "comprehensive review", "main points"]
    # Use original question for keyword check as condensing might lose keywords
    is_map_reduce_request = any(keyword in question.lower() for keyword in map_reduce_keywords)

    # --- 3. Handle based on Intent ---
    if is_map_reduce_request:
        # --- Route to Map-Reduce Flow ---
        # Pass condensed question for relevance, but original question might be useful in prompts
        return _handle_map_reduce_query(condensed_question, class_name, persona, chat_history)
    else:
        # --- Standard RAG Flow for Specific Questions ---
        logging.info("   Intent: Specific Question")
        final_context_docs: list[Document] = []
        sources_list = []
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
                return "I found some initial potential matches, but none seemed highly relevant to your specific question after closer review. Could you please rephrase or ask about a different aspect?", []

        except Exception as e:
            logging.error(f"   Error during Supabase retrieval/re-ranking: {e}")
            # Provide a user-friendly error message
            return f"Sorry, I encountered an error trying to find information for your question. Please try again later.", []

        # --- 4. Build the Final Prompt (Includes history and final context) ---
        # Pass the ORIGINAL question for the final prompt context, but condensed was used for retrieval
        final_prompt = _build_rag_prompt(question, final_context_docs, persona, class_name, chat_history)

        # --- 5. Generate Standard Response ---
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
    # TEST_QUESTION = "Create a study guide covering everything so far" # Broad (Map-Reduce)
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