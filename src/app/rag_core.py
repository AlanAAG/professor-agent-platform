# src/app/rag_core.py

import os
import json
import re # Needed for parsing topic list
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document # For type hinting
from langchain_core.messages import HumanMessage, AIMessage # Added message types
import logging
from src.shared.utils import (
    cohere_rerank,
    EMBEDDING_MODEL_NAME,
    retrieve_rag_documents,
    retrieve_rag_documents_keyword_fallback,
    _to_langchain_documents,
)

try:  # pragma: no cover - defensive import
    from src.harvester.config import COURSE_MAP, LEGACY_COURSE_MAP
except Exception:  # pragma: no cover - fallback when config import fails
    COURSE_MAP = {}
    LEGACY_COURSE_MAP = {}

# --- Setup Logging ---
# Configure logging for better tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Load Environment Variables ---
# Loads variables from .env file for local development
load_dotenv()

# --- Load Professor Personas ---
PERSONA_FILE_PATH = Path(__file__).with_name("persona.json")


def _load_professor_personas() -> Dict[str, Dict[str, str]]:
    """Load professor personas from persona.json."""
    try:
        with PERSONA_FILE_PATH.open("r", encoding="utf-8") as persona_file:
            data = json.load(persona_file)
            if not isinstance(data, dict):
                logging.error("Persona file %s does not contain a dictionary. Found %s", PERSONA_FILE_PATH, type(data))
                return {}
            logging.info("Loaded %d professor personas from %s", len(data), PERSONA_FILE_PATH)
            return data
    except FileNotFoundError:
        logging.error("Persona file not found at %s", PERSONA_FILE_PATH)
    except json.JSONDecodeError as exc:
        logging.error("Failed to parse persona file %s: %s", PERSONA_FILE_PATH, exc)
    except Exception as exc:  # pragma: no cover - unexpected errors
        logging.error("Unexpected error loading personas from %s: %s", PERSONA_FILE_PATH, exc)
    return {}


PROFESSOR_PERSONAS: Dict[str, Dict[str, str]] = _load_professor_personas()
DEFAULT_PERSONA_KEY: str = (
    "Startup"
    if "Startup" in PROFESSOR_PERSONAS
    else next(iter(PROFESSOR_PERSONAS), "Startup")
)


def _get_fallback_persona() -> Dict[str, str]:
    """Return a safe default persona."""
    return PROFESSOR_PERSONAS.get(
        DEFAULT_PERSONA_KEY,
        {
            "professor_name": "Professor",
            "style_prompt": "Maintain an authoritative, actionable, and business-focused teaching style that prioritizes real-world implementation.",
        },
    )

# --- Initialize LLM (using LangChain for consistency) ---
llm = None
try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    # Initialize the Gemini model via LangChain
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Cost-effective model with large context
        google_api_key=gemini_api_key,
        convert_system_message_to_human=True, # Helps with certain prompt structures
        temperature=0.5 # Lower temperature for more factual, less creative responses
    )
    logging.info("RAG Core: Gemini model initialized.")
except Exception as e:
    logging.error(f"RAG Core: Error initializing Gemini model: {e}")

# --- Constants ---
INITIAL_RETRIEVAL_K = 20 # Number of chunks to fetch initially from vector store
RELAXED_RETRIEVAL_K = int(os.environ.get("RAG_RELAXED_MATCH_COUNT", INITIAL_RETRIEVAL_K * 2))
FINAL_CONTEXT_K = 7    # Number of chunks to send to LLM after re-ranking (for specific questions)
MAP_REDUCE_RETRIEVAL_K = 10 # Number of chunks to retrieve per topic in Map step
CLASS_HINT_SCORE_BONUS = 8  # Additional score applied when a class hint is present
REDIRECT_MIN_SCORE = 9  # Minimum raw keyword score to trigger a redirect to another course
STRICT_MATCH_THRESHOLD = float(os.environ.get("RAG_STRICT_MATCH_THRESHOLD", "0.55"))
RELAXED_MATCH_THRESHOLD = float(os.environ.get("RAG_RELAXED_MATCH_THRESHOLD", "0.35"))
KEYWORD_FALLBACK_LIMIT = max(
    FINAL_CONTEXT_K * 2,
    RELAXED_RETRIEVAL_K,
    12,
)

SUBJECT_KEYWORDS: Dict[str, List[str]] = {
    "AIML": ["ai", "artificial intelligence", "machine learning", "ml", "copilot", "azure", "power bi", "dynamics 365"],
    "Excel": ["excel", "spreadsheet", "worksheet", "pivot table", "vlookup", "xlookup", "formula", "cell reference"],
    "Statistics": ["statistics", "probability", "distribution", "variance", "hypothesis", "regression", "confidence interval", "law of large numbers"],
    "Calculus": ["calculus", "derivative", "integral", "differential", "limit", "gradient", "series"],
    "Dropshipping": ["dropshipping", "drop shipping", "e-commerce", "ecommerce", "online store", "shopify", "fulfillment", "d2c", "digital product"],
    "PublicSpeaking": ["public speaking", "speech", "presentation", "stage fright", "pitch", "storytelling", "voice"],
    "Startup": ["startup", "entrepreneur", "entrepreneurship", "innovation", "family business", "incubator", "venture", "business model", "design thinking"],
    "Networking": ["networking", "referral", "relationship marketing", "word of mouth", "leads", "connections", "business networking"],
    "OOP": ["object oriented", "o.o.p", "encapsulation", "inheritance", "polymorphism", "software design", "class diagram"],
    "MarketAnalysis": ["market analysis", "economic", "international trade", "manufacturing", "policy", "valuation of services", "export", "competitiveness", "economics", "trade policy", "service absorption", "international competitiveness", "elasticity", "price elasticity", "elasticity of demand"],
    "MarketGaps": ["market gap", "market gaps", "unmet need", "white space", "positioning", "differentiation", "competitive gap"],
    "MetaMarketing": ["meta marketing", "personalization", "pricing strategy", "promotions", "ai-driven pricing", "consumer brand", "voucher", "loyalty"],
    "CRO": ["cro", "conversion rate", "optimization", "landing page", "a/b test", "funnel", "checkout"],
    "FinanceBasics": [
        "finance",
        "valuation",
        "npv",
        "cash flow",
        "cashflow",
        "investment",
        "capital markets",
        "fundraising",
        "financial model",
        "discounted cash",
        "income statement",
        "profit and loss",
        "profit",
        "profitability",
        "profit margin",
        "profit formula",
        "formula for profit",
        "financial terminology",
        "basic financial terminology",
        "p&l",
        "balance sheet",
        "cash flow statement",
        "financial statement",
        "financial reporting",
        "general ledger",
    ],
    "HowToDecodeGlobalTrendsAndNavigateEconomicTransformations": ["global trend", "economic transformation", "monetary", "blockchain", "cryptocurrency", "systemic risk", "policy", "regulation", "macro"],
}


def _build_display_name_map() -> Dict[str, str]:
    """Create a mapping from nickname subjects to their full course names."""
    mapping: Dict[str, str] = {}
    for code, legacy_meta in LEGACY_COURSE_MAP.items():
        nickname = (legacy_meta or {}).get("name")
        if not nickname:
            continue
        course_meta = COURSE_MAP.get(code, {})
        full_name = (
            course_meta.get("full_name")
            or course_meta.get("name")
            or nickname
        )
        # Preserve the first encountered mapping to avoid accidental overwrites.
        mapping.setdefault(nickname, full_name)
    return mapping


COURSE_DISPLAY_NAME_MAP: Dict[str, str] = _build_display_name_map()


def get_course_display_name(subject: Optional[str]) -> str:
    """Return a human-readable course name for a given subject nickname."""
    if not subject:
        return ""
    return COURSE_DISPLAY_NAME_MAP.get(subject, subject)


def classify_subject(
    query: str,
    class_hint: Optional[str] = None,
    *,
    return_scores: bool = False,
) -> Union[str, Tuple[str, Dict[str, int], Dict[str, int]]]:
    """
    Analyze the user's query and return the most relevant professor persona key.
    Performs keyword scoring across predefined subject keywords and persona names.
    """
    if not query:
        if class_hint and class_hint in PROFESSOR_PERSONAS:
            return class_hint
        return DEFAULT_PERSONA_KEY

    lowered_query = query.lower()
    keyword_scores: Dict[str, int] = {}

    for subject, keywords in SUBJECT_KEYWORDS.items():
        for keyword in keywords:
            if keyword and keyword in lowered_query:
                keyword_scores[subject] = keyword_scores.get(subject, 0) + len(keyword)

    boosted_scores: Dict[str, int] = dict(keyword_scores)

    for subject in PROFESSOR_PERSONAS.keys():
        normalized_subject = subject.lower()
        if normalized_subject and normalized_subject in lowered_query:
            keyword_scores[subject] = keyword_scores.get(subject, 0) + len(normalized_subject)
            boosted_scores[subject] = boosted_scores.get(subject, 0) + len(normalized_subject)

    if class_hint and class_hint in PROFESSOR_PERSONAS:
        boosted_scores[class_hint] = boosted_scores.get(class_hint, 0) + CLASS_HINT_SCORE_BONUS

    def _finalize(subject: str) -> Union[str, Tuple[str, Dict[str, int], Dict[str, int]]]:
        if return_scores:
            return subject, boosted_scores, keyword_scores
        return subject

    if boosted_scores:
        selected_subject = max(boosted_scores, key=boosted_scores.get)
        logging.info(
            "Classified query into subject '%s' with boosted score %s (keyword score %s)",
            selected_subject,
            boosted_scores[selected_subject],
            keyword_scores.get(selected_subject, 0),
        )
        return _finalize(selected_subject)

    if class_hint and class_hint in PROFESSOR_PERSONAS:
        logging.info("No keyword match found; using provided class hint '%s'.", class_hint)
        boosted_scores[class_hint] = boosted_scores.get(class_hint, 0) + CLASS_HINT_SCORE_BONUS
        return _finalize(class_hint)

    logging.info("No keyword match found; defaulting to '%s'.", DEFAULT_PERSONA_KEY)
    return _finalize(DEFAULT_PERSONA_KEY)


def _select_active_subject(class_hint: Optional[str], classified_subject: Optional[str]) -> str:
    """
    Choose the subject whose materials must be used to answer the current question.
    Preference order:
    1. Explicit class hint (current course) when valid.
    2. Classified subject when it has a persona.
    3. Default persona key.
    4. First available persona key (if default missing).
    """
    if class_hint and class_hint in PROFESSOR_PERSONAS:
        return class_hint

    if classified_subject and classified_subject in PROFESSOR_PERSONAS:
        return classified_subject

    if DEFAULT_PERSONA_KEY in PROFESSOR_PERSONAS:
        return DEFAULT_PERSONA_KEY

    return next(iter(PROFESSOR_PERSONAS), DEFAULT_PERSONA_KEY)


def _build_redirection_message(
    *,
    active_subject: str,
    target_subject: Optional[str],
) -> Optional[str]:
    """Format the polite redirection message when no context is available."""
    if not target_subject or target_subject not in PROFESSOR_PERSONAS:
        return None

    referred_professor = PROFESSOR_PERSONAS[target_subject].get("professor_name", "the relevant professor")
    course_display_name = get_course_display_name(target_subject) or target_subject
    course_reference = f"\"{course_display_name}\"" if course_display_name != target_subject else course_display_name
    message = f"That question belongs to the {course_reference} course. You should ask {referred_professor} for that."

    if active_subject == "MarketGaps":
        return _enforce_market_gaps_voice(message)

    return message


def _maybe_redirect_for_irrelevant_query(
    *,
    active_subject: str,
    classified_subject: Optional[str],
    subject_scores: Dict[str, int],
    raw_subject_scores: Dict[str, int],
    class_hint: Optional[str],
) -> Optional[str]:
    """
    Decide whether a query should redirect to another course because no context was found
    for the active subject and another subject has a strong signal.
    """
    if not classified_subject or classified_subject == active_subject:
        return None

    if class_hint and class_hint == active_subject:
        logging.info(
            "   Retaining student in hinted class '%s' despite classification toward '%s'.",
            class_hint,
            classified_subject,
        )
        # Continue evaluating using raw scores to determine if a redirect is truly necessary.

    active_keyword_score = raw_subject_scores.get(active_subject, 0)
    alt_keyword_score = raw_subject_scores.get(classified_subject, 0)

    if alt_keyword_score < REDIRECT_MIN_SCORE:
        return None

    if active_keyword_score > 0:
        return None

    return _build_redirection_message(active_subject=active_subject, target_subject=classified_subject)

# --- Helper Functions: Build Prompts (for standard RAG) ---
def _build_system_prompt(persona: Dict[str, str]) -> str:
    """Compose the canonical system prompt section for the active persona."""
    name = persona.get("professor_name", "Professor")
    style_prompt = persona.get(
        "style_prompt",
        "Maintain an authoritative, actionable, and business-focused teaching style that prioritizes real-world implementation.",
    )
    base_rules = (
        "You are an authoritative professor and must answer all subject-related questions with 100% confidence. "
        "DO NOT use disclaimers like \"I don't have the course materials\" or \"I'll answer based on my general knowledge.\" "
        "Your primary goal is to provide actionable, strategic advice for students building real businesses, focusing on practical implementation, "
        "value-creation, and real-world impact relevant to their chosen field of study."
    )
    formatting_rules = (
        "Formatting rules: Respond in plain text sentences without markdown headings, bullet points, or emphasis. "
        "Only use special formatting when it is strictly required for equations."
    )
    brevity_rules = (
        "Brevity rules: Default to a single short paragraph of no more than five sentences unless the student explicitly asks for detailed steps, lists, or a study guide."
    )
    system_prompt = (
        f"{base_rules}\nYour name is {name}. Follow this style guide: {style_prompt}\n{formatting_rules}\n{brevity_rules}"
    )
    return system_prompt


def _format_context_section(
    context_docs: List[Document],
    chat_history: List[Dict[str, str]],
    persona_name: str,
) -> str:
    """Assemble retrieved context and recent history into a single string."""
    context_segments: List[str] = []

    if chat_history:
        history_lines = ["Previous conversation (most recent first):"]
        for message in chat_history[-6:]:
            role = "Student" if message.get("role") == "user" else persona_name or "Professor"
            history_lines.append(f"{role}: {message.get('content', '')}")
        context_segments.append("\n".join(history_lines))

    if context_docs:
        for i, doc in enumerate(context_docs, start=1):
            chunk_lines = [f"Context Chunk {i}:"]
            chunk_lines.append(doc.page_content)

            links = doc.metadata.get("links") if isinstance(doc.metadata, dict) else None
            if links:
                chunk_lines.append("Relevant Links:")
                chunk_lines.extend(f"- {link}" for link in links)

            context_segments.append("\n".join(chunk_lines))
    else:
        context_segments.append(
            "Context Retrieval Note: No indexed course excerpts were pulled for this query. Draw on your expertise and persona directives to answer decisively."
        )

    return "\n\n".join(segment.strip() for segment in context_segments if segment).strip()


def _build_rag_prompt(
    question: str,
    context_docs: List[Document],
    persona: Dict[str, str],
    subject: str,
    chat_history: List[Dict[str, str]],
) -> str:
    """Build the final prompt with system rules, retrieved context, and the student's question."""

    system_prompt = _build_system_prompt(persona)
    persona_name = persona.get("professor_name", "Professor")
    context_block = _format_context_section(context_docs, chat_history, persona_name)

    prompt = (
        f"{system_prompt}\n\n"
        f"Strict course adherence rules:\n"
        f"- Use only the materials, terminology, formulas, and methodologies from the '{subject}' course when answering.\n"
        "- If the student's question overlaps with other courses, still answer from the perspective of the current course and do not mention other courses in your response.\n"
        "- Redirect to another course only if no relevant information exists within the current course context and you are explicitly instructed to do so.\n\n"
        f"Retrieved context for subject '{subject}':\n{context_block}\n\n"
        f"User question:\n{question}\n\n"
        f"Respond as {persona_name}, delivering decisive, implementation-ready guidance grounded in the context. "
        "If the context does not contain the answer, rely on your expertise to give practical business direction without hesitation, while staying within the scope of the current course. "
        "Keep the response in plain text sentences without markdown headings, bullet lists, or emphasis unless an equation requires special formatting. "
        "Aim for no more than five concise sentences unless the student explicitly requests detailed steps, lists, or a study guide."
    )
    return prompt.strip()

# Additional helper functions
def _enforce_market_gaps_voice(answer: str) -> str:
    """Ensure MarketGaps persona responses strictly follow its stylistic rules."""
    content = (answer or "").strip()
    if not content:
        content = "Here is how we attack the market white space with precision and keep competitors on the back foot."

    # Replace direct second-person references with "Boss"
    content = re.sub(r"\byou\b", "Boss", content, flags=re.IGNORECASE)

    if "boss" not in content.lower():
        content = f"{content} Boss"

    # Ensure the response opens with multiple affirmations using "Okay"
    if not content.lower().startswith("okay"):
        content = f"Okay Boss, {content}"
    elif "boss" not in content.lower().split(",", 1)[0]:
        content = content.replace("Okay", "Okay Boss", 1)

    if content.lower().count("okay") < 2:
        content = f"Okay, {content}"

    max_words = 60
    words = content.split()
    if len(words) > max_words:
        content = " ".join(words[:max_words]).rstrip(", ")

    # Guarantee the closing question
    normalized = content.rstrip("?.! \n")
    final_response = f"{normalized}. Are you able to get it?"
    return final_response


# --- Re-ranking Function (Using zero-cost RRF/MMR utility) ---
def _rerank_documents(query: str, documents: list) -> list:
    """Re-ranks documents using shared utility for better relevance."""
    # The shared utility handles the logic and converts input/output types as needed.
    if not documents:
        logging.info("   No documents to re-rank.")
        return []
    # Note: cohere_rerank now calls the RRF/MMR logic
    reranked = cohere_rerank(query, documents)
    return reranked


def _retrieve_documents_with_backoff(
    query: str,
    subject: str,
) -> List[dict]:
    """
    Retrieve documents using progressively more permissive strategies:
    1. Strict vector similarity.
    2. Relaxed vector similarity with higher recall.
    3. Keyword fallback directly against Supabase.
    """
    attempts = (
        ("strict", INITIAL_RETRIEVAL_K, STRICT_MATCH_THRESHOLD),
        ("relaxed", RELAXED_RETRIEVAL_K, RELAXED_MATCH_THRESHOLD),
    )

    for label, match_count, threshold in attempts:
        if match_count <= 0:
            continue
        try:
            docs = retrieve_rag_documents(
                query=query,
                selected_class=subject,
                match_count=match_count,
                match_threshold=threshold,
            )
        except Exception as exc:
            logging.error(
                "   %s retrieval attempt failed for subject '%s': %s",
                label.capitalize(),
                subject,
                exc,
            )
            continue

        if docs:
            logging.info(
                "   %s vector retrieval returned %s chunks for '%s' (threshold=%.2f).",
                label.capitalize(),
                len(docs),
                subject,
                threshold,
            )
            return docs

        logging.info(
            "   %s vector retrieval returned 0 chunks for '%s' (threshold=%.2f).",
            label.capitalize(),
            subject,
            threshold,
        )

    try:
        keyword_docs = retrieve_rag_documents_keyword_fallback(
            query=query,
            selected_class=subject,
            limit=KEYWORD_FALLBACK_LIMIT,
        )
    except Exception as exc:
        logging.error(
            "   Keyword fallback retrieval failed for '%s': %s",
            subject,
            exc,
        )
        keyword_docs = []

    if keyword_docs:
        logging.info(
            "   Keyword fallback retrieval returned %s chunks for '%s'.",
            len(keyword_docs),
            subject,
        )
    else:
        logging.warning(
            "   Keyword fallback retrieval returned no results for '%s'.",
            subject,
        )
    return keyword_docs

# --- Function to Identify Topics using LLM ---
def _identify_topics_with_llm(context_text: str, subject: str) -> list[str]:
    """Uses an LLM to extract key topics or themes from general course materials."""
    logging.info("   Attempting to identify topics using LLM from general materials...")
    if not llm or not context_text:
        logging.warning("   LLM or context text missing for topic identification.")
        return []

    topic_prompt = f"""
Analyze the following collection of course materials (excerpts from lectures, readings, etc.) for the class '{subject}'.
Identify the main topics or recurring themes covered. List them clearly as a numbered list, with each topic on a new line, starting from 1. Be concise and capture the core subject of each topic.

Course Materials (excerpts):
---
{context_text}
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
def _handle_map_reduce_query(
    question: str,
    subject: str,
    persona: Dict[str, str],
    chat_history: List[Dict[str, str]],
) -> str:
    """Handle broad queries (e.g., study guides) using Map-Reduce without requiring a dedicated syllabus."""
    logging.info("   Intent: Map-Reduce Query (Study Guide / Broad Summary)")
    persona_name = persona.get("professor_name", "Professor")
    system_prompt = _build_system_prompt(persona)
    topics = []
    try:
        # --- 1. Retrieve General Course Materials for Topic Identification ---
        logging.info("   Retrieving general course materials for topic identification...")

        broad_query = f"Overall course content and key concepts for {subject}"
        # Retrieve raw dicts directly for the topic identification step
        general_docs_raw = retrieve_rag_documents(
            query=broad_query,
            selected_class=subject,
            match_count=30,
        )
        # Convert to LangChain documents for easier text processing
        general_docs = _to_langchain_documents(general_docs_raw)

        # If no general materials found, fallback to a broad summary flow
        if not general_docs:
            logging.warning("   No general course materials found. Falling back to broad summary.")
            initial_docs_raw = retrieve_rag_documents(
                query=question,  # Use original question for broad retrieval
                selected_class=subject,
                match_count=INITIAL_RETRIEVAL_K * 2,
            )
            reranked_fallback_docs_raw = _rerank_documents(question, initial_docs_raw)
            final_context_docs = _to_langchain_documents(reranked_fallback_docs_raw[:FINAL_CONTEXT_K * 2])
            
            if not final_context_docs:
                return "I couldn't find any relevant information to summarize."

            context_block = _format_context_section(final_context_docs, chat_history, persona_name)
            fallback_prompt = (
                f"{system_prompt}\n\n"
                f"Retrieved context for subject '{subject}':\n{context_block}\n\n"
                f"User question:\n{question}\n\n"
                f"Respond as {persona_name}, delivering a structured, actionable study guide that synthesizes these materials into strategic, real-world guidance. "
                "Keep the response in plain text sentences without markdown headings, bullet lists, or emphasis unless an equation requires special formatting. "
                "Aim for no more than five concise sentences unless the student explicitly requests detailed steps, lists, or a study guide."
            )
            chain = ChatPromptTemplate.from_template("{fallback_prompt_text}") | llm | StrOutputParser()
            final_summary = chain.invoke({"fallback_prompt_text": fallback_prompt})
            return final_summary.strip()

        # Combine general materials text into a single context
        context_for_topics = "\n\n".join([doc.page_content for doc in general_docs])

        # --- 2. Identify Topics using LLM from general context ---
        topics = _identify_topics_with_llm(context_for_topics, subject)

        if not topics:
            logging.warning("   LLM topic identification returned no topics. Falling back to broad summary.")
            initial_docs_raw = retrieve_rag_documents(
                query=question,
                selected_class=subject,
                match_count=INITIAL_RETRIEVAL_K * 2,
            )
            reranked_fallback_docs_raw = _rerank_documents(question, initial_docs_raw)
            final_context_docs = _to_langchain_documents(reranked_fallback_docs_raw[:FINAL_CONTEXT_K * 2])
            
            if not final_context_docs:
                return "I couldn't find any relevant information to summarize."

            context_block = _format_context_section(final_context_docs, chat_history, persona_name)
            fallback_prompt = (
                f"{system_prompt}\n\n"
                f"Retrieved context for subject '{subject}':\n{context_block}\n\n"
                f"User question:\n{question}\n\n"
                f"Respond as {persona_name}, delivering a structured, actionable study guide that synthesizes these materials into strategic, real-world guidance. "
                "Keep the response in plain text sentences without markdown headings, bullet lists, or emphasis unless an equation requires special formatting. "
                "Aim for no more than five concise sentences unless the student explicitly requests detailed steps, lists, or a study guide."
            )
            chain = ChatPromptTemplate.from_template("{fallback_prompt_text}") | llm | StrOutputParser()
            final_summary = chain.invoke({"fallback_prompt_text": fallback_prompt})
            return final_summary.strip()

    except Exception as e:
        logging.error(f"   Error retrieving materials or identifying topics: {e}")
        return f"Sorry, I encountered an error trying to structure the study guide: {e}"

    # --- 3. Map Step: Summarize each identified topic ---
    topic_summaries = []
    # Define LangChain chain for summarizing individual topics
    map_chain = ChatPromptTemplate.from_template(
        "Summarize the key points, concepts, definitions, and important examples related to the specific topic '{topic}' based *only* on the following context retrieved from {subject} course materials. "
        "Be concise yet thorough for this topic and respond using plain sentences without markdown headings, bullet lists, or emphasis.\n\n"
        "Context:\n{context}\n\n"
        "Plain summary for {topic}:"
    ) | llm | StrOutputParser()

    logging.info(f"   Starting Map step for {len(topics)} identified topics...")
    for i, topic_name in enumerate(topics):
        logging.info(f"      Mapping topic {i+1}/{len(topics)}: '{topic_name}'")
        try:
            # Retrieve chunks specifically relevant to this topic name (raw dicts)
            topic_docs_raw = retrieve_rag_documents(
                query=f"Detailed explanation, examples, formulas, and key concepts related to {topic_name} in {subject}",
                selected_class=subject,
                match_count=MAP_REDUCE_RETRIEVAL_K,
            )
            # Re-rank the retrieved docs for the topic summary for better focus
            reranked_topic_docs_raw = _rerank_documents(topic_name, topic_docs_raw)

            if not reranked_topic_docs_raw:
                logging.warning(f"      No relevant documents found for topic: '{topic_name}' after re-ranking. Skipping summary.")
                continue # Skip if no relevant docs found for this topic

            # Convert to LangChain Documents for token handling and page_content access
            reranked_topic_docs = _to_langchain_documents(reranked_topic_docs_raw)

            # Use a subset of re-ranked docs for context to manage token limits
            topic_context = "\n\n---\n\n".join([doc.page_content for doc in reranked_topic_docs[:max(5, MAP_REDUCE_RETRIEVAL_K // 2)]]) # Use top half or 5

            # Generate summary for the topic
            summary = map_chain.invoke({"topic": topic_name, "context": topic_context, "subject": subject})
            topic_summaries.append(f"{topic_name}: {summary.strip()}")
            logging.info(f"      Generated summary for topic {i+1}.")

        except Exception as e:
            logging.error(f"      Error summarizing topic '{topic_name}': {e}")
            topic_summaries.append(f"{topic_name}: [An error occurred while summarizing this topic.]")

    # --- 4. Reduce Step: Combine topic summaries ---
    if not topic_summaries:
        logging.warning("   Map step produced no valid summaries.")
        # Return message indicating failure
        return "I couldn't generate a study guide as no relevant topic summaries could be created from the materials."

    logging.info("   Starting Reduce step to combine topic summaries...")
    combined_summaries = "\n\n".join(topic_summaries) # Join summaries with double newline

    # Define prompt for the final combination step
    reduce_prompt = (
        f"{system_prompt}\n\n"
        f"Topic summaries derived from retrieved context for subject '{subject}':\n{combined_summaries}\n\n"
        f"User question:\n{question}\n\n"
        f"Respond as {persona_name}, merging the summaries into a cohesive, implementation-ready study guide. "
        "Base your answer strictly on the summaries, ensure a logical flow, and include an opening context sentence plus a decisive closing recommendation. "
        "Keep the response in plain text sentences without markdown headings, bullet lists, or emphasis unless an equation requires special formatting. "
        "Aim for no more than five concise sentences unless the student explicitly requests detailed steps, lists, or a study guide."
    )
    try:
        # Generate the final study guide
        reduce_chain = ChatPromptTemplate.from_template("{reduce_prompt_text}") | llm | StrOutputParser()
        final_answer = reduce_chain.invoke({"reduce_prompt_text": reduce_prompt})
        logging.info("   Reduce step complete.")
        # Return the final guide
        return final_answer.strip()
    except Exception as e:
        logging.error(f"   Error during Reduce step: {e}")
        # Return error message
        return f"Sorry, I could generate summaries for individual topics but failed to combine them into a final guide: {e}"


# --- Main RAG Function (Orchestrator) ---
def get_rag_response(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    class_hint: Optional[str] = None,
    persona_override: Optional[Dict[str, str]] = None,
) -> str:
    """
    Handle a user query by selecting the appropriate professor persona, assembling context,
    and generating an authoritative, business-focused answer.
    """
    if not llm:
        error_msg = "LLM model not initialized. Check configuration and API keys."
        logging.error(f"RAG Core: ERROR - {error_msg}")
        return error_msg

    chat_history = chat_history or []
    logging.info("RAG Core: Received query: '%s'", question)
    logging.info("   Chat history length: %s", len(chat_history))

    classification_result = classify_subject(question, class_hint, return_scores=True)
    if isinstance(classification_result, tuple):
        if len(classification_result) == 3:
            classified_subject, subject_scores, raw_subject_scores = classification_result
        else:
            classified_subject, subject_scores = classification_result
            raw_subject_scores = subject_scores
    else:
        classified_subject = classification_result
        subject_scores, raw_subject_scores = {}, {}
    logging.info("   Classified subject candidate: %s", classified_subject)

    active_subject = _select_active_subject(class_hint, classified_subject)

    persona = persona_override or PROFESSOR_PERSONAS.get(active_subject)
    if persona is None:
        logging.warning(
            "   Persona for subject '%s' not found; using fallback persona instructions.",
            active_subject,
        )
        persona = _get_fallback_persona()

    persona_name = persona.get("professor_name", "Professor")
    logging.info("   Active persona: %s (%s)", persona_name, active_subject)

    condensed_question = _condense_query_with_history(question, chat_history)

    map_reduce_keywords = [
        "summary",
        "overview",
        "study guide",
        "all topics",
        "whole course",
        "everything so far",
        "comprehensive review",
        "main points",
    ]
    is_map_reduce_request = any(keyword in question.lower() for keyword in map_reduce_keywords)

    if is_map_reduce_request:
        logging.info("   Routing to map-reduce flow.")
        answer = _handle_map_reduce_query(condensed_question, active_subject, persona, chat_history)
    else:
        logging.info("   Intent: Specific Question")
        final_context_docs: List[Document] = []
        try:
            initial_docs_raw = _retrieve_documents_with_backoff(
                condensed_question,
                active_subject,
            )
            logging.info(
                "   Retrieval pipeline returned %s raw chunks for '%s'.",
                len(initial_docs_raw),
                active_subject,
            )
            if not initial_docs_raw:
                redirect_message = _maybe_redirect_for_irrelevant_query(
                    active_subject=active_subject,
                    classified_subject=classified_subject,
                    subject_scores=subject_scores,
                    raw_subject_scores=raw_subject_scores,
                    class_hint=class_hint,
                )
                if redirect_message:
                    return redirect_message
                return (
                    "I couldn't find any documents related to your question in the current course materials. "
                    "Please try rephrasing or asking about a different concept within this course."
                )

            reranked_docs_raw = _rerank_documents(condensed_question, initial_docs_raw)
            final_context_docs = _to_langchain_documents(reranked_docs_raw[:FINAL_CONTEXT_K])

            if final_context_docs:
                logging.info("   Selected %s chunks for final context after re-ranking.", len(final_context_docs))
            else:
                logging.warning("   No relevant documents remained after re-ranking.")
                redirect_message = _maybe_redirect_for_irrelevant_query(
                    active_subject=active_subject,
                    classified_subject=classified_subject,
                    subject_scores=subject_scores,
                    raw_subject_scores=raw_subject_scores,
                    class_hint=class_hint,
                )
                if redirect_message:
                    return redirect_message
                return (
                    "I found some potential matches, but none were strong enough to answer that question from this course's perspective. "
                    "Could you try a different angle or rephrase the request using this course's terminology?"
                )

        except Exception as retrieval_error:
            logging.error("   Error during retrieval/re-ranking: %s", retrieval_error)
            return "Sorry, I encountered an error trying to find information for your question. Please try again later."

        final_prompt = _build_rag_prompt(question, final_context_docs, persona, active_subject, chat_history)

        try:
            logging.info("   Generating final answer with LLM...")
            chain = ChatPromptTemplate.from_template("{rag_prompt_text}") | llm | StrOutputParser()
            answer = chain.invoke({"rag_prompt_text": final_prompt}).strip()
            logging.info("   LLM generation complete.")
        except Exception as generation_error:
            logging.error("   Error during LLM generation: %s", generation_error)
            return "Sorry, I encountered an error while generating the response. Please try again later."

    if active_subject == "MarketGaps":
        answer = _enforce_market_gaps_voice(answer)

    return answer


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
    TEST_CLASS = "Statistics" 
    TEST_QUESTION = "What did the professor say about the law of large numbers?"

    # --- Dummy History for testing follow-up ---
    # test_chat_history = [
    #     {"role": "user", "content": "What is the law of large numbers?"},
    #     {"role": "assistant", "content": "The law of large numbers states that as you repeat an experiment many times, the average result will get closer to the expected value."}
    # ]
    test_chat_history = [] # Default to empty history

    try:
        if llm:
            answer = get_rag_response(
                TEST_QUESTION,
                chat_history=test_chat_history,
                class_hint=TEST_CLASS,
            )

            print("\n--- TEST RESULT ---")
            print("Answer:\n", answer)
        else:
            logging.error("RAG components not initialized. Cannot run test.")

    except Exception as e:
        logging.error(f"An unexpected error occurred during testing: {e}", exc_info=True)