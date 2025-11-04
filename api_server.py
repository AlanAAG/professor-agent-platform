from fastapi import FastAPI, HTTPException, Depends, Security, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from src.shared.utils import (
    EMBEDDING_MODEL_NAME,
    cohere_rerank,
    retrieve_rag_documents,
    retrieve_rag_documents_keyword_fallback,
)
_genai_import_errors = []
genai = None
_GENAI_SDK_FLAVOR = "unknown"

try:
    from google import genai as _modern_genai  # type: ignore[import]
except ImportError as _modern_err:  # pragma: no cover - SDK not installed this way
    _genai_import_errors.append(_modern_err)
else:
    genai = _modern_genai
    _GENAI_SDK_FLAVOR = "google-genai"

if genai is None:
    try:
        import google.generativeai as _legacy_genai
    except ImportError as _legacy_err:  # pragma: no cover - SDK missing entirely
        _genai_import_errors.append(_legacy_err)
    else:
        genai = _legacy_genai
        _GENAI_SDK_FLAVOR = "google-generativeai"

if genai is None:  # pragma: no cover - fail fast if SDK unavailable
    raise ImportError(
        "Unable to import the Google Gemini SDK. Install either 'google-genai' or 'google-generativeai'."
    )

_GENAI_HAS_GENERATIVE_MODEL = hasattr(genai, "GenerativeModel")
_GENAI_HAS_CLIENT = hasattr(genai, "Client")
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from urllib.parse import urlparse
import logging
import time

# --- Logging Setup ---
logger = logging.getLogger("api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - api - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- Sentry Integration (Optional) ---
try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
    HAS_SENTRY = True
except ImportError:
    sentry_sdk = None
    HAS_SENTRY = False


def sanitize_sentry_event(event, hint):
    """Remove sensitive data before sending to Sentry."""
    if "request" in event and "env" in event["request"]:
        sensitive_keys = [
            "GEMINI_API_KEY",
            "SUPABASE_KEY",
            "SECRET_API_KEY",
            "OPENAI_API_KEY",
            "COACH_PASSWORD",
            "SENTRY_DSN",
        ]
        for key in sensitive_keys:
            if key in event["request"]["env"]:
                event["request"]["env"][key] = "[REDACTED]"

    if "request" in event and "headers" in event["request"]:
        if "x-api-key" in event["request"]["headers"]:
            event["request"]["headers"]["x-api-key"] = "[REDACTED]"
        if "authorization" in event["request"]["headers"]:
            event["request"]["headers"]["authorization"] = "[REDACTED]"

    if "request" in event:
        if "query_string" in event["request"]:
            event["request"]["query_string"] = "[SANITIZED]"
        if "data" in event["request"] and isinstance(event["request"]["data"], dict):
            if "api_key" in event["request"]["data"]:
                event["request"]["data"]["api_key"] = "[REDACTED]"

    return event


# Initialize Sentry if DSN is provided
SENTRY_DSN = os.getenv("SENTRY_DSN")
SENTRY_ENVIRONMENT = os.getenv("SENTRY_ENVIRONMENT", "production")
SENTRY_TRACES_SAMPLE_RATE = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))

if SENTRY_DSN and HAS_SENTRY:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        environment=SENTRY_ENVIRONMENT,
        traces_sample_rate=SENTRY_TRACES_SAMPLE_RATE,
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
        ],
        before_send=sanitize_sentry_event,
        profiles_sample_rate=0.1,
    )
    logging.info("Sentry monitoring initialized")
elif SENTRY_DSN and not HAS_SENTRY:
    logging.warning("SENTRY_DSN provided but sentry-sdk not installed. Install with: pip install sentry-sdk[fastapi]")


# --- CORS Utilities ---


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _normalize_origin(origin: str) -> str:
    return origin.strip().rstrip("/") if origin else ""


def _wildcard_origin_to_regex(origin: str) -> Optional[str]:
    normalized = _normalize_origin(origin)
    if not normalized or "*" not in normalized:
        return None
    if "://" not in normalized:
        normalized = f"https://{normalized}"
    escaped = re.escape(normalized)
    pattern = "^" + escaped.replace("\\*", ".*") + "$"
    return pattern


def _derive_lovable_variants(origins: List[str]) -> List[str]:
    variants: List[str] = []
    for origin in origins:
        if ".lovable." not in origin:
            continue
        try:
            parsed = urlparse(origin if "://" in origin else f"https://{origin}")
        except Exception:
            continue
        host = parsed.netloc or ""
        if not host:
            continue
        if "-preview" in host:
            alt_host = host.replace("-preview", "", 1)
            if alt_host and alt_host != host:
                scheme = parsed.scheme or "https"
                variants.append(f"{scheme}://{alt_host}".rstrip("/"))
    return variants


def _merge_regex_patterns(*patterns: Optional[str]) -> Optional[str]:
    cleaned: List[str] = []
    for pattern in patterns:
        if not pattern:
            continue
        candidate = pattern.strip()
        if candidate:
            cleaned.append(candidate)
    if not cleaned:
        return None
    unique = list(dict.fromkeys(cleaned))
    if len(unique) == 1:
        return unique[0]
    return "|".join(f"(?:{p})" for p in unique)


def _prepare_cors_allowances(
    raw_origins: List[str],
    regex: Optional[str],
    default_lovable_suffixes: Optional[List[str]] = None,
    allow_dynamic_lovable: bool = True,
    default_dynamic_domains: Optional[List[str]] = None,
) -> Tuple[List[str], Optional[str], bool]:
    allow_credentials = True

    if any((origin or "").strip() == "*" for origin in raw_origins):
        return ["*"], None, False

    explicit: List[str] = []
    wildcard_regexes: List[str] = []

    dynamic_domains: set[str] = set(
        domain.strip().lower().lstrip(".")
        for domain in (default_dynamic_domains or [])
        if domain and domain.strip()
    )

    for origin in raw_origins:
        normalized = _normalize_origin(origin)
        if not normalized:
            continue
        if "*" in normalized:
            wildcard_pattern = _wildcard_origin_to_regex(normalized)
            if wildcard_pattern:
                wildcard_regexes.append(wildcard_pattern)
            continue
        explicit.append(normalized)

    explicit = list(dict.fromkeys(explicit))

    derived = _derive_lovable_variants(explicit)
    for variant in derived:
        if variant not in explicit:
            explicit.append(variant)

    final_regex = _merge_regex_patterns(regex, *wildcard_regexes)

    if allow_dynamic_lovable:
        normalized_defaults = {
            suffix.strip().lower().lstrip(".")
            for suffix in (default_lovable_suffixes or [])
            if suffix and suffix.strip()
        }
    else:
        normalized_defaults = set()

    lovable_suffixes = set(normalized_defaults)
    for origin in explicit:
        try:
            parsed = urlparse(origin if "://" in origin else f"https://{origin}")
        except Exception:
            continue
        host = (parsed.netloc or "").split(":", 1)[0]
        host_lower = host.lower()
        if not host_lower:
            continue
        if "lovable." in host_lower:
            suffix = host_lower.split("lovable.", 1)[1]
            if suffix:
                lovable_suffixes.add(suffix)
        if host_lower.endswith("lovableproject.com"):
            dynamic_domains.add("lovableproject.com")

    if lovable_suffixes:
        lovable_patterns = [
            rf"^https://(?:.*\.)?lovable\.{re.escape(suffix)}(?::\d+)?$"
            for suffix in sorted(lovable_suffixes)
        ]
        final_regex = _merge_regex_patterns(final_regex, *lovable_patterns)

    if dynamic_domains:
        domain_patterns = []
        for domain in sorted(dynamic_domains):
            sanitized = domain.strip().lower()
            if not sanitized:
                continue
            if sanitized.startswith("http://") or sanitized.startswith("https://"):
                try:
                    parsed = urlparse(sanitized)
                    sanitized = (parsed.netloc or "").split(":", 1)[0]
                except Exception:
                    continue
            domain_patterns.append(
                rf"^https://(?:.*\.)?{re.escape(sanitized)}(?::\d+)?$"
            )
        if domain_patterns:
            final_regex = _merge_regex_patterns(final_regex, *domain_patterns)

    return explicit, final_regex, allow_credentials


# --- Rate Limiter Setup ---
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Performance Metrics (In-Memory) ---
REQUEST_COUNT: Dict[str, int] = {}
REQUEST_DURATION: Dict[str, List[float]] = {}
EMBEDDING_COSTS = {"total_tokens": 0, "estimated_cost_usd": 0.0}
VECTOR_DB_METRICS = {"query_count": 0, "avg_latency_ms": 0.0, "cache_hits": 0}


@app.middleware("http")
async def performance_monitoring_middleware(request: Request, call_next):
    """Track request performance and log slow endpoints."""
    start_time = time.time()
    path = request.url.path

    # Increment request count per path
    REQUEST_COUNT[path] = REQUEST_COUNT.get(path, 0) + 1

    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000

        # Track request duration per path
        if path not in REQUEST_DURATION:
            REQUEST_DURATION[path] = []
        REQUEST_DURATION[path].append(duration_ms)

        # Log slow requests (>2 seconds)
        if duration_ms > 2000:
            logging.warning(
                "SLOW_REQUEST | path=%s duration_ms=%.2f status=%s",
                path,
                duration_ms,
                response.status_code,
            )

            if SENTRY_DSN and HAS_SENTRY:
                with sentry_sdk.configure_scope() as scope:
                    scope.set_context(
                        "performance",
                        {
                            "endpoint": path,
                            "duration_ms": duration_ms,
                            "threshold_ms": 2000,
                        },
                    )
                    sentry_sdk.capture_message(
                        f"Slow endpoint: {path}",
                        level="warning",
                    )

        return response
    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000
        logging.error(
            "REQUEST_ERROR | path=%s duration_ms=%.2f error=%s",
            path,
            duration_ms,
            type(exc).__name__,
        )
        raise


# --- CORS Setup (Robust, Environment-based) ---
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
raw_allowed_origins = [origin.strip().rstrip("/") for origin in allowed_origins_env.split(",") if origin.strip()]

render_external_url = os.getenv("RENDER_EXTERNAL_URL", "").strip()
if render_external_url:
    normalized_render_url = render_external_url.rstrip("/")
    if normalized_render_url and normalized_render_url not in raw_allowed_origins:
        raw_allowed_origins.append(normalized_render_url)

allowed_origin_regex_env = os.getenv("ALLOWED_ORIGIN_REGEX", "").strip() or None

allow_dynamic_lovable_origins = _parse_bool(
    os.getenv("ALLOW_DYNAMIC_LOVABLE_ORIGINS"),
    True,
)
lovable_dynamic_suffixes: List[str] = []
if allow_dynamic_lovable_origins:
    suffixes_raw = os.getenv("LOVABLE_DYNAMIC_SUFFIXES", "app")
    lovable_dynamic_suffixes = [
        suffix.strip().lstrip(".")
        for suffix in suffixes_raw.split(",")
        if suffix and suffix.strip()
    ]

lovable_dynamic_domains: List[str] = []
if allow_dynamic_lovable_origins:
    domains_raw = os.getenv(
        "LOVABLE_DYNAMIC_DOMAINS",
        "lovable.app,lovableproject.com,lovable.dev",
    )
    lovable_dynamic_domains = [
        domain.strip().lstrip(".")
        for domain in domains_raw.split(",")
        if domain and domain.strip()
    ]

allowed_origins, allowed_origin_regex, allow_credentials = _prepare_cors_allowances(
    raw_allowed_origins,
    allowed_origin_regex_env,
    lovable_dynamic_suffixes,
    allow_dynamic_lovable_origins,
    lovable_dynamic_domains,
)

# Fallback defaults when env var not provided (Quick Test / local dev)
if not allowed_origins and not allowed_origin_regex:
    allowed_origins = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

logger.info(
    "CORS configuration | origins=%s | origin_regex=%s | credentials=%s | dynamic_lovable=%s | lovable_suffixes=%s",
    allowed_origins,
    allowed_origin_regex,
    allow_credentials,
    allow_dynamic_lovable_origins,
    {
        "suffixes": lovable_dynamic_suffixes,
        "domains": lovable_dynamic_domains,
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=allowed_origin_regex,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize clients ---
_GENAI_CLIENT = None
_GENAI_MODEL_NAME = "gemini-2.5-flash"
_api_key = os.getenv("GEMINI_API_KEY")
if _api_key:
    client_ctor = getattr(genai, "Client", None)
    if callable(client_ctor):
        try:
            _GENAI_CLIENT = client_ctor(api_key=_api_key)
        except Exception as client_err:  # pragma: no cover - SDK differences
            logger.warning("Gemini Client initialization failed via Client(): %s", client_err)
            _GENAI_CLIENT = None
    if _GENAI_CLIENT is None and hasattr(genai, "configure"):
        try:
            genai.configure(api_key=_api_key)
            logger.debug("Configured Gemini SDK via configure().")
        except Exception as cfg_err:  # pragma: no cover - configuration failure
            logger.warning("Gemini SDK configure() failed: %s", cfg_err)

logger.info(
    "Gemini SDK flavor=%s | has_generative_model=%s | has_client=%s",
    _GENAI_SDK_FLAVOR,
    _GENAI_HAS_GENERATIVE_MODEL,
    _GENAI_HAS_CLIENT,
)

# Optional Supabase client for health checks
supabase = None
try:
    from src.shared.utils import _get_supabase_client
    supabase = _get_supabase_client()
except Exception:
    supabase = None

EMBEDDING_MODEL = EMBEDDING_MODEL_NAME

# --- API Key Authentication Setup ---
api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)


def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Robust API key validation with immediate rejection of invalid requests."""
    expected = os.getenv("SECRET_API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="API key not configured")
    
    api_key = api_key.strip() if api_key else ""
    
    if len(api_key) < 8:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    if api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# --- Graceful Fallback Message ---
NO_DOCUMENTS_ANSWER = (
    "I couldn't find relevant materials. This could be because:\n"
    "- The content hasn't been uploaded yet\n"
    "- Try rephrasing your question\n"
    "- Contact alanayalag@gmail.com if this persists"
)

def _check_db_status() -> Dict[str, Optional[bool]]:
    """Perform a trivial Supabase query to confirm connectivity."""
    if supabase is None:
        return {"connected": False, "ok": False}
    try:
        _ = supabase.table("documents").select("id").limit(1).execute()
        return {"connected": True, "ok": True}
    except Exception:
        return {"connected": True, "ok": False}


def _get_rag_params() -> Dict[str, object]:
    """Read RAG tuning parameters from environment with safe defaults."""
    try:
        match_threshold = float(os.getenv("RAG_MATCH_THRESHOLD", "0.5"))
    except Exception:
        match_threshold = 0.5
    try:
        match_count = int(os.getenv("RAG_MATCH_COUNT", "8"))
    except Exception:
        match_count = 8
    relaxed_threshold = min(match_threshold, 0.3)
    relaxed_count = max(match_count, 10)
    return {
        "match_threshold": match_threshold,
        "match_count": match_count,
        "relaxed_threshold": relaxed_threshold,
        "relaxed_count": relaxed_count,
    }


def _stream_genai_response(
    contents: List[Dict[str, object]],
    system_instruction: str,
):
    """Return a streaming iterator compatible with both Gemini SDKs."""
    def _try_call(callable_obj, variants: List[Dict[str, object]]):
        last_type_error: Optional[TypeError] = None
        for kwargs in variants:
            try:
                return callable_obj(**kwargs)
            except TypeError as exc:
                last_type_error = exc
                continue
        if last_type_error:
            logger.debug(
                "Gemini streaming variant rejected for %s: %s",
                getattr(callable_obj, "__qualname__", repr(callable_obj)),
                last_type_error,
            )
        return None

    model_ctor = getattr(genai, "GenerativeModel", None)
    if callable(model_ctor):
        init_variants: List[Dict[str, object]] = []
        if _GENAI_CLIENT is not None:
            init_variants.extend(
                [
                    {
                        "model_name": _GENAI_MODEL_NAME,
                        "system_instruction": system_instruction,
                        "client": _GENAI_CLIENT,
                    },
                    {
                        "model": _GENAI_MODEL_NAME,
                        "system_instruction": system_instruction,
                        "client": _GENAI_CLIENT,
                    },
                ]
            )
        init_variants.extend(
            [
                {
                    "model_name": _GENAI_MODEL_NAME,
                    "system_instruction": system_instruction,
                },
                {
                    "model": _GENAI_MODEL_NAME,
                    "system_instruction": system_instruction,
                },
                {"model_name": _GENAI_MODEL_NAME},
                {"model": _GENAI_MODEL_NAME},
            ]
        )

        model_instance = None
        for init_kwargs in init_variants:
            try:
                model_instance = model_ctor(**init_kwargs)
                break
            except TypeError:
                continue

        if model_instance is None:
            model_instance = model_ctor(_GENAI_MODEL_NAME)

        stream_call_variants: List[Dict[str, object]] = []
        if system_instruction:
            stream_call_variants.extend(
                [
                    {
                        "contents": contents,
                        "system_instruction": system_instruction,
                    },
                    {
                        "contents": contents,
                        "config": {"system_instruction": system_instruction},
                    },
                ]
            )
        stream_call_variants.extend(
            [
                {"contents": contents},
                {"input": contents},
            ]
        )

        stream_method = getattr(model_instance, "generate_content_stream", None)
        if callable(stream_method):
            response = _try_call(stream_method, stream_call_variants)
            if response is not None:
                return response

        legacy_variants: List[Dict[str, object]] = []
        if system_instruction:
            legacy_variants.extend(
                [
                    {
                        "contents": contents,
                        "system_instruction": system_instruction,
                        "stream": True,
                    },
                    {
                        "contents": contents,
                        "config": {"system_instruction": system_instruction},
                        "stream": True,
                    },
                    {
                        "content": contents,
                        "system_instruction": system_instruction,
                        "stream": True,
                    },
                ]
            )
        legacy_variants.extend(
            [
                {"contents": contents, "stream": True},
                {"content": contents, "stream": True},
            ]
        )

        legacy_method = getattr(model_instance, "generate_content", None)
        if callable(legacy_method):
            response = _try_call(legacy_method, legacy_variants)
            if response is not None:
                return response

    client = _GENAI_CLIENT
    if client is not None:
        models_iface = getattr(client, "models", None)
        if models_iface is not None:
            models_stream_variants: List[Dict[str, object]] = []
            if system_instruction:
                models_stream_variants.extend(
                    [
                        {
                            "model": _GENAI_MODEL_NAME,
                            "contents": contents,
                            "system_instruction": system_instruction,
                        },
                        {
                            "model": _GENAI_MODEL_NAME,
                            "contents": contents,
                            "config": {"system_instruction": system_instruction},
                        },
                    ]
                )
            models_stream_variants.extend(
                [
                    {
                        "model": _GENAI_MODEL_NAME,
                        "contents": contents,
                    },
                    {
                        "model": _GENAI_MODEL_NAME,
                        "input": contents,
                    },
                ]
            )

            stream_method = getattr(models_iface, "generate_content_stream", None)
            if callable(stream_method):
                response = _try_call(stream_method, models_stream_variants)
                if response is not None:
                    return response

            generate_method = getattr(models_iface, "generate_content", None)
            if callable(generate_method):
                legacy_variants: List[Dict[str, object]] = []
                for variant in models_stream_variants:
                    legacy_variant = dict(variant)
                    legacy_variant["stream"] = True
                    legacy_variants.append(legacy_variant)
                response = _try_call(generate_method, legacy_variants)
                if response is not None:
                    return response

        responses_iface = getattr(client, "responses", None)
        if responses_iface is not None and hasattr(responses_iface, "stream_generate"):
            response_kwargs = {
                "model": _GENAI_MODEL_NAME,
                "contents": contents,
            }
            if system_instruction:
                try:
                    return responses_iface.stream_generate(
                        system_instruction=system_instruction,
                        **response_kwargs,
                    )
                except TypeError:
                    logger.debug(
                        "Gemini responses.stream_generate does not accept system_instruction directly; continuing without it."
                    )
            return responses_iface.stream_generate(**response_kwargs)

    raise RuntimeError(
        "Gemini client is not initialized with a compatible SDK. Check GEMINI_API_KEY and installed SDK version."
    )


class Message(BaseModel):
    role: str
    content: str
    sources: Optional[List[Dict[str, Any]]] = None


class ChatRequest(BaseModel):
    messages: List[Message]
    selectedClass: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("selectedClass", "class_id"),
    )
    persona: str = "balanced"

    model_config = ConfigDict(populate_by_name=True)


class RAGRequest(BaseModel):
    query: str
    selectedClass: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("selectedClass", "class_id"),
    )

    model_config = ConfigDict(populate_by_name=True)


@app.on_event("startup")
async def startup_event():
    """Track application start time for uptime metrics."""
    app.state.start_time = time.time()


@app.get("/")
@limiter.limit("100/minute")
async def health_check(request: Request):
    """Enhanced health check for monitoring."""
    db_status = _check_db_status()
    return {
        "status": "healthy",
        "service": "AI Tutor API",
        "version": "1.0.0-beta",
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
        "model_loaded": _GENAI_CLIENT is not None,
        "endpoints": {
            "chat": "/api/chat",
            "rag_search": "/api/rag-search",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
@limiter.limit("100/minute")
async def health(request: Request):
    """Dedicated health endpoint for uptime monitoring."""
    db_status = _check_db_status()
    return {"status": "ok", "database": db_status}


@app.get("/metrics")
async def get_metrics(api_key: str = Depends(get_api_key)):
    """Expose performance metrics for monitoring (protected by API key)."""
    metrics = {
        "request_counts": REQUEST_COUNT,
        "avg_response_times_ms": {
            path: (sum(durations) / len(durations)) if durations else 0
            for path, durations in REQUEST_DURATION.items()
        },
        "embedding_costs": EMBEDDING_COSTS,
        "vector_db_metrics": VECTOR_DB_METRICS,
        "uptime_seconds": time.time() - getattr(app.state, "start_time", time.time()),
    }
    return metrics


@app.post("/api/rag-search")
@limiter.limit("20/minute")
async def rag_search(request: Request, payload: RAGRequest, api_key: str = Depends(get_api_key)):
    _t0 = time.time()
    try:
        params = _get_rag_params()
        documents = retrieve_rag_documents(
            query=payload.query,
            selected_class=payload.selectedClass,
            match_count=params["match_count"],
            match_threshold=params["match_threshold"],
        )
        documents = cohere_rerank(payload.query, documents)
        
        if not documents:
            documents = retrieve_rag_documents(
                query=payload.query,
                selected_class=payload.selectedClass,
                match_count=params["relaxed_count"],
                match_threshold=params["relaxed_threshold"],
            )
            documents = cohere_rerank(payload.query, documents)
        
        if documents:
            avg_sim = sum([d.get("similarity", 0) or 0 for d in documents]) / max(len(documents), 1)
            logger.info(
                "RAG_SEARCH metrics | query_len=%s docs=%s avg_similarity=%.4f latency_ms=%s",
                len(payload.query or ""),
                len(documents),
                avg_sim,
                int((time.time() - _t0) * 1000),
            )
        
        if not documents:
            documents = retrieve_rag_documents_keyword_fallback(
                query=payload.query,
                selected_class=payload.selectedClass,
                limit=params["relaxed_count"],
            )
        
        if not documents:
            logger.warning(
                "RAG_SEARCH failure | NO_DOCUMENTS_ANSWER | query_len=%s latency_ms=%s",
                len(payload.query or ""),
                int((time.time() - _t0) * 1000),
            )
            return {
                "answer": NO_DOCUMENTS_ANSWER,
                "sources": [],
            }
        return {"documents": documents}
    except Exception as e:
        logger.exception("RAG_SEARCH error | latency_ms=%s", int((time.time() - _t0) * 1000))
        raise HTTPException(status_code=500, detail=str(e))


@app.options("/api/chat")
async def chat_preflight() -> Response:
    """Handle browser CORS preflight checks explicitly."""
    return Response(status_code=200)


@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat_stream(request: Request, payload: ChatRequest, api_key: str = Depends(get_api_key)):
    _t0 = time.time()
    try:
        user_messages = [m for m in payload.messages if (m.role or "").lower() == "user"]
        last_query = user_messages[-1].content if user_messages else ""

        params = _get_rag_params()
        documents = retrieve_rag_documents(
            query=last_query,
            selected_class=payload.selectedClass,
            match_count=params["match_count"],
            match_threshold=params["match_threshold"],
        )
        documents = cohere_rerank(last_query, documents)
        
        if not documents:
            documents = retrieve_rag_documents(
                query=last_query,
                selected_class=payload.selectedClass,
                match_count=params["relaxed_count"],
                match_threshold=params["relaxed_threshold"],
            )
            documents = cohere_rerank(last_query, documents)
        
        if not documents:
            documents = retrieve_rag_documents_keyword_fallback(
                query=last_query,
                selected_class=payload.selectedClass,
                limit=params["relaxed_count"],
            )
        
        if documents:
            avg_sim = sum([d.get("similarity", 0) or 0 for d in documents]) / max(len(documents), 1)
        else:
            avg_sim = 0.0
        logger.info(
            "CHAT metrics | query_len=%s docs=%s avg_similarity=%.4f latency_ms=%s",
            len(last_query or ""),
            len(documents or []),
            avg_sim,
            int((time.time() - _t0) * 1000),
        )
        
        context = "\n\n".join([
            f"Source {i+1}:\nClass: {doc.get('class_name', 'N/A')}\n{doc.get('content', '')}"
            for i, doc in enumerate(documents)
        ]) if documents else "No relevant course materials found."

        personas = {
            "study": "You are an encouraging and approachable study buddy who is also a professor. Help the student review material by offering clear summaries, explaining concepts, and providing practical examples. Use a supportive, friendly, and collaborative tone, like you are studying with them.",
            "professor": "You are an experienced, authoritative, and highly knowledgeable professor and the primary speaker for the course. Always answer in the first person (I/me/my), adopting the conversational, expert tone of a dedicated teacher. Crucially, be brief and highly concise.",
            "socratic": "You are a professor who employs the Socratic method. Your goal is to guide the student to the answer by asking probing, sequential questions rather than giving direct answers. Use a thoughtful, questioning, and patient tone. Only provide a direct answer if the student asks for confirmation or is significantly struggling.",
            "balanced": "You are a balanced and adaptable tutor and professor. Start by providing a concise, direct answer or explanation, and then immediately follow up with a related question to check for understanding or encourage further thought. Maintain a professional, helpful, and measured tone, combining direct instruction with light questioning."
        }
        
        if documents:
            rules = (
                "Course Material Restriction: ONLY use information derived from the provided course materials/documents. Treat these documents as the sole source of truth.\n"
                "Integration and Citation (Professor Tone): Seamlessly integrate information into your first-person response. When referencing a specific fact, definition, or concept from the material, cite the source parenthetically (e.g., **(Source: Document Name/Section)**). You are the speaker and the authority, using the materials to support your teaching.\n"
                "Tone & Accuracy: Maintain a conversational yet highly accurate and professional tone appropriate to your selected persona."
                "Format your answer using only plain paragraphs. DO NOT use markdown headings (#) or lists (*) unless absolutely necessary for clarity, and keep any such lists very short"
                "Be conversational, accurate, and most importantly, brief."
)
        else:
           rules = (
                "Context Note: No course materials were found for this query, so the answer will be based on my general knowledge and expertise.\n"
                "Answering Approach: Answer based on your extensive general knowledge and academic reasoning, maintaining the selected professor persona.\n"
                "Answer Format: Preface the answer with a brief, clear note that course materials were unavailable (e.g., 'Since I don't have the course materials for this topic, I'll answer based on my general knowledge...').\n"
                "Quality: Be concise, accurate, and highly helpful, treating the answer as a general academic explanation."
                "Format your answer using only plain paragraphs. DO NOT use markdown headings (#) or lists (*) unless absolutely necessary for clarity, and keep any such lists very short"
                "Be conversational, accurate, and most importantly, brief."
)
        system_prompt = f"""{personas.get(payload.persona, personas['balanced'])}

COURSE MATERIALS:
{context}

RULES:
{rules}"""
        
        def _to_genai_contents(chat_messages: List[Dict[str, str]]):
            contents: List[Dict[str, object]] = []
            for m in chat_messages:
                role_raw = (m.get("role") or "").lower()
                if role_raw == "system":
                    continue
                role = "user" if role_raw == "user" else "model" if role_raw == "assistant" else "user"
                contents.append({
                    "role": role,
                    "parts": [{"text": m.get("content", "")}],
                })
            return contents
        
        clean_messages: List[Dict[str, str]] = [
            {"role": m.role, "content": m.content} for m in payload.messages
        ]
        genai_contents = _to_genai_contents(clean_messages)
        
        async def generate():
            sources_payload = {
                "sources": [
                    {
                        "content": doc.get("content", ""),
                        "metadata": {
                            "class_name": doc.get("class_name"),
                            "section": doc.get("section"),
                            "title": doc.get("title"),
                            "url": doc.get("url")
                        },
                        "similarity": doc.get("similarity", 0)
                    }
                    for doc in (documents or [])
                ]
            }
            if not documents:
                logger.warning(
                    "CHAT fallback | proceeding without materials | query_len=%s latency_ms=%s",
                    len(last_query or ""),
                    int((time.time() - _t0) * 1000),
                )
            yield f"data: {json.dumps(sources_payload)}\n\n"
            
            if _GENAI_CLIENT is None and not _GENAI_HAS_GENERATIVE_MODEL:
                raise RuntimeError("Gemini client is not initialized. Check GEMINI_API_KEY.")

            try:
                response = _stream_genai_response(
                    contents=genai_contents,
                    system_instruction=system_prompt,
                )

                for chunk in response:
                    chunk_text = getattr(chunk, "text", None)

                    if not chunk_text and getattr(chunk, "candidates", None):
                        parts: List[str] = []
                        for candidate in chunk.candidates:
                            content = getattr(candidate, "content", None)
                            if not content:
                                continue
                            for part in getattr(content, "parts", []) or []:
                                text = getattr(part, "text", None)
                                if text:
                                    parts.append(text)
                        chunk_text = "".join(parts) if parts else None

                    if chunk_text:
                        data = {
                            "choices": [{
                                "delta": {"content": chunk_text}
                            }]
                        }
                        yield f"data: {json.dumps(data)}\n\n"
            except Exception as stream_err:
                err_payload = {"error": str(stream_err)}
                yield f"data: {json.dumps(err_payload)}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
        
    except Exception as e:
        logger.exception("CHAT error | latency_ms=%s", int((time.time() - _t0) * 1000))
        raise HTTPException(status_code=500, detail=str(e))