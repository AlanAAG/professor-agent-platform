import sys
import types

# Provide lightweight stubs so rag_core can be imported without heavy optional dependencies.
if "langchain_google_genai" not in sys.modules:
    fake_genai = types.ModuleType("langchain_google_genai")

    class _FakeLLM:  # pragma: no cover - test-only stub
        def __init__(self, *args, **kwargs):
            pass

    fake_genai.ChatGoogleGenerativeAI = _FakeLLM
    sys.modules["langchain_google_genai"] = fake_genai

if "langchain_core" not in sys.modules:
    langchain_core = types.ModuleType("langchain_core")
    sys.modules["langchain_core"] = langchain_core
else:
    langchain_core = sys.modules["langchain_core"]

if "langchain_core.prompts" not in sys.modules:
    prompts_module = types.ModuleType("langchain_core.prompts")

    class _FakePromptTemplate:  # pragma: no cover - test-only stub
        @classmethod
        def from_template(cls, template: str):
            return cls()

        def __or__(self, other):
            return self

        def invoke(self, *args, **kwargs):
            return ""

    prompts_module.ChatPromptTemplate = _FakePromptTemplate
    sys.modules["langchain_core.prompts"] = prompts_module
    setattr(langchain_core, "prompts", prompts_module)

if "langchain_core.output_parsers" not in sys.modules:
    output_parsers_module = types.ModuleType("langchain_core.output_parsers")

    class _FakeStrOutputParser:  # pragma: no cover - test-only stub
        def __ror__(self, other):
            return self

        def invoke(self, *args, **kwargs):
            return ""

    output_parsers_module.StrOutputParser = _FakeStrOutputParser
    sys.modules["langchain_core.output_parsers"] = output_parsers_module
    setattr(langchain_core, "output_parsers", output_parsers_module)

if "langchain_core.documents" not in sys.modules:
    documents_module = types.ModuleType("langchain_core.documents")

    class _FakeDocument:  # pragma: no cover - test-only stub
        def __init__(self, page_content: str = "", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    documents_module.Document = _FakeDocument
    sys.modules["langchain_core.documents"] = documents_module
    setattr(langchain_core, "documents", documents_module)

if "langchain_core.messages" not in sys.modules:
    messages_module = types.ModuleType("langchain_core.messages")

    class _FakeHumanMessage:  # pragma: no cover - test-only stub
        def __init__(self, content: str = ""):
            self.content = content

    class _FakeAIMessage(_FakeHumanMessage):  # pragma: no cover - test-only stub
        pass

    messages_module.HumanMessage = _FakeHumanMessage
    messages_module.AIMessage = _FakeAIMessage
    sys.modules["langchain_core.messages"] = messages_module
    setattr(langchain_core, "messages", messages_module)

from src.app import rag_core


def test_classify_subject_prefers_finance_for_profit_question():
    subject, boosted_scores, raw_scores = rag_core.classify_subject(
        "what is the formula for profit?",
        "FinanceBasics",
        return_scores=True,
    )

    assert subject == "FinanceBasics"
    assert raw_scores.get("FinanceBasics", 0) > 0
    # Ensure Excel does not overpower the finance course any more.
    assert boosted_scores.get("FinanceBasics", 0) >= boosted_scores.get("Excel", 0)


def test_redirect_triggers_for_unrelated_elasticity_question():
    subject, boosted_scores, raw_scores = rag_core.classify_subject(
        "give me the elasticity of a product",
        "Dropshipping",
        return_scores=True,
    )

    # Classification should recognise the MarketAnalysis course for this query.
    assert subject == "MarketAnalysis"
    redirect_message = rag_core._maybe_redirect_for_irrelevant_query(
        active_subject="Dropshipping",
        classified_subject=subject,
        subject_scores=boosted_scores,
        raw_subject_scores=raw_scores,
        class_hint="Dropshipping",
    )

    assert redirect_message is not None
    assert "Market" in redirect_message or "Garima" in redirect_message
