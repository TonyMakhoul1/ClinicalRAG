import json
import time
from typing import Generator

import structlog
from pydantic import BaseModel

from src.agents_src.crew import run_crew
from src.backend_src.services.semantic_cache import check_cache, store_cache
from src.backend_src.services.guardrails import check_input, check_output

logger = structlog.get_logger()

UNGROUNDED = "I could not find a reliable answer in the provided documents."
REJECTED = "This query is outside the scope of the available documents."


class SourceDoc(BaseModel):
    content: str
    source:  str
    page:    int


class RAGResponse(BaseModel):
    query:      str
    answer:     str
    sources:    list[SourceDoc]
    confidence: float


def stream_rag_response(query: str) -> Generator:
    # Bind query to this logger instance so every log call below automatically includes it
    log = logger.bind(query=query[:80])

    t0 = time.perf_counter()

    cached = check_cache(query)
    if cached:
        log.info("cache_hit", confidence=cached["confidence"])
        yield f"data: {json.dumps({'type': 'confidence', 'data': cached['confidence']})}\n\n"
        yield f"data: {json.dumps({'type': 'sources',    'data': cached['sources']})}\n\n"
        for word in cached["answer"].split(" "):
            yield f"data: {json.dumps({'type': 'token', 'data': word + ' '})}\n\n"
        yield "data: [DONE]\n\n"
        return

    log.info("cache_miss")

    is_valid, reason = check_input(query)
    if not is_valid:
        # log the reason comes from the guardrail
        log.warning("input_rejected", reason=reason)
        yield f"data: {json.dumps({'type': 'error', 'data': REJECTED})}\n\n"
        yield "data: [DONE]\n\n"
        return

    yield f"data: {json.dumps({'type': 'thinking', 'data': 'Analyzing and researching your question...'})}\n\n"

    try:
        raw = run_crew(query)
    except Exception:
        # attach the full traceback to the log event as a single field
        log.error("crew_failed", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'data': 'Service temporarily unavailable. Please try again in a moment.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    sources = [
        SourceDoc(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            page=int(doc.metadata.get("page", 0)),
        )
        for doc in raw["source_documents"]
    ]

    yield f"data: {json.dumps({'type': 'confidence', 'data': raw['confidence']})}\n\n"
    yield f"data: {json.dumps({'type': 'sources', 'data': [s.model_dump() for s in sources]})}\n\n"

    # If the crew retrieved no documents, there is nothing to ground the answer in.
    if not sources:
        log.warning("no_sources_retrieved", confidence=raw["confidence"])
        yield f"data: {json.dumps({'type': 'retract', 'data': UNGROUNDED})}\n\n"
        yield "data: [DONE]\n\n"
        return

    context = "\n\n".join(s.content for s in sources)
    is_grounded, reason = check_output(query, context, raw["result"])
    if not is_grounded:
        log.warning("output_not_grounded", reason=reason,
                    confidence=raw["confidence"])
        yield f"data: {json.dumps({'type': 'retract', 'data': UNGROUNDED})}\n\n"
        yield "data: [DONE]\n\n"
        return

    full_answer = raw["result"]
    for word in full_answer.split(" "):
        yield f"data: {json.dumps({'type': 'token', 'data': word + ' '})}\n\n"

    store_cache(query, full_answer, [s.model_dump()
                for s in sources], raw["confidence"])

    # duration_ms measures the full pipeline: cache miss → guardrails → crew → answer
    duration_ms = int((time.perf_counter() - t0) * 1000)
    log.info("response_sent",
             confidence=raw["confidence"], duration_ms=duration_ms)

    yield "data: [DONE]\n\n"


def get_rag_response(query: str) -> RAGResponse:
    log = logger.bind(query=query[:80])
    t0 = time.perf_counter()

    cached = check_cache(query)
    if cached:
        log.info("cache_hit", confidence=cached["confidence"])
        return RAGResponse(
            query=query,
            answer=cached["answer"],
            sources=[SourceDoc(**s) for s in cached["sources"]],
            confidence=cached["confidence"],
        )

    log.info("cache_miss")

    is_valid, reason = check_input(query)
    if not is_valid:
        log.warning("input_rejected", reason=reason)
        return RAGResponse(query=query, answer=REJECTED, sources=[], confidence=0.0)

    raw = run_crew(query)

    sources = [
        SourceDoc(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            page=int(doc.metadata.get("page", 0)),
        )
        for doc in raw["source_documents"]
    ]

    if not sources:
        log.warning("no_sources_retrieved", confidence=raw["confidence"])
        return RAGResponse(query=query, answer=UNGROUNDED, sources=[], confidence=raw["confidence"])

    context = "\n\n".join(s.content for s in sources)
    is_grounded, reason = check_output(query, context, raw["result"])
    if not is_grounded:
        log.warning("output_not_grounded", reason=reason,
                    confidence=raw["confidence"])
        return RAGResponse(query=query, answer=UNGROUNDED, sources=sources, confidence=raw["confidence"])

    response = RAGResponse(
        query=query,
        answer=raw["result"],
        sources=sources,
        confidence=raw["confidence"],
    )
    store_cache(query, response.answer, [s.model_dump()
                for s in response.sources], response.confidence)

    duration_ms = int((time.perf_counter() - t0) * 1000)
    log.info("response_sent", confidence=response.confidence,
             duration_ms=duration_ms)

    return response
