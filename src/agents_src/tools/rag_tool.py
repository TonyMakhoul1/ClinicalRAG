from crewai.tools import tool
from src.rag_ingestion.retrieval import retrieve

# Accumulates every retrieve() call result made during one crew run.
results_store: list = []


@tool("Medical Document Search")
def rag_search(query: str) -> str:
    """
    Search the medical document database for a specific, focused medical question.
    Use this tool once per sub-question. Input must be a single medical question.
    Returns an answer, confidence score, and supporting source excerpts.
    """
    result = retrieve(query)

    # Store the full result so crew.py can collect sources and confidence later
    results_store.append(result)

    # Return a readable summary to the agent — it uses this to write its findings
    sources_text = "\n".join(
        f"  [Page {doc.metadata.get('page', '?')}] {doc.page_content[:120]}..."
        for doc in result["source_documents"][:3]
    )
    return (
        f"Answer: {result['result']}\n"
        f"Confidence: {result['confidence']:.0%}\n"
        f"Top sources:\n{sources_text}"
    )


def get_results() -> list:
    """Return all retrieve() results accumulated during the current crew run."""
    return list(results_store)


def clear_results() -> None:
    """Reset the store before each new request."""
    results_store.clear()
