from src.agents_src.tools.rag_tool import get_results, clear_results
from src.agents_src.tasks import planning_task, research_task, synthesis_task
from src.agents_src.agents import planner_agent, researcher_agent, synthesizer_agent
from src.utils.retry import groq_retry
from crewai import Crew, Process
import time
import structlog
import litellm
from langsmith import traceable

logger = structlog.get_logger()


litellm.success_callback = ["langsmith"]


@traceable(name="CrewAI RAG Pipeline")
def run_crew(query: str) -> dict:
    """
    Run the full agentic RAG pipeline for a query.

    Flow:
      1. Planner   → decomposes query into sub-questions
      2. Researcher → calls RAG tool for each sub-question
      3. Synthesizer → combines all results into a final answer

    Returns the same dict shape as retrieve():
        {
            "query"           : str,
            "result"          : str,             # final synthesized answer
            "source_documents": list[Document],  # all docs from all sub-queries
            "confidence"      : float,           # average across all sub-queries
        }
    """
    clear_results()

    logger.info("crew_started", query=query[:80])

    crew = Crew(
        agents=[planner_agent, researcher_agent, synthesizer_agent],
        tasks=[planning_task, research_task, synthesis_task],
        process=Process.sequential,
        verbose=False,
    )

    @groq_retry
    def _kickoff():
        return crew.kickoff(inputs={"query": query})

    t0 = time.perf_counter()
    crew_output = _kickoff()
    duration_ms = int((time.perf_counter() - t0) * 1000)

    final_answer = str(crew_output)

    # Collect all sources and confidences from every retrieve() call
    all_results = get_results()
    all_source_docs = []
    confidences = []

    for result in all_results:
        all_source_docs.extend(result["source_documents"])
        confidences.append(result["confidence"])

    overall_confidence = round(
        sum(confidences) / len(confidences), 4) if confidences else 0.0

    logger.info(
        "crew_done",
        confidence=overall_confidence,
        duration_ms=duration_ms,
        # sub_queries tells you how many RAG tool calls the researcher made.
        sub_queries=len(confidences),
    )

    return {
        "query":            query,
        "result":           final_answer,
        "source_documents": all_source_docs,
        "confidence":       overall_confidence,
    }
