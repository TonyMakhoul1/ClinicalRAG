from crewai import Agent, LLM
from src.backend_src.config.backend_settings import BackendSettings
from src.agents_src.tools.rag_tool import rag_search

settings = BackendSettings()

# Full model for the synthesizer — final answer quality matters.
llm = LLM(
    model=f"groq/{settings.MODEL_NAME}",
    temperature=0.0,
    api_key=settings.GROQ_API_KEY,
    num_retries=3,
)

llm_fast = LLM(
    model=f"groq/{settings.GUARDRAIL_MODEL}",
    temperature=0.0,
    api_key=settings.GROQ_API_KEY,
    num_retries=3,
)


planner_agent = Agent(
    role="Medical Query Analyst",
    goal="Analyze medical questions and produce a precise, focused research plan.",
    backstory=(
        "You are an expert at understanding complex medical questions. "
        "You can tell immediately whether a question is simple (single topic) "
        "or complex (multi-topic, comparison, or multi-part). "
        "For complex questions you break them into the minimum number of focused "
        "sub-questions needed — never more than 4."
    ),
    llm=llm_fast,
    allow_delegation=False,
    verbose=False,
)


researcher_agent = Agent(
    role="Medical Researcher",
    goal="Retrieve accurate medical information for every sub-question in the plan.",
    backstory=(
        "You are a thorough medical researcher with access to a clinical document database. "
        "You always call the Medical Document Search tool once for each sub-question — "
        "never skipping any, never combining multiple questions into one search. "
        "Your job is to collect information, not to interpret it."
    ),
    tools=[rag_search],
    llm=llm,
    allow_delegation=False,
    verbose=False,
)


synthesizer_agent = Agent(
    role="Medical Writer",
    goal="Synthesize all research findings into one clear, accurate, well-structured answer.",
    backstory=(
        "You are a precise medical writer. You combine multiple research results "
        "into a single coherent response. You only state what the research supports — "
        "you never add outside knowledge or speculate beyond what was found. "
        "If information was missing from the research, you say so explicitly."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=False,
)
