from crewai import Task
from src.agents_src.agents import planner_agent, researcher_agent, synthesizer_agent


planning_task = Task(
    description="""Analyze this medical question: {query}

Classify it:
- SIMPLE (single focused topic) → return the question itself as one item
- COMPLEX (multiple topics, comparison, or multi-part) → break into 2-4 focused sub-questions

Rules:
- Each sub-question must be self-contained and searchable on its own
- Use precise medical language in sub-questions
- Return ONLY the numbered list, nothing else — no headers, no explanations

Example — simple: "What is hypertension?"
1. What is hypertension?

Example — complex: "Compare symptoms and treatment of hypothyroidism vs hyperthyroidism"
1. What are the symptoms of hypothyroidism?
2. What is the treatment for hypothyroidism?
3. What are the symptoms of hyperthyroidism?
4. What is the treatment for hyperthyroidism?""",
    expected_output="A numbered list of focused medical sub-questions, one per line. Nothing else.",
    agent=planner_agent,
)


research_task = Task(
    description="""You have received a numbered list of medical sub-questions to research.

Instructions:
1. Read each sub-question from the list above
2. Call the Medical Document Search tool ONCE per sub-question
3. Do NOT combine multiple sub-questions into one search
4. Do NOT skip any sub-question

After all searches are complete, compile your findings:
- List each sub-question
- Its answer from the search
- The confidence score returned
- Key source excerpts""",
    expected_output=(
        "A compiled report with each sub-question, its retrieved answer, "
        "confidence score, and supporting source excerpts."
    ),
    agent=researcher_agent,  # receives the sub-question list from the planner
    context=[planning_task],
)


synthesis_task = Task(
    description="""Original question: {query}

You have received research findings for all sub-questions above.
Write the final answer to the original question using ONLY information present in those findings.

Guidelines:
- Single-topic question → one clear, direct paragraph
- Multi-topic question → use ## headers to organize sections
- Do NOT add any information not explicitly present in the research findings
- Do NOT use your own training knowledge — only what the documents returned
- Be concise and accurate
- Do NOT mention confidence scores or percentages — those are internal metadata, not part of the answer

CRITICAL RULE: If the research findings contain no relevant information (empty results, "no
information found", or similar), you MUST respond with ONLY this exact sentence and nothing else:
"I could not find information about this in the provided documents." """,
    expected_output=(
        "A comprehensive, well-structured final answer to the original question "
        "grounded entirely in the research findings, OR the exact sentence "
        "'I could not find information about this in the provided documents.' if nothing was found."
    ),
    agent=synthesizer_agent,  # receives all retrieved answers from the researcher
    context=[research_task],
)
