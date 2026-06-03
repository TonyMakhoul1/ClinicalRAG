from src.rag_ingestion.retrieval import retrieve
from langsmith import Client as LangSmithClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas import evaluate
import os
import time
import uuid
import pandas as pd
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()  # must run before LangChain imports so LANGCHAIN_TRACING_V2 is in os.environ


TEST_SET = [
    {
        "question": "What are the symptoms of diabetes?",
        "reference": (
            "Symptoms of diabetes include blurred vision, chest pain, shortness of breath, "
            "edema, erectile dysfunction, fatty liver, and peripheral numbness."
        ),
    },
    {
        "question": "How is hypertension defined and diagnosed?",
        "reference": (
            "Hypertension is persistently elevated blood pressure. "
            "It is diagnosed by repeated blood pressure measurements at or above 140/90 mmHg."
        ),
    },
    {
        "question": "What symptoms are associated with urinary tract issues in a clinical assessment?",
        "reference": (
            "Urinary symptoms assessed include dysuria and other urinary symptoms along with "
            "associated systemic symptoms such as fever, nausea, and vomiting."
        ),
    },
]


def build_samples() -> list[SingleTurnSample]:
    """
    Run each test question through the retrieval pipeline and wrap results
    into RAGAS SingleTurnSample objects.

    retrieve() returns:
        {
            "query": str,
            "result": str,                    # LLM-generated answer
            "source_documents": list[Document] # top-3 reranked chunks
        }

    SingleTurnSample fields we populate:
        user_input         — the original question
        retrieved_contexts — list of chunk texts (what the retriever returned)
        response           — the LLM answer
        reference          — the ground-truth answer (used by Precision/Recall)
    """
    samples = []

    for i, item in enumerate(TEST_SET, 1):
        print(f"  [{i}/{len(TEST_SET)}] {item['question']}")
        result = retrieve(item["question"])

        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                retrieved_contexts=[
                    doc.page_content for doc in result["source_documents"]],
                response=result["result"],
                reference=item["reference"],
            )
        )
        if i < len(TEST_SET):
            print(f"  Waiting 65s for Groq rate limit to reset...")
            time.sleep(65)

    return samples


def run_evaluation() -> None:
    print("=" * 60)
    print("RAG Evaluation — RAGAS")
    print("=" * 60)

    print("\nStep 1: Collecting pipeline outputs...")
    samples = build_samples()
    dataset = EvaluationDataset(samples=samples)

    print("\nStep 2: Configuring RAGAS judge (Groq LLaMA + HuggingFace embeddings)...")
    llm = LangchainLLMWrapper(
        ChatGroq(
            model=os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile"),
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0.0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )

    print("\nStep 3: Running RAGAS evaluation (this takes ~1-2 minutes)...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,       # answer grounded in retrieved chunks?
            answer_relevancy,   # answer addresses the question?
            context_precision,  # useful chunks ranked highest?
            context_recall,     # all needed info retrieved?
        ],
        llm=llm,
        embeddings=embeddings,
    )

    # Step 4: Print results
    print("\n" + "=" * 60)
    print("AGGREGATE SCORES (average across all questions)")
    print("=" * 60)
    print(results)

    print("\n" + "=" * 60)
    print("PER-QUESTION BREAKDOWN")
    print("=" * 60)
    df = results.to_pandas()
    cols = [c for c in ["user_input", "faithfulness", "answer_relevancy",
                        "context_precision", "context_recall"] if c in df.columns]
    print(df[cols].to_string(index=False))

    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE")
    print("=" * 60)
    print("  faithfulness      : 1.0 = fully grounded, 0.0 = hallucinated")
    print("  answer_relevancy  : 1.0 = directly answers the question")
    print("  context_precision : 1.0 = most useful chunks ranked first")
    print("  context_recall    : 1.0 = all needed info was retrieved")
    print("\nTarget: all metrics >= 0.8 for a production-ready system.")

    print("\nStep 5: Pushing results to LangSmith...")
    push_to_langsmith(df)


def push_to_langsmith(df: pd.DataFrame) -> None:
    """
    Push RAGAS evaluation results to LangSmith as a tagged run.

    Creates one parent run (aggregate scores) with one child run per question
    (individual scores). All runs are tagged 'ragas-evaluation' so they can be
    filtered in smith.langchain.com → Projects → rag-project → filter by tag.

    Each run's outputs contain the metric scores, making it easy to compare
    evaluation runs side by side over time.

    No extra LLM calls — just an API upload of the already-computed scores.
    """
    client = LangSmithClient()
    project = os.environ.get("LANGCHAIN_PROJECT", "rag-project")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    metric_cols = ["faithfulness", "answer_relevancy",
                   "context_precision", "context_recall"]

    # Aggregate scores — NaN values are excluded from the average
    aggregates = {
        col: round(float(df[col].mean(skipna=True)), 4)
        for col in metric_cols
        if col in df.columns and not df[col].isna().all()
    }

    parent_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    client.create_run(
        id=parent_id,
        name=f"RAGAS Evaluation — {timestamp}",
        run_type="chain",
        inputs={"questions": df["user_input"].tolist()},
        outputs=aggregates,
        project_name=project,
        tags=["ragas-evaluation"],
        start_time=now,
        end_time=now,
    )

    for _, row in df.iterrows():
        scores = {
            col: (None if pd.isna(row[col]) else round(float(row[col]), 4))
            for col in metric_cols
            if col in row
        }
        child_id = str(uuid.uuid4())
        client.create_run(
            id=child_id,
            parent_run_id=parent_id,
            name=row["user_input"][:70],
            run_type="chain",
            inputs={"question": row["user_input"]},
            outputs=scores,
            project_name=project,
            tags=["ragas-evaluation"],
            start_time=now,
            end_time=now,
        )

    print(f"  Done - go to smith.langchain.com -> Projects -> {project}")
    print(f"  Filter by tag 'ragas-evaluation' to see all evaluation runs.")


if __name__ == "__main__":
    run_evaluation()
