def main():
    print("=" * 55)
    print("  Production-Grade Agentic RAG — Startup Guide")
    print("=" * 55)
    print()
    print("Step 1 — Ingest documents into the vector store:")
    print("  uv run python -c \"from src.rag_ingestion.rag_ingestion import ingestion; ingestion()\"")
    print()
    print("Step 2 — Start the FastAPI backend (Terminal 1):")
    print("  uv run uvicorn src.backend_src.api.app:app --reload")
    print("  API docs: http://localhost:8000/docs")
    print()
    print("Step 3 — Start the Streamlit frontend (Terminal 2):")
    print("  uv run streamlit run src/frontend_src/app.py")
    print()


if __name__ == "__main__":
    main()
