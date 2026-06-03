import math

from src.utils.retry import groq_retry
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_groq import ChatGroq

from src.rag_ingestion.config.rag_ingestion_settings import DocIngestionSettings

settings = DocIngestionSettings()

llm = ChatGroq(
    model=settings.MODEL_NAME,
    temperature=settings.MODEL_TEMPERATURE,
    api_key=settings.GROQ_API_KEY,
)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vector_store = Chroma(
    persist_directory=settings.VECTOR_STORE_DIR,
    embedding_function=embeddings,
    collection_name=settings.COLLECTION_NAME,
)

stored = vector_store.get()
all_docs = [
    Document(page_content=text, metadata=meta)
    for text, meta in zip(stored["documents"], stored["metadatas"])
]
bm25_retriever = BM25Retriever.from_documents(all_docs, k=20)
dense_retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 20}
)
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5],
)

cross_encoder = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever,
)


prompt = PromptTemplate(
    template="""You are a helpful assistant. Answer only using the provided context.
Do not use outside knowledge.
If the answer is not in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question: {question}

Helpful answer:""",
    input_variables=["context", "question"],
)


@groq_retry
def invoke_generate(context: str, question: str) -> str:
    # if the call fails and tenacity retries, the chain is rebuilt fresh each time.
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def compute_confidence(query: str, docs: list[Document]) -> float:
    if not docs:
        return 0.0
    raw_scores = cross_encoder.score(
        [(query, doc.page_content) for doc in docs])
    top_score = float(max(raw_scores))
    return round(1.0 / (1.0 + math.exp(-top_score)), 4)


NOT_FOUND = "I could not find the answer in the provided documents."


def stream_retrieve(query: str):
    source_documents = reranking_retriever.invoke(query)
    confidence = compute_confidence(query, source_documents)

    yield "sources", source_documents
    yield "confidence", confidence

    if not source_documents:
        yield "token", NOT_FOUND
        return

    context = "\n\n".join(doc.page_content for doc in source_documents)
    chain = prompt | llm | StrOutputParser()
    for chunk in chain.stream({"context": context, "question": query}):
        yield "token", chunk


def retrieve(query: str) -> dict:
    source_documents = reranking_retriever.invoke(query)
    confidence = compute_confidence(query, source_documents)

    if not source_documents:
        return {
            "query": query,
            "result": NOT_FOUND,
            "source_documents": [],
            "confidence": 0.0,
        }

    context = "\n\n".join(doc.page_content for doc in source_documents)
    result = invoke_generate(context, query)

    return {
        "query": query,
        "result": result,
        "source_documents": source_documents,
        "confidence": confidence,
    }
