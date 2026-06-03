from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.rag_ingestion.config.rag_ingestion_settings import DocIngestionSettings

settings = DocIngestionSettings()

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def ingestion():
    loader = DirectoryLoader(
        path=settings.DOCUMENTS_DIR,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader,
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(documents=documents)

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "Unknown")
        source = source.replace("\\", "/").split("/")[-1]
        page = chunk.metadata.get("page", 0)
        chunk.metadata.update(
            {
                "source": source,
                "page": page,
                "chunk_id": f"{source}:p{page}:chunk_{i}",
                "total_chunks": len(chunks),
            }
        )

    # Wipe the existing collection so re-running ingestion never duplicates chunks.
    existing = Chroma(
        persist_directory=settings.VECTOR_STORE_DIR,
        embedding_function=embeddings,
        collection_name=settings.COLLECTION_NAME,
    )
    existing.delete_collection()

    vector_store = Chroma.from_documents(
        documents=chunks,
        persist_directory=settings.VECTOR_STORE_DIR,
        embedding=embeddings,
        collection_name=settings.COLLECTION_NAME,
    )

    return vector_store
