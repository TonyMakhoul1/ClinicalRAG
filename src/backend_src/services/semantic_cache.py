import structlog
import numpy as np
import json
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

from src.backend_src.config.backend_settings import BackendSettings
from src.utils.retry import redis_retry

logger = structlog.get_logger()

settings = BackendSettings()

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

SCHEMA = {
    "index": {
        "name": "rag_semantic_cache",
        "prefix": "rag_cache",
    },
    "fields": [
        {"name": "query", "type": "text"},
        {"name": "answer", "type": "text"},
        {"name": "sources", "type": "text"},
        {"name": "confidence", "type": "text"},
        {"name": "cached_at", "type": "text"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "dims": 384,
                "distance_metric": "cosine",
                "algorithm": "hnsw",
                "datatype": "float32",
            },
        },
    ]
}

# index is built from the schema definition, the connection to redis deferred to init_cache()
# which is called explicitly at startup
index = SearchIndex.from_dict(SCHEMA)


# tracks whether redis was reachable at startip (only set true if init_cache() succeeds)
cache_available = False

DISTANCE_THRESHOLD = 0.08


def init_cache() -> None:
    # Called once at application startup from the FastAPI lifespan hook.
    # Attempts to connect to Redis and create the vector index.
    # if success logs the url and put the variable true
    # if failure put put the varibale false and logs a warning and leave
    # so the app continue without semantic caching
    global cache_available
    try:
        index.connect(settings.REDIS_URL)
        index.create(overwrite=False)
        cache_available = True
        logger.info("redis_connected", url=settings.REDIS_URL)
    except Exception:
        logger.warning(
            "redis_unavailable",
            url=settings.REDIS_URL,
            detail="semantic cache disabled — app runs without caching",
        )
        cache_available = False


@redis_retry
def _query_cache(embedding: list) -> list:
    return index.query(
        VectorQuery(
            vector=embedding,
            vector_field_name="embedding",
            return_fields=["answer", "sources",
                           "confidence", "query", "cached_at"],
            num_results=1,
        )
    )


def check_cache(query: str) -> dict | None:
    # Fast exit if Redis was unavailable at startup, no network call, no retry.
    if not cache_available:
        return None

    try:
        embedding = embeddings.embed_query(query)
        result = _query_cache(embedding)
    except Exception:
        # Redis was up at startup but failed during this request
        # Degrade gracefully, return None so the full pipeline runs.
        logger.warning("cache_read_failed")
        return None

    if result and float(result[0]["vector_distance"]) <= DISTANCE_THRESHOLD:
        return {
            "answer": result[0]["answer"],
            "sources": json.loads(result[0]["sources"]),
            "confidence": float(result[0].get("confidence") or 0.0),
        }
    return None


@redis_retry
def _load_cache(entry: dict) -> None:
    index.load([entry])


def store_cache(query: str, answer: str, sources: list, confidence: float = 0.0) -> None:
    # Same fast exit, if Redis never came up, don't attempt to write.
    if not cache_available:
        return

    try:
        embedding = embeddings.embed_query(query)
        _load_cache({
            "query":      query,
            "answer":     answer,
            "sources":    json.dumps(sources),
            "confidence": str(confidence),
            "cached_at":  datetime.utcnow().isoformat(),
            "embedding":  np.array(embedding, dtype=np.float32).tobytes(),
        })
    except Exception:
        logger.warning("cache_write_failed")
