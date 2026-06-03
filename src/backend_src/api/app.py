from src.backend_src.services.get_rag_resp import RAGResponse, get_rag_response, stream_rag_response
from src.backend_src.services.semantic_cache import init_cache
from src.backend_src.auth.auth import create_access_token, verify_password, verify_token
from src.backend_src.config.backend_settings import BackendSettings
from src.utils.logging import setup_logging
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from slowapi import Limiter
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request, Depends
from contextlib import asynccontextmanager
import structlog
import uuid
from dotenv import load_dotenv
load_dotenv()


setup_logging()

logger = structlog.get_logger()


# create a rate limiter to track request counts, get_remote_add: exctract client's ip address
limiter = Limiter(key_func=get_remote_address)


# FastAPI calls it once when the server start and once when shut down
# it runs first cz it runs before any request can arrive
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app_starting")

    # Try to connect to Redis and create the vector index.
    # if redis down, the app runs without caching and logs warning
    # if redis up, logs redis_connected, cache available ture
    init_cache()

    logger.info("app_ready")

    yield   # server is running and accepting requests while suspended here

    # runs when the app stops, (ctrl+c)
    logger.info("app_shutdown")


# lifespan= wires the context manager into FastAPI's startup/shutdown cycle.
app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)


# registers the limiter with FastAPI so slowapi can find it on every request
app.state.limiter = limiter


# When a client hits the limit, slowapi raises RateLimitExceeded internally.
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning(
        "rate_limit_exceeded",
        # exc.detail contains the limit string: "5 per 1 minute"
        limit=str(exc.detail),
        path=request.url.path,
    )
    return JSONResponse(
        status_code=429,
        content={"error": f"Rate limit exceeded: {exc.detail}. Please slow down."},
        headers={"Retry-After": "60"},
    )


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    # remove old requests
    structlog.contextvars.clear_contextvars()
    # add req id to every log
    structlog.contextvars.bind_contextvars(request_id=request_id)
    # passes request to the next step (endpoint)
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_settings = BackendSettings()


class ChatRequest(BaseModel):
    query: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@app.get("/health")
@limiter.limit("60/minute")
def health(request: Request):
    return {"status": "ok"}


@app.post("/auth/login", response_model=TokenResponse)
@limiter.limit("10/minute")
def login(request: Request, body: LoginRequest):
    # Verify username then password — separate checks to avoid timing differences
    # leaking whether the username exists.
    if body.username != _settings.ADMIN_USERNAME or not verify_password(
        body.password, _settings.ADMIN_PASSWORD_HASH
    ):
        logger.warning("login_failed", username=body.username)
        raise HTTPException(
            status_code=401, detail="Invalid username or password.")
    logger.info("login_success", username=body.username)
    return TokenResponse(access_token=create_access_token(body.username))


@app.post("/chat/answer", response_model=RAGResponse)
@limiter.limit("5/minute")
def chat(request: Request, body: ChatRequest, username: str = Depends(verify_token)):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    try:
        return get_rag_response(body.query)
    except Exception as e:
        logger.error("request_failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
@limiter.limit("5/minute")
def chat_stream(request: Request, body: ChatRequest, username: str = Depends(verify_token)):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    return StreamingResponse(
        stream_rag_response(body.query),
        media_type="text/event-stream",
    )
