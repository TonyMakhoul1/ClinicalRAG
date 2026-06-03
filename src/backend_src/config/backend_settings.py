from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    GROQ_API_KEY: str
    MODEL_NAME: str = "llama-3.3-70b-versatile"
    MODEL_TEMPERATURE: float = 0.0
    VECTOR_STORE_DIR: str
    COLLECTION_NAME: str = "my_collection1"
    CHAT_ENDPOINT_URL: str = "http://localhost:8000/chat/answer"
    REDIS_URL: str = "redis://localhost:6379"

    GUARDRAIL_MODEL: str = "llama-3.1-8b-instant"

    # JWT authentication
    SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60
    ADMIN_USERNAME: str
    ADMIN_PASSWORD_HASH: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )
