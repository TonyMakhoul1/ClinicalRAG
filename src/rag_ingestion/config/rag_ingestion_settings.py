from pydantic_settings import BaseSettings, SettingsConfigDict


class DocIngestionSettings(BaseSettings):
    DOCUMENTS_DIR: str
    VECTOR_STORE_DIR: str
    GROQ_API_KEY: str
    MODEL_NAME: str = "llama-3.3-70b-versatile"
    GUARDRAIL_MODEL: str = "llama-3.1-8b-instant"
    MODEL_TEMPERATURE: float = 0.0
    COLLECTION_NAME: str = "my_collection1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )
