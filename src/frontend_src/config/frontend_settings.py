from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontendSettings(BaseSettings):
    CHAT_ENDPOINT_URL: str = "http://localhost:8000/chat/answer"
    CHAT_STREAM_URL: str = "http://localhost:8000/chat/stream"
    AUTH_URL: str = "http://localhost:8000/auth/login"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )
