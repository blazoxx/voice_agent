from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str
    elevenlabs_api_key: str = ""

    llm_model: str = "openai/gpt-oss-20b"
    stt_model: str = "whisper-large-v3-turbo"
    tts_voice_id: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()