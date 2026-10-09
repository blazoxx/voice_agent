from fastapi import FastAPI

from app.api.chat import router as chat_router

app = FastAPI(
    title="Bloxi API",
    description="A modular conversational AI agent with tool-calling capabilities.",
    version="0.1.0",
)

app.include_router(chat_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Check whether the API process is running."""
    return {"status": "ok"}
