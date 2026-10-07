from app.providers.llm import GroqProvider


def test_llm_generates_response():
    provider = GroqProvider()

    response = provider.generate(
        [
            {
                "role": "user",
                "content": "Reply with exactly: Hello from Bloxi.",
            }
        ]
    )

    assert response
    assert "Hello from Bloxi" in response