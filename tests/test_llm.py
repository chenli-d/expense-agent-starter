import pytest

from app import config
from app.llm import GeminiLLM


def test_gemini_raises_clear_error_when_model_env_is_empty(monkeypatch):
    monkeypatch.setattr(config, "MODEL", "")
    with pytest.raises(ValueError, match="MODEL"):
        GeminiLLM(api_key="test-key-not-used")


def test_gemini_raises_clear_error_when_model_is_blank():
    with pytest.raises(ValueError, match="MODEL"):
        GeminiLLM(api_key="test-key-not-used", model="   ")
