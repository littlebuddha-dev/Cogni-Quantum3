# /llm_api/config.py
# タイトル: Centralized Settings Management (Complete Provider Support)
# 役割: プロジェクト全体の設定を管理する。全プロバイダーのデフォルト設定を含む。

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    プロジェクト全体の設定を管理するクラス。
    """
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    # --- API Keys ---
    OPENAI_API_KEY: Optional[str] = None
    CLAUDE_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    HF_TOKEN: Optional[str] = None
    SERPAPI_API_KEY: Optional[str] = None

    # --- Provider Defaults ---
    OLLAMA_API_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: float = 1200.0
    OLLAMA_MAX_RETRIES: int = 3
    OLLAMA_BACKOFF_FACTOR: float = 2.0
    
    # --- Llama.cpp Server Settings ---
    LLAMACPP_API_BASE_URL: Optional[str] = "http://localhost:8000"
    LLAMACPP_DEFAULT_MODEL_PATH: Optional[str] = "./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

    # --- Default Models ---
    OPENAI_DEFAULT_MODEL: str = "gpt-4o-mini"
    CLAUDE_DEFAULT_MODEL: str = "claude-3-haiku-20240307"
    GEMINI_DEFAULT_MODEL: str = "gemini-1.5-flash-latest"
    HUGGINGFACE_DEFAULT_MODEL: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    OLLAMA_DEFAULT_MODEL: str = "gemma3:latest"
    LLAMACPP_DEFAULT_MODEL: str = "./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

    # --- CogniQuantum V2 Settings ---
    V2_DEFAULT_MODE: str = "adaptive"

    # --- Logging ---
    LOG_LEVEL: str = "INFO"


settings = Settings()