# /llm_api/providers/ollama.py
# タイトル: OllamaProvider with Centralized Settings
# 役割: Ollamaと対話するための標準プロバイダー。設定値をconfigモジュールから正しく取得する。

import logging
from typing import Any, Dict

import httpx
from .base import LLMProvider, ProviderCapability
from ..config import settings

logger = logging.getLogger(__name__)

class OllamaProvider(LLMProvider):
    """
    Ollamaと対話するための標準プロバイダー
    """
    def __init__(self):
        self.api_base_url = settings.OLLAMA_API_BASE_URL
        self.default_model = settings.OLLAMA_DEFAULT_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT
        super().__init__()
        logger.info(f"Ollama provider initialized with API URL: {self.api_base_url} and default model: {self.default_model}")

    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        """このプロバイダーのケイパビリティを返す。"""
        return {
            ProviderCapability.STANDARD_CALL: True,
            ProviderCapability.ENHANCED_CALL: False,
            ProviderCapability.STREAMING: False,
            ProviderCapability.SYSTEM_PROMPT: True,
            ProviderCapability.TOOLS: False,
            ProviderCapability.JSON_MODE: True,
        }

    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """
        Ollama APIを呼び出し、標準化された辞書形式で結果を返す。
        """
        api_url = f"{self.api_base_url}/api/chat"
        model = kwargs.get("model", self.default_model)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        for key in ['temperature', 'top_p', 'top_k']:
            if key in kwargs:
                payload[key] = kwargs[key]

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(api_url, json=payload)
                response.raise_for_status()
                response_data = response.json()

            full_response = response_data.get('message', {}).get('content', '')
            
            return {
                "text": full_response,
                "model": model,
                "usage": {
                    "prompt_tokens": response_data.get("prompt_eval_count", 0),
                    "completion_tokens": response_data.get("eval_count", 0),
                    "total_tokens": response_data.get("prompt_eval_count", 0) + response_data.get("eval_count", 0)
                },
                "error": None
            }
        except httpx.HTTPStatusError as e:
            error_msg = f"Ollama API HTTPエラー: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            return {"text": "", "error": error_msg}
        except Exception as e:
            error_msg = f"Ollama API呼び出し中にエラー: {e}"
            logger.error(error_msg, exc_info=True)
            return {"text": "", "error": error_msg}

    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        """標準プロバイダーは拡張機能を使用しない。"""
        return False