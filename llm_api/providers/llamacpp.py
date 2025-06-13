# /llm_api/providers/llamacpp.py
# タイトル: Llama.cpp Standard Provider (Complete)
# 役割: llama-cpp-pythonのOpenAI互換サーバーと直接対話するための標準プロバイダー。

import logging
from typing import Any, Dict

import httpx
from .base import LLMProvider, ProviderCapability
from ..config import settings

logger = logging.getLogger(__name__)

class LlamaCppProvider(LLMProvider):
    """
    llama-cpp-pythonのOpenAI互換サーバーと対話するための標準プロバイダー
    """
    def __init__(self):
        if not settings.LLAMACPP_API_BASE_URL:
            raise ValueError("LLAMACPP_API_BASE_URLが.envファイルに設定されていません。")
        
        self.api_url = f"{settings.LLAMACPP_API_BASE_URL.rstrip('/')}/v1/chat/completions"
        self.default_model = settings.LLAMACPP_DEFAULT_MODEL_PATH or "llama-model"
        self.client = httpx.AsyncClient(timeout=600.0)
        super().__init__()
        logger.info(f"LlamaCpp provider initialized with API URL: {self.api_url}")

    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        """このプロバイダーのケイパビリティを返す。"""
        return {
            ProviderCapability.STANDARD_CALL: True,
            ProviderCapability.ENHANCED_CALL: False,
            ProviderCapability.STREAMING: True,
            ProviderCapability.SYSTEM_PROMPT: True,
            ProviderCapability.TOOLS: False,
            ProviderCapability.JSON_MODE: True,
        }

    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        """標準プロバイダーは拡張機能を使用しない。"""
        return False
        
    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """Llama.cppサーバーを呼び出し、標準化された辞書形式で結果を返す。"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # llama-cpp-pythonサーバーはmodel引数を必要としない場合が多い
        payload = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": False,
        }

        # top_pやtop_kがあれば追加
        for param in ['top_p', 'top_k', 'repeat_penalty']:
            if param in kwargs:
                payload[param] = kwargs[param]

        try:
            response = await self.client.post(self.api_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage", {})

            return {
                "text": content.strip(),
                "model": response_data.get("model", self.default_model),
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                "error": None,
            }
        except httpx.HTTPStatusError as e:
            error_msg = f"Llama.cpp API HTTPエラー: {e.response.status_code}"
            try:
                error_detail = e.response.text
                error_msg += f" - {error_detail}"
            except:
                pass
            logger.error(error_msg)
            return {"text": "", "error": error_msg}
        except httpx.RequestError as e:
            error_msg = f"Llama.cpp API接続エラー: {e}"
            logger.error(error_msg)
            return {"text": "", "error": error_msg}
        except Exception as e:
            error_msg = f"Llama.cpp API呼び出し中にエラー: {e}"
            logger.error(error_msg, exc_info=True)
            return {"text": "", "error": error_msg}
            
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()