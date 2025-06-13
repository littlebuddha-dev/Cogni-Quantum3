# /llm_api/providers/ollama.py
# タイトル: OllamaProvider with Centralized Settings and Functional Retry Logic
# 役割: Ollamaと対話するための標準プロバイダー。設定値をconfigモジュールから正しく取得し、堅牢なリトライ機能を提供する。

import logging
import asyncio  # 修正: asyncioをインポート
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
        リトライ可能なエラーが発生した場合は、呼び出し元で処理できるよう例外を送出する。
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

        # 改善: temperatureなどのパラメータをoptionsにネスト
        options = {}
        supported_options = [
            'temperature', 'top_p', 'top_k', 'mirostat', 'mirostat_eta',
            'mirostat_tau', 'num_ctx', 'repeat_last_n', 'repeat_penalty',
            'seed', 'stop', 'tfs_z', 'num_predict'
        ]
        for key in supported_options:
            if key in kwargs:
                options[key] = kwargs[key]
        if options:
            payload['options'] = options
        
        # 改善: JSONモードをサポート
        if kwargs.get('json_mode'):
            payload['format'] = 'json'

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(api_url, json=payload)
                # 修正: 2xx以外のステータスコードで例外を送出
                response.raise_for_status()
                response_data = response.json()

            full_response = response_data.get('message', {}).get('content', '')
            
            # Nullチェックを追加してトークン数を安全に計算
            prompt_tokens = response_data.get("prompt_eval_count") or 0
            completion_tokens = response_data.get("eval_count") or 0

            return {
                "text": full_response,
                "model": model,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                "error": None
            }
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            # 修正: リトライ可能なエラーはそのまま送出する
            logger.warning(f"Ollama API call failed, propagating exception: {e}")
            raise
        except Exception as e:
            # 修正: 予期せぬエラーは捕捉し、エラー辞書を返す
            error_msg = f"Ollama API呼び出し中に予期せぬエラー: {e}"
            logger.error(error_msg, exc_info=True)
            return {"text": "", "model": model, "usage": {}, "error": error_msg}
    
    # 修正: メソッドをクラス内に正しくインデント
    async def standard_call_with_retry(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """
        リトライロジックを実装したstandard_callのラッパー。
        5xxエラーや接続エラーが発生した場合に指数関数的バックオフでリトライする。
        """
        last_exception = None
        model = kwargs.get("model", self.default_model)

        for attempt in range(settings.OLLAMA_MAX_RETRIES):
            try:
                # 修正: 例外を送出する可能性のある standard_call を呼び出す
                return await self.standard_call(prompt, system_prompt, **kwargs)
            except httpx.HTTPStatusError as e:
                last_exception = e
                # 改善: 5xx系のサーバーエラーの場合にリトライ
                if 500 <= e.response.status_code < 600 and attempt < settings.OLLAMA_MAX_RETRIES - 1:
                    wait_time = settings.OLLAMA_BACKOFF_FACTOR ** attempt
                    logger.warning(
                        f"Ollama API returned status {e.response.status_code}. Retrying in {wait_time:.2f}s... "
                        f"(Attempt {attempt + 1}/{settings.OLLAMA_MAX_RETRIES})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # リトライ対象外のエラー (4xxなど) またはリトライ回数上限に達した場合
                    error_msg = f"Ollama API HTTP Error: {e.response.status_code} - {e.response.text}"
                    logger.error(error_msg)
                    return {"text": "", "model": model, "usage": {}, "error": error_msg}
            except httpx.RequestError as e: # 改善: タイムアウトや接続エラーもリトライ対象に
                last_exception = e
                if attempt < settings.OLLAMA_MAX_RETRIES - 1:
                    wait_time = settings.OLLAMA_BACKOFF_FACTOR ** attempt
                    logger.warning(
                        f"Ollama API request failed: {e}. Retrying in {wait_time:.2f}s... "
                        f"(Attempt {attempt + 1}/{settings.OLLAMA_MAX_RETRIES})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    error_msg = f"Ollama API request error after max retries: {e}"
                    logger.error(error_msg)
                    return {"text": "", "model": model, "usage": {}, "error": error_msg}
        
        # このパスには通常到達しないはずだが、フォールバックとして残す
        final_error_msg = f"Failed to get a response from Ollama after {settings.OLLAMA_MAX_RETRIES} attempts. Last exception: {last_exception}"
        logger.error(final_error_msg)
        return {"text": "", "model": model, "usage": {}, "error": final_error_msg}

    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        """標準プロバイダーは拡張機能を使用しない。"""
        return False