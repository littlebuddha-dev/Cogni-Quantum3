# /llm_api/providers/gemini.py
import logging
from typing import Any, Dict

import google.generativeai as genai
from .base import LLMProvider, ProviderCapability
from ..config import settings

logger = logging.getLogger(__name__)

class GeminiProvider(LLMProvider):
    """
    Google Gemini APIと対話するための標準プロバイダー
    """
    def __init__(self):
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEYが設定されていません。")
        genai.configure(api_key=api_key)
        self.default_model = settings.GEMINI_DEFAULT_MODEL
        # モデルの初期化は呼び出し時に行うことで、モデル名の動的変更に対応
        super().__init__()

    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        """このプロバイダーのケイパビリティを返す。"""
        return {
            ProviderCapability.STANDARD_CALL: True,
            ProviderCapability.ENHANCED_CALL: False,
            ProviderCapability.STREAMING: True,
            ProviderCapability.SYSTEM_PROMPT: True,
            ProviderCapability.TOOLS: True,
            ProviderCapability.JSON_MODE: True,
        }

    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        """標準プロバイダーは拡張機能を使用しない。"""
        return False

    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """Gemini APIを呼び出し、標準化された辞書形式で結果を返す。"""
        model_name = kwargs.get("model", self.default_model)
        try:
            model = genai.GenerativeModel(model_name)
            
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

            response = await model.generate_content_async(full_prompt)
            
            return {
                "text": response.text.strip(),
                "model": model_name,
                "usage": {}, # Gemini APIは現在、トークン使用量を直接返さない
                "error": None,
            }
        except Exception as e:
            logger.error(f"Gemini API呼び出し中にエラー: {e}", exc_info=True)
            return {"text": "", "error": str(e)}