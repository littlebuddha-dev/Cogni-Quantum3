# /llm_api/providers/huggingface.py
import logging
from typing import Any, Dict

from huggingface_hub import AsyncInferenceClient
from .base import LLMProvider, ProviderCapability
from ..config import settings

logger = logging.getLogger(__name__)

class HuggingFaceProvider(LLMProvider):
    """
    Hugging Face Inference APIと対話するための標準プロバイダー
    """
    def __init__(self):
        self.client = AsyncInferenceClient(token=settings.HF_TOKEN)
        self.default_model = settings.HUGGINGFACE_DEFAULT_MODEL
        super().__init__()

    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        """このプロバイダーのケイパビリティを返す。"""
        return {
            ProviderCapability.STANDARD_CALL: True,
            ProviderCapability.ENHANCED_CALL: False,
            ProviderCapability.STREAMING: False,
            ProviderCapability.SYSTEM_PROMPT: True,
            ProviderCapability.TOOLS: False,
            ProviderCapability.JSON_MODE: False,
        }

    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        """標準プロバイダーは拡張機能を使用しない。"""
        return False
        
    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """Hugging Face Inference APIを呼び出し、標準化された辞書形式で結果を返す。"""
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>"
        else:
            full_prompt = prompt
            
        model_to_use = kwargs.get("model", self.default_model)

        try:
            response_text = await self.client.text_generation(
                prompt=full_prompt,
                model=model_to_use,
                max_new_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.7),
            )
            
            return {
                "text": response_text.strip(),
                "model": model_to_use,
                "usage": {}, # HF APIはトークン使用量を返さない
                "error": None,
            }
        except Exception as e:
            logger.error(f"Hugging Face API呼び出し中にエラー: {e}", exc_info=True)
            return {"text": "", "error": str(e)}