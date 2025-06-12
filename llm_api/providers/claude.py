# /llm_api/providers/claude.py
import logging
from typing import Any, Dict

from anthropic import AsyncAnthropic
from .base import LLMProvider, ProviderCapability
from ..config import settings

logger = logging.getLogger(__name__)

class ClaudeProvider(LLMProvider):
    """
    Anthropic Claude APIと対話するための標準プロバイダー
    """
    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.CLAUDE_API_KEY)
        self.default_model = settings.CLAUDE_DEFAULT_MODEL
        super().__init__()

    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        """このプロバイダーのケイパビリティを返す。"""
        return {
            ProviderCapability.STANDARD_CALL: True,
            ProviderCapability.ENHANCED_CALL: False,
            ProviderCapability.STREAMING: True,
            ProviderCapability.SYSTEM_PROMPT: True,
            ProviderCapability.TOOLS: True,
            ProviderCapability.JSON_MODE: False,
        }

    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        """標準プロバイダーは拡張機能を使用しない。"""
        return False

    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """Claude APIを呼び出し、標準化された辞書形式で結果を返す。"""
        model_to_use = kwargs.get("model", self.default_model)
        try:
            response = await self.client.messages.create(
                model=model_to_use,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1024),
            )
            
            content = response.content[0].text
            usage = response.usage

            return {
                "text": content.strip(),
                "model": response.model,
                "usage": {
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                    "total_tokens": usage.input_tokens + usage.output_tokens,
                },
                "error": None,
            }
        except Exception as e:
            logger.error(f"Claude API呼び出し中にエラー: {e}", exc_info=True)
            return {"text": "", "error": str(e)}