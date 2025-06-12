# /llm_api/providers/enhanced_gemini_v2.py
# タイトル: Refactored EnhancedGeminiProviderV2
# 役割: GeminiプロバイダーにCogniQuantum V2の機能を提供する。共通ロジックは基底クラスに委譲し、パラメータ最適化に特化。

from typing import Any, Dict

from .base import EnhancedLLMProvider, ProviderCapability

class EnhancedGeminiProviderV2(EnhancedLLMProvider):
    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        return await self.standard_provider.standard_call(prompt, system_prompt, **kwargs)

    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        return kwargs.get('force_v2', False) or kwargs.get('mode', 'simple') in [
            'efficient', 'balanced', 'decomposed', 'adaptive', 'paper_optimized', 'parallel',
            'quantum_inspired', 'edge'
        ]

    def _get_optimized_params(self, mode: str, kwargs: Dict) -> Dict:
        """Geminiに最適化されたモデルパラメータを返す。"""
        params = kwargs.copy()
        if 'model' not in params:
            if mode in ['decomposed', 'paper_optimized', 'quantum_inspired']:
                params['model'] = 'gemini-1.5-pro-latest' # 高度なタスクにはPro
            else:
                params['model'] = 'gemini-1.5-flash-latest' # それ以外はFlash
        return params

    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        capabilities = self.standard_provider.get_capabilities()
        capabilities[ProviderCapability.ENHANCED_CALL] = True
        return capabilities