# /llm_api/providers/enhanced_huggingface_v2.py
# タイトル: Refactored EnhancedHuggingFaceProviderV2
# 役割: HuggingFaceプロバイダーにCogniQuantum V2の機能を提供する。共通ロジックは基底クラスに委譲し、パラメータ最適化に特化。

from typing import Any, Dict

from .base import EnhancedLLMProvider, ProviderCapability

class EnhancedHuggingFaceProviderV2(EnhancedLLMProvider):
    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        return await self.standard_provider.standard_call(prompt, system_prompt, **kwargs)

    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        return kwargs.get('force_v2', False) or kwargs.get('mode', 'simple') in [
            'efficient', 'balanced', 'decomposed', 'adaptive', 'paper_optimized', 'parallel',
            'quantum_inspired', 'edge'
        ]

    def _get_optimized_params(self, mode: str, kwargs: Dict) -> Dict:
        """HuggingFaceに最適化されたモデルパラメータを返す。"""
        params = kwargs.copy()
        if 'model' not in params:
            params['model'] = 'meta-llama/Meta-Llama-3-8B-Instruct'
        return params

    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        capabilities = self.standard_provider.get_capabilities()
        capabilities[ProviderCapability.ENHANCED_CALL] = True
        return capabilities