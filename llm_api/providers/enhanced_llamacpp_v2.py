# /llm_api/providers/enhanced_llamacpp_v2.py
# タイトル: Enhanced Llama.cpp Provider V2
# 役割: Llama.cppプロバイダーにCogniQuantum V2の機能を提供する。

from typing import Any, Dict

from .base import EnhancedLLMProvider, ProviderCapability

class EnhancedLlamaCppProviderV2(EnhancedLLMProvider):
    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        return await self.standard_provider.standard_call(prompt, system_prompt, **kwargs)

    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        return kwargs.get('force_v2', False) or kwargs.get('mode', 'simple') in [
            'efficient', 'balanced', 'decomposed', 'adaptive', 'paper_optimized', 'parallel',
            'quantum_inspired', 'edge', 'speculative_thought'
        ]

    def _get_optimized_params(self, mode: str, kwargs: Dict) -> Dict:
        """Llama.cppに最適化されたモデルパラメータを返す。"""
        params = kwargs.copy()
        temp_map = {'efficient': 0.2, 'balanced': 0.5, 'decomposed': 0.4}
        if mode in temp_map and 'temperature' not in params:
            params['temperature'] = temp_map[mode]
        return params

    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        capabilities = self.standard_provider.get_capabilities()
        capabilities[ProviderCapability.ENHANCED_CALL] = True
        return capabilities