# /llm_api/providers/enhanced_ollama_v2.py
# タイトル: Refactored EnhancedOllamaProviderV2 with Corrected Default Model
# 役割: OllamaプロバイダーにCogniQuantum V2の機能を提供する。デフォルトモデルをユーザー環境に合わせて修正。

import logging
from typing import Any, Dict

from .base import EnhancedLLMProvider, ProviderCapability
from ..utils.helper_functions import get_model_family

logger = logging.getLogger(__name__)

class EnhancedOllamaProviderV2(EnhancedLLMProvider):
    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        return await self.standard_provider.standard_call(prompt, system_prompt, **kwargs)

    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        return kwargs.get('force_v2', False) or kwargs.get('mode', 'simple') in [
            'efficient', 'balanced', 'decomposed', 'adaptive', 'paper_optimized', 'parallel',
            'quantum_inspired', 'edge'
        ]

    def _get_optimized_params(self, mode: str, kwargs: Dict) -> Dict:
        """Ollamaに最適化されたモデルパラメータを返す。"""
        params = kwargs.copy()
        model_name = kwargs.get('model')

        if mode == 'edge' and not model_name:
            effective_model_name = 'gemma:2b'
            logger.info(f"エッジモードのため、デフォルトの軽量モデル '{effective_model_name}' を選択しました。")
        else:
            # ユーザー環境に存在する可能性が高い gemma3:latest をデフォルトにする
            effective_model_name = model_name or 'gemma3:latest'

        family = get_model_family(effective_model_name)
        logger.info(f"モデル '{effective_model_name}' (ファミリー: {family}) のパラメータを最適化中")

        if 'model' not in params:
            params['model'] = effective_model_name

        temp_map = {'efficient': 0.2, 'balanced': 0.5, 'decomposed': 0.4, 'edge': 0.3}
        if mode in temp_map and 'temperature' not in params:
            params['temperature'] = temp_map[mode]

        if family == 'llama' and 'top_p' not in params:
            params['top_p'] = 0.9
        elif family == 'qwen' and 'temperature' not in params:
            params['temperature'] = temp_map.get(mode, 0.4)

        return params

    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        capabilities = self.standard_provider.get_capabilities()
        capabilities[ProviderCapability.ENHANCED_CALL] = True
        return capabilities