# /llm_api/providers/base.py
# タイトル: Abstract Base Classes for LLM Providers with Corrected Initialization Order
# 役割: 全てのLLMプロバイダーの基底クラスを定義する。EnhancedLLMProviderのコンストラクタの処理順序を修正し、初期化時のエラーを解決する。

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict

# 循環参照を避けるため、型チェック時のみインポート
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..cogniquantum.system import CogniQuantumSystemV2
    from ..cogniquantum.enums import ComplexityRegime


logger = logging.getLogger(__name__)

class ProviderCapability(Enum):
    """プロバイダーの機能を定義するEnum"""
    STANDARD_CALL = "standard_call"
    ENHANCED_CALL = "enhanced_call"
    STREAMING = "streaming"
    SYSTEM_PROMPT = "system_prompt"
    TOOLS = "tools"
    JSON_MODE = "json_mode"

class LLMProvider(ABC):
    """
    全てのLLMプロバイダーの抽象基底クラス（ABC）
    """
    def __init__(self):
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
        self.capabilities = self._get_default_capabilities()

    def _get_default_capabilities(self) -> Dict[ProviderCapability, bool]:
        """capabilitiesのデフォルト値を生成する。実装クラスでオーバーライド推奨。"""
        if hasattr(self, 'get_capabilities'):
            try:
                return self.get_capabilities()
            except TypeError:
                 pass
        return {cap: False for cap in ProviderCapability}


    @abstractmethod
    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        """プロバイダーの機能を定義した辞書を返す。"""
        pass

    async def call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """
        メインの呼び出しメソッド。
        拡張機能を使うべきか判断し、enhanced_callまたはstandard_callに処理を振り分ける。
        """
        use_enhancement = self.get_capabilities().get(ProviderCapability.ENHANCED_CALL, False) and \
                          self.should_use_enhancement(prompt, **kwargs)

        if use_enhancement:
            if hasattr(self, 'enhanced_call') and callable(self.enhanced_call):
                logger.debug(f"プロバイダー '{self.provider_name}' の enhanced_call を呼び出します。")
                enhanced_call_method = getattr(self, "enhanced_call")
                return await enhanced_call_method(prompt, system_prompt, **kwargs)
            else:
                 logger.warning(f"'{self.provider_name}' はENHANCED_CALLケイパビリティを持つと報告しましたが、enhanced_callメソッドが見つかりません。")

        logger.debug(f"プロバイダー '{self.provider_name}' の standard_call を呼び出します。")
        return await self.standard_call(prompt, system_prompt, **kwargs)

    @abstractmethod
    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """
        標準的な（拡張機能なしの）LLM呼び出し。全ての具象プロバイダーで実装必須。
        """
        pass

    @abstractmethod
    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        """
        拡張機能（enhanced_call）を使用すべきかどうかを判断する。
        全ての具象プロバイダーで実装必須。
        """
        pass

class EnhancedLLMProvider(LLMProvider):
    """
    標準プロバイダーをラップし、CogniQuantum V2システムを介して追加機能を提供する拡張プロバイダーの基底クラス。
    """
    def __init__(self, standard_provider: LLMProvider):
        if not isinstance(standard_provider, LLMProvider):
             raise TypeError("EnhancedLLMProviderには有効なstandard_providerインスタンスが必要です。")
        self.standard_provider = standard_provider # ★ 先に standard_provider を設定

        # 親クラスの__init__を呼び出す。これにより self.capabilities が設定される。
        # get_capabilities() が呼ばれるが、この時点で self.standard_provider は存在する。
        super().__init__()

        # 親クラスの__init__ではクラス名からprovider_nameが設定されるため、
        # ラップしているプロバイダーの名前に上書きする。
        self.provider_name = standard_provider.provider_name

    def _determine_force_regime(self, mode: str) -> 'ComplexityRegime' or None:
        """モード文字列から強制する複雑性レジームを決定する。"""
        # このインポートは実行時にのみ行われる
        from ..cogniquantum.enums import ComplexityRegime
        if mode in ['efficient', 'edge']: return ComplexityRegime.LOW
        if mode == 'balanced': return ComplexityRegime.MEDIUM
        if mode == 'decomposed': return ComplexityRegime.HIGH
        return None

    @abstractmethod
    def _get_optimized_params(self, mode: str, kwargs: Dict) -> Dict:
        """
        プロバイダーとモードに固有のモデルパラメータを最適化する。
        具象クラスで必ず実装する必要がある。
        """
        pass

    async def enhanced_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """
        CogniQuantumシステムを利用した拡張呼び出しの共通ロジック。
        """
        # 循環参照を避けるため、実行時にインポート
        from ..cogniquantum.system import CogniQuantumSystemV2

        try:
            mode = kwargs.get('mode', 'adaptive')
            logger.info(f"{self.provider_name} V2拡張呼び出し実行 (モード: {mode})")

            force_regime = self._determine_force_regime(mode)
            base_model_kwargs = self._get_optimized_params(mode, kwargs)

            cq_system = CogniQuantumSystemV2(self.standard_provider, base_model_kwargs)

            # CogniQuantumSystemV2.solve_problemに渡す引数をkwargsから抽出
            system_kwargs = {
                'use_rag': kwargs.get('use_rag', False),
                'knowledge_base_path': kwargs.get('knowledge_base_path'),
                'use_wikipedia': kwargs.get('use_wikipedia', False),
                'real_time_adjustment': kwargs.get('real_time_adjustment', True),
                'mode': mode
            }

            result = await cq_system.solve_problem(
                prompt,
                system_prompt=system_prompt,
                force_regime=force_regime,
                **system_kwargs
            )

            if not result.get('success'):
                error_message = result.get('error', f'CogniQuantumシステム({self.provider_name})で不明なエラーが発生しました。')
                logger.error(f"CogniQuantumシステムがエラーを返しました: {error_message}")
                return {"text": "", "error": error_message}

            paper_based_improvements = result.get('complexity_analysis', {})
            paper_based_improvements.update(result.get('v2_improvements', {}))

            return {
                'text': result.get('final_solution', ''),
                'image_url': result.get('image_url'),
                'model': base_model_kwargs.get('model', 'default'),
                'usage': {},
                'error': None,
                'version': 'v2',
                'paper_based_improvements': paper_based_improvements
            }
        except Exception as e:
            logger.error(f"{self.provider_name} V2拡張プロバイダーで予期せぬエラー: {e}", exc_info=True)
            return {"text": "", "error": str(e)}