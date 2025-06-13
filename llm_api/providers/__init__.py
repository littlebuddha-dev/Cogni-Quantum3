# /llm_api/providers/__init__.py
"""
LLMプロバイダーの動的ロードとファクトリー機能
"""
import importlib
import inspect
import logging
import os
import pkgutil
from typing import Dict, List, Type

from .base import LLMProvider, EnhancedLLMProvider

logger = logging.getLogger(__name__)

# プロバイダーを格納するグローバル辞書
standard_providers: Dict[str, Type[LLMProvider]] = {}
enhanced_providers: Dict[str, set] = {"v2": set(), "all": set()}
_initialized = False

def _initialize_providers():
    """プロバイダーモジュールを動的にインポートして初期化する"""
    global _initialized
    if _initialized:
        return

    logger.info("プロバイダーモジュール初期化開始")
    
    # 標準プロバイダーのロード
    package_path = os.path.dirname(__file__)
    for _, name, _ in pkgutil.iter_modules([package_path]):
        if name.startswith('enhanced_') or name == 'base':
            continue
        
        try:
            module = importlib.import_module(f".{name}", package=__name__)
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, LLMProvider) and 
                    obj is not LLMProvider and 
                    obj is not EnhancedLLMProvider and
                    not obj.__name__.startswith('Enhanced')):
                    provider_name = name.lower()
                    standard_providers[provider_name] = obj
                    logger.debug(f"標準プロバイダー '{provider_name}' を登録: {obj.__name__}")
        except Exception as e:
            logger.warning(f"プロバイダーモジュール '{name}' のロードに失敗: {e}")

    # 拡張プロバイダーのロード (V2のみ)
    for _, name, _ in pkgutil.iter_modules([package_path]):
        if name.startswith('enhanced_') and name.endswith('_v2'):
            provider_name = name.replace('enhanced_', '').replace('_v2', '')
            if provider_name in standard_providers:
                enhanced_providers["v2"].add(provider_name)
                enhanced_providers["all"].add(provider_name)
                logger.debug(f"V2拡張プロバイダー '{provider_name}' を発見")

    _initialized = True
    logger.info("プロバイダーモジュール初期化完了")
    logger.info(f"標準プロバイダー: {sorted(standard_providers.keys())}")
    logger.info(f"拡張プロバイダー V2: {sorted(enhanced_providers['v2'])}")


def list_providers() -> List[str]:
    _initialize_providers()
    return sorted(list(standard_providers.keys()))

def list_enhanced_providers() -> Dict[str, List[str]]:
    _initialize_providers()
    return {
        "v2": sorted(list(enhanced_providers["v2"])),
    }

def get_provider(name: str, enhanced: bool = False) -> LLMProvider:
    """
    指定された名前のプロバイダークラスのインスタンスを取得する。
    V2アーキテクチャに合わせ、V1の概念を削除。
    """
    _initialize_providers()
    
    if enhanced:
        logger.info(f"V2拡張プロバイダーを選択: {name}")
        provider_class = _get_enhanced_provider_class(name)
        # 拡張プロバイダーは標準プロバイダーを内部で利用する
        standard_provider = get_provider(name, enhanced=False)
        return provider_class(standard_provider)
    else:
        logger.info(f"標準プロバイダーを選択: {name}")
        provider_class = _get_standard_provider_class(name)
        return provider_class()

def _get_standard_provider_class(name: str):
    """標準プロバイダークラスを取得する。"""
    if name not in standard_providers:
        available = list(standard_providers.keys())
        raise ValueError(f"標準プロバイダー '{name}' が見つかりません。利用可能: {available}")
    return standard_providers[name]

def _get_enhanced_provider_class(name: str):
    """
    V2拡張プロバイダークラスを取得する。(堅牢版)
    モジュール内を検査し、EnhancedLLMProviderのサブクラスを動的に見つけ出す。
    """
    if name not in enhanced_providers['v2']:
        available = list_enhanced_providers()
        error_msg = f"V2拡張プロバイダー '{name}' が見つかりません。利用可能: {available['v2']}"
        raise ValueError(error_msg)

    try:
        module_name = f".enhanced_{name}_v2"
        module = importlib.import_module(module_name, package='llm_api.providers')
        
        # モジュール内のすべてのクラスを検査
        for class_name, obj in inspect.getmembers(module, inspect.isclass):
            # EnhancedLLMProviderのサブクラスで、ベースクラス自身ではないものを探す
            if issubclass(obj, EnhancedLLMProvider) and obj is not EnhancedLLMProvider:
                logger.debug(f"動的探索によりクラス '{class_name}' を発見")
                return obj # 発見したクラスを返す
        
        # ループを抜けてしまった場合（クラスが見つからない）
        raise AttributeError(f"モジュール '{module_name}' 内に EnhancedLLMProvider を継承したクラスが見つかりません。")

    except ImportError as e:
        raise ImportError(f"V2拡張プロバイダーモジュール '{name}' のインポートに失敗しました: {e}") from e


def check_provider_health(provider_name: str, enhanced: bool) -> Dict:
    """プロバイダーの健全性チェック"""
    _initialize_providers()
    try:
        get_provider(provider_name, enhanced=enhanced)
        return {'available': True, 'reason': '正常に初期化可能'}
    except (ValueError, ImportError) as e:
        return {'available': False, 'reason': str(e)}