# /cli/handler.py
"""
CLIのコアロジックを担うハンドラクラス
"""
import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from typing import Dict, Any, List, Optional

from llm_api.providers import get_provider, list_providers, list_enhanced_providers, check_provider_health
from llm_api.providers.base import ProviderCapability
from llm_api.utils.helper_functions import format_json_output, read_from_pipe_or_file

logger = logging.getLogger(__name__)

class CogniQuantumCLIV2Fixed:
    def __init__(self):
        self.session_history = deque(maxlen=100)
        
        # V2専用モード定義
        self.v2_modes = {
            'efficient', 'balanced', 'decomposed', 'adaptive', 'paper_optimized', 'parallel'
        }

    async def check_system_health(self, provider_name: str) -> Dict[str, Any]:
        """システムの健全性をチェック"""
        health_report = {
            'provider_name': provider_name,
            'timestamp': time.time(),
            'checks': {}
        }
        
        # 標準プロバイダーのチェック
        try:
            standard_health = check_provider_health(provider_name, enhanced=False)
            health_report['checks']['standard'] = standard_health
        except Exception as e:
            health_report['checks']['standard'] = {'available': False, 'error': str(e)}
        
        # 拡張プロバイダーのチェック
        try:
            enhanced_health = check_provider_health(provider_name, enhanced=True)
            health_report['checks']['enhanced_v2'] = enhanced_health
        except Exception as e:
            health_report['checks']['enhanced_v2'] = {'available': False, 'error': str(e)}
        
        # Ollamaの場合、モデル可用性もチェック
        if provider_name == 'ollama':
            health_report['checks']['ollama_models'] = await self._check_ollama_models()
        
        return health_report

    async def _check_ollama_models(self) -> Dict[str, Any]:
        """Ollamaモデルの可用性チェック"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:11434")
                if response.status_code != 200:
                    return {'server_available': False, 'error': f'Ollama server not responding. Status: {response.status_code}'}

                response = await client.get("http://localhost:11434/api/tags")
                response.raise_for_status()
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                
                # モデルが一つも読み込まれていない場合もチェック
                if not models:
                    return {
                        'server_available': True,
                        'models_loaded': False,
                        'error': 'Ollama server is running, but no models are loaded. Use `ollama pull <model_name>`.'
                    }

                return {
                    'server_available': True,
                    'models_loaded': True,
                    'models_available': models,
                    'model_count': len(models)
                }
        except (httpx.RequestError, ConnectionRefusedError):
            return {'server_available': False, 'error': 'Ollama server is not running.'}
        except Exception as e:
            return {'server_available': False, 'error': f'An unexpected error occurred: {str(e)}'}


    async def process_request_with_fallback(self, provider_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        フォールバック機能付きのリクエスト処理（V1ロジックを削除）
        """
        start_time = time.time()
        
        mode = kwargs.get('mode', 'simple')
        use_v2 = mode in self.v2_modes or kwargs.get('force_v2', False)
        
        errors_encountered = []
        
        # 戦略1: V2拡張プロバイダーを試行
        if use_v2 and not kwargs.get('no_fallback'):
            try:
                logger.info(f"V2拡張プロバイダーを試行: {provider_name}")
                provider = get_provider(provider_name, enhanced=True)
                
                enhanced_kwargs = self._enhance_kwargs_v2(kwargs)
                response = await provider.call(prompt, **enhanced_kwargs)
                
                if not response.get('error'):
                    self._update_session_history(prompt, response, provider_name, 'v2_enhanced')
                    return response
                else:
                    errors_encountered.append(f"V2拡張: {response.get('error')}")
                    
            except Exception as e:
                error_msg = f"V2拡張プロバイダーエラー: {str(e)}"
                logger.warning(error_msg, exc_info=True)
                errors_encountered.append(error_msg)
        
        # 戦略2: 標準プロバイダーを試行
        try:
            logger.info(f"標準プロバイダーを試行: {provider_name}")
            provider = get_provider(provider_name, enhanced=False)
            
            standard_kwargs = self._convert_to_standard_kwargs(kwargs)
            response = await provider.call(prompt, **standard_kwargs)
            
            if not response.get('error'):
                if errors_encountered: # フォールバックが発生した場合
                    response['fallback_used'] = True
                    response['fallback_type'] = 'standard'
                    response['original_errors'] = errors_encountered
                self._update_session_history(prompt, response, provider_name, 'standard_fallback')
                return response
            else:
                errors_encountered.append(f"標準: {response.get('error')}")
                
        except Exception as e:
            error_msg = f"標準プロバイダーエラー: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors_encountered.append(error_msg)
        
        # 全ての戦略が失敗した場合
        return {
            'text': f"全てのプロバイダー戦略が失敗しました。詳細なエラー情報は下記を参照してください。",
            'error': True,
            'execution_time': time.time() - start_time,
            'provider': provider_name,
            'all_errors': errors_encountered,
            'suggestions': self._generate_error_suggestions(provider_name, errors_encountered)
        }

    def _enhance_kwargs_v2(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """V2用のkwargs拡張"""
        enhanced = kwargs.copy()
        mode = kwargs.get('mode', 'simple')
        
        # V2フラグ追加
        enhanced['force_v2'] = True
        
        # 論文ベースの温度調整
        if 'temperature' not in enhanced:
            mode_temp_map = {
                'efficient': 0.3,
                'balanced': 0.6,
                'decomposed': 0.5,
                'adaptive': 0.6,
                'paper_optimized': 0.6
            }
            enhanced['temperature'] = mode_temp_map.get(mode, 0.7)
        
        return enhanced

    def _convert_to_standard_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """V2専用モードを標準モードに変換"""
        standard = kwargs.copy()
        mode = kwargs.get('mode', 'simple')
        
        # V2専用モードを標準モードにマッピング
        mode_conversion = {
            'efficient': 'simple',
            'balanced': 'reasoning',
            'decomposed': 'reasoning',
            'adaptive': 'reasoning',
            'paper_optimized': 'reasoning'
        }
        
        if mode in mode_conversion:
            standard['mode'] = mode_conversion[mode]
            logger.info(f"モード変換: {mode} -> {standard['mode']}")
        
        # V2専用フラグを削除
        standard.pop('force_v2', None)
        
        return standard

    def _generate_error_suggestions(self, provider_name: str, errors: List[str]) -> List[str]:
        """エラーに基づく改善提案"""
        suggestions = []
        
        # Ollamaの場合
        if provider_name == 'ollama':
            if any('not found' in error.lower() or '404' in error for error in errors):
                suggestions.extend([
                    "Ollamaサーバーが起動していることを確認: ollama serve",
                    "必要なモデルをプル: ollama pull gemma3:latest",
                    "利用可能なモデルを確認: ollama list",
                ])
        
        # API キーエラーの場合
        if any('api' in error.lower() and 'key' in error.lower() for error in errors):
            suggestions.extend([
                "API キーが正しく設定されているか確認",
                ".env ファイルの設定を確認",
                "環境変数が正しく読み込まれているか確認"
            ])
        
        # 接続エラーの場合
        if any('connection' in error.lower() or 'timeout' in error.lower() for error in errors):
            suggestions.extend([
                "ネットワーク接続を確認",
                "プロキシ設定を確認",
            ])
        
        # 一般的な提案
        suggestions.extend([
            "他のプロバイダーを試す: openai, claude",
            "シンプルなモードで試す: --mode simple",
            "詳細なログを確認: LOG_LEVEL=DEBUG python fetch_llm_v2.py ...",
        ])
        
        return suggestions

    def _update_session_history(self, prompt: str, response: dict, provider: str, strategy: str):
        """セッション履歴を更新する（戦略情報を含む）"""
        entry = {
            'timestamp': time.time(),
            'provider': provider,
            'strategy': strategy,
            'prompt': prompt,
            'response': response.get('text'),
            'metadata': {k: v for k, v in response.items() if k != 'text'},
            'success': not response.get('error', False)
        }
        self.session_history.append(entry)

    def print_system_status(self):
        """システム状態の表示"""
        print("\n=== CogniQuantum V2 システム状態 ===")
        
        # 利用可能なプロバイダー
        standard_providers = list_providers()
        enhanced_info = list_enhanced_providers()
        
        print(f"標準プロバイダー: {', '.join(standard_providers)}")
        print(f"拡張プロバイダー V2: {', '.join(enhanced_info['v2'])}")
        
        # V2専用モード
        print(f"V2専用モード: {', '.join(sorted(self.v2_modes))}")

    def print_troubleshooting_guide(self):
        """トラブルシューティングガイド"""
        guide = """
=== CogniQuantum V2 トラブルシューティング ===

【よくある問題と解決策】

1. Ollamaモデルが見つからない
   → ollama serve  (サーバー起動)
   → ollama pull gemma3:latest  (モデルダウンロード)
   → ollama list  (利用可能モデル確認)

2. V2機能が動作しない
   → python fetch_llm_v2.py --system-status  (システム状態確認)
   → --force-v2 フラグを使用
   → ログレベルをDEBUGに設定

3. API接続エラー
   → .env ファイルのAPI キー確認
   → ネットワーク接続確認
   → 他のプロバイダーで試行

4. パフォーマンス問題
   → --mode efficient で軽量実行
   → --max-tokens で制限設定

【推奨デバッグ手順】
1. システム状態確認: `python fetch_llm_v2.py [provider] --health-check`
2. 標準モードで基本動作確認
3. 段階的に拡張機能を有効化

【サポート情報収集】
LOG_LEVEL=DEBUG python fetch_llm_v2.py [provider] [prompt] --json
        """
        print(guide)