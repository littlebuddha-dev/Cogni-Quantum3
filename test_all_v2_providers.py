# /test_all_v2_providers.py
# タイトル: 全V2プロバイダーの総合テストスクリプト (現在の実装対応版)
# 役割: 現在のプロジェクト構造に合わせて修正されたプロバイダーの動作確認と性能測定を行う。

import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, Any, List
from pathlib import Path

# パスの設定
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# dotenvを先に読み込む
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenvがインストールされていません。環境変数の読み込みをスキップします。")

# 他のモジュールをインポート
from llm_api.providers import get_provider, list_providers, list_enhanced_providers, check_provider_health
from llm_api.config import settings

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class V2ProviderTester:
    """V2プロバイダーの総合テスター"""
    
    def __init__(self, providers_to_test=None, modes_to_test=None):
        self.test_results: Dict[str, Any] = {}
        # 利用可能なプロバイダーを動的に設定
        self.available_providers = self._get_available_providers()
        self.providers_to_test = providers_to_test or self.available_providers
        self.v2_modes = modes_to_test or ['efficient', 'balanced', 'decomposed', 'adaptive', 'parallel', 'quantum_inspired', 'edge', 'speculative_thought']

    def _get_available_providers(self) -> List[str]:
        """APIキーが設定されているなど、利用可能なプロバイダーのリストを取得する"""
        available = []
        all_providers = list_providers()
        
        # APIキーの存在と有効性をチェック
        api_key_checks = {
            'openai': settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.startswith('sk-') and len(settings.OPENAI_API_KEY) > 20,
            'claude': settings.CLAUDE_API_KEY and settings.CLAUDE_API_KEY.startswith('sk-ant-') and len(settings.CLAUDE_API_KEY) > 20,
            'gemini': settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.startswith('AIza') and len(settings.GEMINI_API_KEY) > 20,
            'huggingface': settings.HF_TOKEN and settings.HF_TOKEN.startswith('hf_') and len(settings.HF_TOKEN) > 20,
        }
        
        for provider, has_valid_key in api_key_checks.items():
            if has_valid_key and provider in all_providers:
                available.append(provider)
        
        # Ollamaは常にチェック対象とする（ローカルで動作）
        if 'ollama' in all_providers:
            available.append('ollama')
            
        # LlamaCppもローカルで動作する可能性がある（設定されている場合のみ）
        if 'llamacpp' in all_providers and settings.LLAMACPP_API_BASE_URL:
            available.append('llamacpp')
            
        return list(set(available))

    async def check_ollama_connection(self) -> tuple[bool, List[str]]:
        """Ollamaサーバーの接続確認とモデル一覧取得"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                try:
                    response = await client.get("http://localhost:11434/api/tags")
                    if response.status_code == 200:
                        data = response.json()
                        models = [model['name'] for model in data.get('models', [])]
                        return True, models
                    else:
                        return False, []
                except Exception:
                    return False, []
        except ImportError:
            return False, []

    async def run_comprehensive_tests(self):
        """総合テストの実行"""
        print("🚀 CogniQuantum V2 プロバイダー総合テスト開始")
        print(f"🔬 テスト対象プロバイダー: {self.providers_to_test}")
        print(f"🕹️ テスト対象モード: {self.v2_modes}")
        print("=" * 60)
        
        await self.collect_system_info()
        await self.check_all_providers_health()
        await self.test_v2_features()
        await self.run_performance_tests()
        self.generate_report()

    async def collect_system_info(self):
        """システム情報の収集"""
        print("\n📊 システム情報を収集中...")
        
        # APIキーの状態チェック
        api_key_status = {
            'OpenAI': bool(settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.startswith('sk-')),
            'Claude': bool(settings.CLAUDE_API_KEY and settings.CLAUDE_API_KEY.startswith('sk-ant-')),
            'Gemini': bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.startswith('AIza')),
            'HuggingFace': bool(settings.HF_TOKEN and settings.HF_TOKEN.startswith('hf_')),
        }
        
        # Ollama接続チェック
        ollama_connected, ollama_models = await self.check_ollama_connection()
        
        # LlamaCpp接続チェック
        llamacpp_available = bool(settings.LLAMACPP_API_BASE_URL)
        
        self.test_results['system_info'] = {
            'timestamp': time.time(),
            'python_version': sys.version,
            'working_directory': str(project_root),
            'standard_providers': list_providers(),
            'enhanced_providers': list_enhanced_providers(),
            'api_key_status': api_key_status,
            'ollama_connected': ollama_connected,
            'ollama_models': ollama_models,
            'llamacpp_configured': llamacpp_available,
            'available_providers': self.available_providers,
        }
        print("✅ システム情報収集完了")
        
        # APIキー状態を表示
        print("🔑 APIキー状態:")
        for service, has_key in api_key_status.items():
            status = "✅ 設定済み" if has_key else "❌ 未設定"
            print(f"   - {service}: {status}")
        
        if ollama_connected:
            print(f"🦙 Ollama接続: ✅ ({len(ollama_models)}モデル利用可能)")
        else:
            print("🦙 Ollama接続: ❌")
            
        if llamacpp_available:
            print(f"🔥 LlamaCpp設定: ✅ ({settings.LLAMACPP_API_BASE_URL})")
        else:
            print("🔥 LlamaCpp設定: ❌")
        
        print(f"🎯 テスト可能プロバイダー: {self.available_providers}")

    async def check_all_providers_health(self):
        """全プロバイダーの健全性チェック"""
        print("\n🏥 プロバイダー健全性チェック中...")
        health_results: Dict[str, Any] = {'providers': {}}
        available_count = 0
        enhanced_v2_count = 0

        for provider_name in list_providers():
            health_results['providers'][provider_name] = {}
            
            # 標準プロバイダーのチェック
            try:
                std_health = check_provider_health(provider_name, enhanced=False)
                health_results['providers'][provider_name]['standard'] = std_health
                if std_health['available']:
                    available_count += 1
                    print(f"   ✅ {provider_name} (標準)")
                else:
                    print(f"   ❌ {provider_name} (標準): {std_health['reason']}")
            except Exception as e:
                health_results['providers'][provider_name]['standard'] = {'available': False, 'reason': str(e)}
                print(f"   ⚠️ {provider_name} (標準): エラー {e}")
                
            # V2拡張プロバイダーのチェック
            enhanced_v2_providers = list_enhanced_providers().get('v2', [])
            if provider_name in enhanced_v2_providers:
                try:
                    enh_health = check_provider_health(provider_name, enhanced=True)
                    health_results['providers'][provider_name]['enhanced_v2'] = enh_health
                    if enh_health['available']:
                        enhanced_v2_count += 1
                        print(f"   ✅ {provider_name} (V2拡張)")
                    else:
                        print(f"   ❌ {provider_name} (V2拡張): {enh_health['reason']}")
                except Exception as e:
                    health_results['providers'][provider_name]['enhanced_v2'] = {'available': False, 'reason': str(e)}
                    print(f"   ⚠️ {provider_name} (V2拡張): エラー {e}")
        
        health_results['summary'] = {
            'total_checked': len(list_providers()),
            'available': available_count,
            'enhanced_v2': enhanced_v2_count
        }
        self.test_results['health_check'] = health_results
        print("✅ 健全性チェック完了")

    async def test_v2_features(self):
        """V2機能の詳細テスト"""
        print("\n🧪 V2機能テスト中...")
        self.test_results['v2_features'] = {}
        
        enhanced_v2_providers = list_enhanced_providers().get('v2', [])
        
        # 利用可能なプロバイダーのみをテスト
        testable_providers = [p for p in self.providers_to_test if p in self.available_providers]
        
        if not testable_providers:
            print("⚠️ テスト可能なプロバイダーがありません。APIキーの設定またはOllamaの起動を確認してください。")
            return
        
        print(f"🎯 実際にテストするプロバイダー: {testable_providers}")
        
        for provider_name in testable_providers:
            if provider_name not in enhanced_v2_providers:
                print(f"⚠️ {provider_name}: V2拡張が利用できません。スキップします。")
                continue

            print(f"\n🔍 {provider_name} V2機能テスト開始...")
            provider_results: Dict[str, Any] = {'modes_tested': {}, 'errors': []}
            
            # プロバイダー固有の事前チェック
            if provider_name == 'ollama':
                ollama_ok, models = await self.check_ollama_connection()
                if not ollama_ok:
                    print(f"   ❌ Ollamaサーバーに接続できません。スキップします。")
                    provider_results['errors'].append("Ollamaサーバー接続失敗")
                    self.test_results['v2_features'][provider_name] = provider_results
                    continue
                elif not models:
                    print(f"   ❌ Ollamaにモデルがありません。スキップします。")
                    provider_results['errors'].append("Ollamaモデル不在")
                    self.test_results['v2_features'][provider_name] = provider_results
                    continue
            elif provider_name == 'llamacpp':
                if not settings.LLAMACPP_API_BASE_URL:
                    print(f"   ❌ LlamaCpp設定が不完全です。スキップします。")
                    provider_results['errors'].append("LlamaCpp設定不完全")
                    self.test_results['v2_features'][provider_name] = provider_results
                    continue
            
            for mode in self.v2_modes:
                try:
                    result = await self.test_provider_mode(provider_name, mode)
                    provider_results['modes_tested'][mode] = result
                    status = "✅ 成功" if result['success'] else f"❌ 失敗: {result.get('error', '不明')[:100]}..."
                    print(f"   - {mode}モード: {status}")
                    
                    # 少し待機してサーバーに負荷をかけすぎないようにする
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    error_msg = f"{mode}モードテスト中にエラー: {str(e)[:100]}..."
                    provider_results['errors'].append(error_msg)
                    print(f"   - {mode}モード: ⚠️ エラー ({str(e)[:50]}...)")
            
            self.test_results['v2_features'][provider_name] = provider_results

    async def test_provider_mode(self, provider_name: str, mode: str) -> Dict[str, Any]:
        """特定のプロバイダーとモードをテスト"""
        prompts = {
            'efficient': "1+1は?",
            'balanced': "機械学習とは何かを簡潔に説明して。",
            'decomposed': "持続可能な都市交通システムの設計案を考えて。",
            'adaptive': "太陽光発電のメリットとデメリットは？",
            'parallel': "量子コンピュータの将来性について。",
            'quantum_inspired': "意識の謎について、複数の視点から考察して。",
            'edge': "色を混ぜるとどうなる？",
            'speculative_thought': "AIの未来について思考実験してください。"
        }
        prompt = prompts.get(mode, "一般的なテストプロンプトです。")
        
        try:
            # V2拡張プロバイダーを直接取得
            provider = get_provider(provider_name, enhanced=True)
            start_time = time.time()
            
            # モデル選択（Ollamaの場合）
            call_kwargs = {'mode': mode}
            if provider_name == 'ollama':
                # 利用可能なモデルを取得
                _, models = await self.check_ollama_connection()
                if models:
                    # 最初のモデルを使用
                    call_kwargs['model'] = models[0]
            
            response = await provider.call(prompt, **call_kwargs)
            execution_time = time.time() - start_time
            
            return {
                'success': not response.get('error'),
                'error': response.get('error'),
                'response_length': len(response.get('text', '')),
                'execution_time': execution_time,
                'version': response.get('version'),
                'v2_improvements': response.get('paper_based_improvements', {}),
                'model_used': call_kwargs.get('model', 'default'),
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def run_performance_tests(self):
        """パフォーマンステスト"""
        print("\n⚡ パフォーマンステスト中...")
        
        # 利用可能なプロバイダーのみをテスト
        testable_providers = [p for p in self.providers_to_test if p in self.available_providers]
        
        if not testable_providers:
            print("⚠️ パフォーマンステスト可能なプロバイダーがありません。")
            self.test_results['performance'] = {}
            return
        
        # 簡単なパフォーマンステストを実行
        performance_results = {}
        
        # 各プロバイダーで同じプロンプトの実行時間を測定
        test_prompt = "Pythonとは何ですか？簡潔に説明してください。"
        
        for provider_name in testable_providers:
            if provider_name not in list_enhanced_providers().get('v2', []):
                continue
                
            # プロバイダー固有の事前チェック
            if provider_name == 'ollama':
                ollama_ok, models = await self.check_ollama_connection()
                if not ollama_ok or not models:
                    continue
            elif provider_name == 'llamacpp' and not settings.LLAMACPP_API_BASE_URL:
                continue
            
            try:
                provider = get_provider(provider_name, enhanced=True)
                
                # 3回実行して平均時間を計測
                times = []
                for i in range(3):
                    start_time = time.time()
                    call_kwargs = {'mode': 'balanced'}
                    if provider_name == 'ollama':
                        _, models = await self.check_ollama_connection()
                        if models:
                            call_kwargs['model'] = models[0]
                    
                    response = await provider.call(test_prompt, **call_kwargs)
                    execution_time = time.time() - start_time
                    
                    if not response.get('error'):
                        times.append(execution_time)
                    
                    await asyncio.sleep(1)  # 1秒待機
                
                if times:
                    performance_results[provider_name] = {
                        'avg_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'runs': len(times)
                    }
                    print(f"   {provider_name}: 平均 {performance_results[provider_name]['avg_time']:.2f}秒")
                
            except Exception as e:
                print(f"   {provider_name}: パフォーマンステストエラー ({str(e)[:50]}...)")
        
        self.test_results['performance'] = performance_results
        print("✅ パフォーマンステスト完了")

    def generate_report(self):
        """最終レポートの生成"""
        print("\n" + "=" * 60)
        print("📊 総合テスト結果レポート")
        print("=" * 60)
        
        # サマリー表示
        health_summary = self.test_results.get('health_check', {}).get('summary', {})
        print(f"\n🏥 健全性: {health_summary.get('available', 0)}/{health_summary.get('total_checked', 0)} のプロバイダーが利用可能")
        print(f"   - V2拡張: {health_summary.get('enhanced_v2', 0)}/{len(list_enhanced_providers().get('v2', []))} が利用可能")
        
        v2_features = self.test_results.get('v2_features', {})
        if v2_features:
            print("\n🧪 V2機能テスト結果:")
            for provider, results in v2_features.items():
                success_count = sum(1 for res in results['modes_tested'].values() if res['success'])
                total_modes = len(results['modes_tested'])
                print(f"   - {provider}: {success_count}/{total_modes} モード成功")
                
                # エラーがあれば表示
                if results.get('errors'):
                    for error in results['errors']:
                        print(f"     ⚠️ {error}")

        # パフォーマンステスト結果
        performance = self.test_results.get('performance', {})
        if performance:
            print("\n⚡ パフォーマンステスト結果:")
            for provider, perf_data in performance.items():
                print(f"   - {provider}: 平均応答時間 {perf_data['avg_time']:.2f}秒")

        self.save_json_report()

    def save_json_report(self):
        """JSONレポートの保存"""
        try:
            report_file = project_root / "v2_test_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n💾 詳細レポートを '{report_file}' に保存しました。")
        except Exception as e:
            print(f"\n❌ レポートの保存に失敗しました: {e}")

async def main():
    """メイン実行関数"""
    import argparse
    parser = argparse.ArgumentParser(description="CogniQuantum V2プロバイダー総合テスト")
    parser.add_argument("--providers", nargs='+', help="テストするプロバイダーを指定 (例: openai ollama)")
    parser.add_argument("--modes", nargs='+', help="テストするモードを指定 (例: efficient balanced)")
    parser.add_argument("--skip-performance", action="store_true", help="パフォーマンステストをスキップ")
    args = parser.parse_args()
    
    tester = V2ProviderTester(providers_to_test=args.providers, modes_to_test=args.modes)
    
    if args.skip_performance:
        # パフォーマンステストをスキップする簡易版
        original_run_performance_tests = tester.run_performance_tests
        async def skip_performance():
            print("\n⚡ パフォーマンステスト: スキップされました")
            tester.test_results['performance'] = {}
        tester.run_performance_tests = skip_performance
    
    await tester.run_comprehensive_tests()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())