# /test_all_v2_providers.py
# タイトル: 全V2プロバイダーの総合テストスクリプト (修正版)
# 役割: 存在しない関数の呼び出しを修正し、実際に利用可能な関数を使ってプロバイダーの動作確認と性能測定を行う。

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
from dotenv import load_dotenv
load_dotenv()

# 他のモジュールをインポート
from cli.handler import CogniQuantumCLIV2Fixed # ヘルスチェック機能を持つCLIハンドラ
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
        self.v2_modes = modes_to_test or ['efficient', 'balanced', 'decomposed', 'adaptive', 'paper_optimized', 'parallel', 'quantum_inspired', 'edge']
        self.cli_handler = CogniQuantumCLIV2Fixed()

    def _get_available_providers(self) -> List[str]:
        """APIキーが設定されているなど、利用可能なプロバイダーのリストを取得する"""
        available = []
        all_providers = list_providers()
        
        # APIキーの存在チェック
        if settings.OPENAI_API_KEY: available.append('openai')
        if settings.CLAUDE_API_KEY: available.append('claude')
        if settings.GEMINI_API_KEY: available.append('gemini')
        if settings.HF_TOKEN: available.append('huggingface')
        
        # Ollamaは常にチェック対象とする
        if 'ollama' in all_providers:
            available.append('ollama')
            
        return list(set(available))

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
        self.test_results['system_info'] = {
            'timestamp': time.time(),
            'python_version': sys.version,
            'working_directory': str(project_root),
            'standard_providers': list_providers(),
            'enhanced_providers': list_enhanced_providers(),
        }
        print("✅ システム情報収集完了")

    async def check_all_providers_health(self):
        """全プロバイダーの健全性チェック"""
        print("\n🏥 プロバイダー健全性チェック中...")
        health_results: Dict[str, Any] = {'providers': {}}
        available_count = 0
        enhanced_v2_count = 0

        for provider_name in list_providers():
            health_results['providers'][provider_name] = {}
            # 標準プロバイダーのチェック
            std_health = check_provider_health(provider_name, enhanced=False)
            health_results['providers'][provider_name]['standard'] = std_health
            if std_health['available']:
                available_count += 1
                
            # V2拡張プロバイダーのチェック
            if provider_name in list_enhanced_providers()['v2']:
                enh_health = check_provider_health(provider_name, enhanced=True)
                health_results['providers'][provider_name]['enhanced_v2'] = enh_health
                if enh_health['available']:
                    enhanced_v2_count += 1
        
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
        
        for provider_name in self.providers_to_test:
            if provider_name not in list_enhanced_providers()['v2']:
                continue

            print(f"\n🔍 {provider_name} V2機能テスト開始...")
            provider_results: Dict[str, Any] = {'modes_tested': {}, 'errors': []}
            
            for mode in self.v2_modes:
                try:
                    result = await self.test_provider_mode(provider_name, mode)
                    provider_results['modes_tested'][mode] = result
                    status = "✅ 成功" if result['success'] else f"❌ 失敗: {result.get('error', '不明')}"
                    print(f"   - {mode}モード: {status}")
                except Exception as e:
                    provider_results['errors'].append(f"{mode}モードテスト中にエラー: {e}")
                    print(f"   - {mode}モード: ⚠️  エラー ({e})")
            
            self.test_results['v2_features'][provider_name] = provider_results

    async def test_provider_mode(self, provider_name: str, mode: str) -> Dict[str, Any]:
        """特定のプロバイダーとモードをテスト"""
        prompts = {
            'efficient': "1+1は?",
            'balanced': "機械学習とは何かを簡潔に説明して。",
            'decomposed': "持続可能な都市交通システムの設計案を考えて。",
            'adaptive': "太陽光発電のメリットとデメリットは？",
            'paper_optimized': "AIの倫理について論じて。",
            'parallel': "量子コンピュータの将来性について。",
            'quantum_inspired': "意識の謎について、複数の視点から考察して。",
            'edge': "色を混ぜるとどうなる？"
        }
        prompt = prompts.get(mode, "一般的なテストプロンプトです。")
        
        try:
            # V2拡張プロバイダーを直接取得
            provider = get_provider(provider_name, enhanced=True)
            start_time = time.time()
            response = await provider.call(prompt, mode=mode, force_v2=True)
            execution_time = time.time() - start_time
            
            return {
                'success': not response.get('error'),
                'error': response.get('error'),
                'response_length': len(response.get('text', '')),
                'execution_time': execution_time,
                'version': response.get('version'),
                'v2_improvements': response.get('paper_based_improvements', {}),
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def run_performance_tests(self):
        """パフォーマンステスト"""
        print("\n⚡ パフォーマンステスト中...")
        self.test_results['performance'] = {}
        # この機能は簡略化のため、今回は実行しない
        print("   (パフォーマンステストは今回はスキップします)")


    def generate_report(self):
        """最終レポートの生成"""
        print("\n" + "=" * 60)
        print("📊 総合テスト結果レポート")
        print("=" * 60)
        
        # サマリー表示
        health_summary = self.test_results.get('health_check', {}).get('summary', {})
        print(f"\n🏥 健全性: {health_summary.get('available', 0)}/{health_summary.get('total_checked', 0)} のプロバイダーが利用可能")
        print(f"   - V2拡張: {health_summary.get('enhanced_v2', 0)}/{len(list_enhanced_providers()['v2'])} が利用可能")
        
        v2_features = self.test_results.get('v2_features', {})
        if v2_features:
            print("\n🧪 V2機能テスト結果:")
            for provider, results in v2_features.items():
                success_count = sum(1 for res in results['modes_tested'].values() if res['success'])
                print(f"   - {provider}: {success_count}/{len(results['modes_tested'])} モード成功")

        self.save_json_report()

    def save_json_report(self):
        """JSONレポートの保存"""
        try:
            report_file = project_root / "v2_test_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 詳細レポートを '{report_file}' に保存しました。")
        except Exception as e:
            print(f"\n❌ レポートの保存に失敗しました: {e}")

async def main():
    """メイン実行関数"""
    import argparse
    parser = argparse.ArgumentParser(description="CogniQuantum V2プロバイダー総合テスト")
    parser.add_argument("--providers", nargs='+', help="テストするプロバイダーを指定 (例: openai ollama)")
    parser.add_argument("--modes", nargs='+', help="テストするモードを指定 (例: efficient balanced)")
    args = parser.parse_args()
    
    tester = V2ProviderTester(providers_to_test=args.providers, modes_to_test=args.modes)
    await tester.run_comprehensive_tests()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())