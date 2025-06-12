# /quick_test_v2.py
"""
CogniQuantum V2 クイックテストスクリプト
問題の診断と基本動作確認用
"""
import asyncio
import json
import logging
import os
import sys
from typing import Dict, Any

# パスの設定
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def check_ollama_status():
    """Ollamaの状態をチェック"""
    print("🔍 Ollama状態チェック中...")
    
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    print(f"✅ Ollamaサーバー: 接続OK")
                    print(f"📦 利用可能モデル: {models}")
                    return True, models
                else:
                    print(f"❌ Ollamaサーバー: HTTP {response.status_code}")
                    return False, []
            except Exception as e:
                print(f"❌ Ollamaサーバー: 接続失敗 ({e})")
                return False, []
    except ImportError:
        print("❌ httpx がインストールされていません")
        return False, []

async def test_basic_functionality():
    """基本機能のテスト"""
    print("\n🧪 基本機能テスト中...")
    
    try:
        from llm_api.providers import list_providers, list_enhanced_providers
        
        standard = list_providers()
        enhanced = list_enhanced_providers()
        
        print(f"📋 標準プロバイダー: {standard}")
        print(f"📋 拡張プロバイダー V1: {enhanced.get('v1', [])}")
        print(f"📋 拡張プロバイダー V2: {enhanced.get('v2', [])}")
        
        return True
    except Exception as e:
        print(f"❌ 基本機能テスト失敗: {e}")
        return False

async def test_provider_creation():
    """プロバイダー作成テスト"""
    print("\n🏭 プロバイダー作成テスト中...")
    success = True
    
    try:
        from llm_api.providers import get_provider
        
        # 標準プロバイダーテスト
        try:
            provider = get_provider('ollama', enhanced=False)
            print("✅ 標準Ollamaプロバイダー: 作成成功")
        except Exception as e:
            print(f"❌ 標準Ollamaプロバイダー: {e}")
            success = False
        
        # 拡張プロバイダーテスト（V2優先）
        try:
            # 'prefer_v2'引数を削除して修正
            provider = get_provider('ollama', enhanced=True)
            print("✅ 拡張Ollamaプロバイダー: 作成成功")
        except Exception as e:
            print(f"❌ 拡張Ollamaプロバイダー: {e}")
            success = False
        
        return success
    except Exception as e:
        print(f"❌ プロバイダー作成テストのインポート中に失敗: {e}")
        return False

async def test_simple_call():
    """シンプルな呼び出しテスト"""
    print("\n📞 シンプル呼び出しテスト中...")
    
    # このテストはOllamaが利用可能な場合にのみ実行
    ollama_ok, models = await check_ollama_status()
    if not ollama_ok or not models:
        print("⚠️  Ollama利用不可のため、呼び出しテストをスキップ")
        # スキップは失敗ではないのでTrueを返す
        return True
    
    try:
        from llm_api.providers import get_provider
        
        selected_model = models[0].split(':')[0]
        print(f"🎯 使用モデル: {selected_model}")
        
        provider = get_provider('ollama', enhanced=False)
        response = await provider.call(
            "Hello, respond with just 'OK'",
            model=selected_model
        )
        
        if response.get('text'):
            print(f"✅ 呼び出し成功: {response['text'][:50]}...")
            return True
        else:
            print(f"❌ 呼び出し失敗: {response.get('error', '空の応答')}")
            return False
            
    except Exception as e:
        print(f"❌ 呼び出しテスト失敗: {e}")
        return False

def show_setup_guide():
    """セットアップガイドの表示"""
    print("""
🔧 セットアップガイド:

1. Ollamaのインストールと起動:
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama serve

2. モデルのプル:
   ollama pull gemma3:latest    # 推奨
   ollama pull llama3.1      # 軽量版

3. 確認:
   ollama list

4. テスト再実行:
   python quick_test_v2.py
""")

def show_troubleshooting():
    """トラブルシューティング情報"""
    print("""
🚨 トラブルシューティング:

【Ollamaサーバーが起動しない】
- ポート11434が使用されていないか確認
- 別ターミナルで: ollama serve

【モデルが見つからない】
- ollama list で確認
- ollama pull <モデル名> でダウンロード

【プロバイダーエラー】
- Python依存関係を確認: pip install -r requirements.txt
- パスの確認: export PYTHONPATH=.

【その他】
- ログレベル変更: export LOG_LEVEL=DEBUG
- 詳細情報: python quick_test_v2.py --verbose
""")

async def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CogniQuantum V2 クイックテスト")
    parser.add_argument("--verbose", action="store_true", help="詳細ログ")
    parser.add_argument("--setup-guide", action="store_true", help="セットアップガイド表示")
    parser.add_argument("--troubleshooting", action="store_true", help="トラブルシューティング表示")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.setup_guide:
        show_setup_guide()
        return
    
    if args.troubleshooting:
        show_troubleshooting()
        return
    
    print("🚀 CogniQuantum V2 クイックテスト開始")
    print("=" * 50)
    
    tests = [
        ("Ollama状態チェック", check_ollama_status),
        ("基本機能テスト", test_basic_functionality),
        ("プロバイダー作成テスト", test_provider_creation),
        ("シンプル呼び出しテスト", test_simple_call)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            # check_ollama_statusはタプルを返すので特別扱い
            if test_name == "Ollama状態チェック":
                result, _ = await test_func()
                results.append((test_name, result))
            else:
                result = await test_func()
                results.append((test_name, result))
        except Exception as e:
            logger.error(f"{test_name}でエラー: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 テスト結果サマリー:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n📈 総合結果: {passed}/{total} テスト合格")
    
    if passed == total:
        print("🎉 全テスト合格！システムは正常に動作します。")
        print("\n次のコマンドで実際のテストを行ってください:")
        print("python fetch_llm_v2.py ollama 'Hello' --mode simple")
    elif not results[0][1]: # Ollamaチェックが失敗した場合
        print("😞 Ollamaの接続に失敗しました。セットアップガイドを確認してください。")
        print("python quick_test_v2.py --setup-guide")
    else:
        print("⚠️  一部のテストが失敗しました。")
        print("python quick_test_v2.py --troubleshooting")

if __name__ == "__main__":
    asyncio.run(main())