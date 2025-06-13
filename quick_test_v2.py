# /quick_test_v2.py
"""
CogniQuantum V2 クイックテストスクリプト
問題の診断と基本動作確認用（現在の実装対応版）
"""
import asyncio
import json
import logging
import os
import sys
from typing import Dict, Any, Tuple, List

# パスの設定
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def check_dependencies():
    """必要な依存関係のチェック"""
    print("🔍 依存関係チェック中...")
    
    required_packages = [
        ('httpx', 'HTTP クライアント'),
        ('pydantic', 'データ検証'),
        ('pydantic_settings', '設定管理'),
        ('asyncio', '非同期処理'),
    ]
    
    optional_packages = [
        ('python-dotenv', '環境変数読み込み'),
        ('spacy', 'NLP分析'),
        ('langdetect', '言語検出'),
        ('langchain', 'RAG機能'),
    ]
    
    missing_required = []
    missing_optional = []
    
    for package, description in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}: {description}")
        except ImportError:
            missing_required.append(package)
            print(f"❌ {package}: {description} - 不足")
    
    for package, description in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}: {description} (オプション)")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠️ {package}: {description} (オプション) - 不足")
    
    if missing_required:
        print(f"\n❌ 必須パッケージが不足しています: {', '.join(missing_required)}")
        print("次のコマンドでインストールしてください:")
        print(f"pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n⚠️ オプションパッケージが不足していますが、基本動作は可能です: {', '.join(missing_optional)}")
    
    return True

async def check_ollama_status() -> Tuple[bool, List[str]]:
    """Ollamaの状態をチェック"""
    print("\n🔍 Ollama状態チェック中...")
    
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
        print(f"📋 拡張プロバイダー V2: {enhanced.get('v2', [])}")
        
        if not standard:
            print("❌ 標準プロバイダーが見つかりません")
            return False
            
        if not enhanced.get('v2', []):
            print("⚠️ V2拡張プロバイダーが見つかりません")
        
        return True
    except Exception as e:
        print(f"❌ 基本機能テスト失敗: {e}")
        return False

async def test_config_loading():
    """設定読み込みテスト"""
    print("\n⚙️ 設定読み込みテスト中...")
    
    try:
        from llm_api.config import settings
        
        print(f"✅ 設定読み込み成功")
        print(f"   - ログレベル: {settings.LOG_LEVEL}")
        print(f"   - OllamaベースURL: {settings.OLLAMA_API_BASE_URL}")
        
        # APIキーの存在確認（値は表示しない）
        api_keys = {
            'OpenAI': bool(settings.OPENAI_API_KEY),
            'Claude': bool(settings.CLAUDE_API_KEY),
            'Gemini': bool(settings.GEMINI_API_KEY),
            'HuggingFace': bool(settings.HF_TOKEN),
        }
        
        for service, has_key in api_keys.items():
            status = "✅ 設定済み" if has_key else "❌ 未設定"
            print(f"   - {service} APIキー: {status}")
        
        return True
    except Exception as e:
        print(f"❌ 設定読み込み失敗: {e}")
        return False

async def test_provider_creation():
    """プロバイダー作成テスト"""
    print("\n🏭 プロバイダー作成テスト中...")
    success = True
    
    try:
        from llm_api.providers import get_provider, list_providers, list_enhanced_providers
        
        # 標準プロバイダーテスト
        try:
            provider = get_provider('ollama', enhanced=False)
            print("✅ 標準Ollamaプロバイダー: 作成成功")
        except Exception as e:
            print(f"❌ 標準Ollamaプロバイダー: {e}")
            success = False
        
        # 拡張プロバイダーテスト（V2）
        enhanced_v2 = list_enhanced_providers().get('v2', [])
        if 'ollama' in enhanced_v2:
            try:
                provider = get_provider('ollama', enhanced=True)
                print("✅ 拡張Ollamaプロバイダー: 作成成功")
            except Exception as e:
                print(f"❌ 拡張Ollamaプロバイダー: {e}")
                success = False
        else:
            print("⚠️ Ollama V2拡張プロバイダーが利用できません")
        
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
        print("⚠️ Ollama利用不可のため、呼び出しテストをスキップ")
        return True
    
    try:
        from llm_api.providers import get_provider
        
        # 最初のモデルを使用（通常はモデル名:タグ形式）
        selected_model = models[0]
        print(f"🎯 使用モデル: {selected_model}")
        
        provider = get_provider('ollama', enhanced=False)
        response = await provider.call(
            "Hello, respond with just 'Test OK'",
            model=selected_model
        )
        
        if response.get('text') and not response.get('error'):
            print(f"✅ 呼び出し成功: {response['text'][:50]}...")
            return True
        else:
            print(f"❌ 呼び出し失敗: {response.get('error', '空の応答')}")
            return False
            
    except Exception as e:
        print(f"❌ 呼び出しテスト失敗: {e}")
        return False

async def test_v2_enhanced_call():
    """V2拡張呼び出しテスト"""
    print("\n🚀 V2拡張呼び出しテスト中...")
    
    # Ollamaの確認
    ollama_ok, models = await check_ollama_status()
    if not ollama_ok or not models:
        print("⚠️ Ollama利用不可のため、V2テストをスキップ")
        return True
    
    try:
        from llm_api.providers import get_provider, list_enhanced_providers
        
        enhanced_v2 = list_enhanced_providers().get('v2', [])
        if 'ollama' not in enhanced_v2:
            print("⚠️ Ollama V2拡張が利用できません")
            return False
        
        selected_model = models[0]
        print(f"🎯 使用モデル: {selected_model}")
        
        provider = get_provider('ollama', enhanced=True)
        response = await provider.call(
            "簡単な質問に答えてください：1+1は？",
            model=selected_model,
            mode='efficient'
        )
        
        if response.get('text') and not response.get('error'):
            print(f"✅ V2拡張呼び出し成功: {response['text'][:50]}...")
            
            # V2固有の情報があるかチェック
            v2_info = response.get('paper_based_improvements') or response.get('v2_improvements')
            if v2_info:
                print(f"🔬 V2機能確認: レジーム={v2_info.get('regime', 'N/A')}")
            
            return True
        else:
            print(f"❌ V2拡張呼び出し失敗: {response.get('error', '空の応答')}")
            return False
            
    except Exception as e:
        print(f"❌ V2拡張テスト失敗: {e}")
        return False

def show_setup_guide():
    """セットアップガイドの表示"""
    print("""
🔧 セットアップガイド:

1. 依存関係のインストール:
   pip install -r requirements.txt

2. Ollamaのインストールと起動:
   # macOS/Linux:
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows:
   https://ollama.ai からダウンロード
   
   # 起動:
   ollama serve

3. モデルのプル:
   ollama pull gemma2:latest    # 推奨
   ollama pull llama3.1:latest  # 代替
   ollama pull phi3:mini        # 軽量版

4. 環境変数の設定（.envファイル作成）:
   cp .env.example .env
   # 必要に応じてAPIキーを設定

5. 確認:
   ollama list
   python quick_test_v2.py

🔑 オプション: 他のLLMプロバイダーのAPIキー設定
   - OpenAI: OPENAI_API_KEY
   - Claude: CLAUDE_API_KEY  
   - Gemini: GEMINI_API_KEY
   - HuggingFace: HF_TOKEN
""")

def show_troubleshooting():
    """トラブルシューティング情報"""
    print("""
🚨 トラブルシューティング:

【依存関係エラー】
- pip install --upgrade pip
- pip install -r requirements.txt
- 仮想環境の使用を推奨: python -m venv venv && source venv/bin/activate

【Ollamaサーバーが起動しない】
- ポート11434が使用されていないか確認: lsof -i :11434
- 別ターミナルで: ollama serve
- ファイアウォール設定を確認

【モデルが見つからない】
- ollama list で確認
- ollama pull <モデル名> でダウンロード
- ディスク容量を確認（モデルは数GB）

【プロバイダーエラー】
- パスの確認: export PYTHONPATH=.
- 権限の確認: chmod +x fetch_llm_v2.py
- Pythonバージョン確認: python --version (3.8+推奨)

【V2拡張機能エラー】
- spaCyモデル: python -m spacy download en_core_web_sm
- メモリ不足の場合: --mode edge を使用

【その他】
- ログレベル変更: export LOG_LEVEL=DEBUG
- 詳細情報: python quick_test_v2.py --verbose
- 問題報告: GitHub Issues
""")

async def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CogniQuantum V2 クイックテスト")
    parser.add_argument("--verbose", action="store_true", help="詳細ログ")
    parser.add_argument("--setup-guide", action="store_true", help="セットアップガイド表示")
    parser.add_argument("--troubleshooting", action="store_true", help="トラブルシューティング表示")
    parser.add_argument("--skip-calls", action="store_true", help="実際の呼び出しテストをスキップ")
    
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
        ("依存関係チェック", check_dependencies),
        ("設定読み込みテスト", test_config_loading),
        ("基本機能テスト", test_basic_functionality),
        ("プロバイダー作成テスト", test_provider_creation),
    ]
    
    if not args.skip_calls:
        tests.extend([
            ("Ollama状態チェック", check_ollama_status),
            ("シンプル呼び出しテスト", test_simple_call),
            ("V2拡張呼び出しテスト", test_v2_enhanced_call)
        ])
    
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
        print("python fetch_llm_v2.py ollama 'Hello' --mode efficient")
        print("python fetch_llm_v2.py ollama '美味しい料理のコツは？' --mode quantum_inspired")
    elif any(name == "依存関係チェック" and not success for name, success in results):
        print("😞 依存関係が不足しています。")
        print("python quick_test_v2.py --setup-guide")
    elif any(name == "Ollama状態チェック" and not success for name, success in results):
        print("😞 Ollamaの接続に失敗しました。セットアップガイドを確認してください。")
        print("python quick_test_v2.py --setup-guide")
    else:
        print("⚠️ 一部のテストが失敗しました。")
        print("python quick_test_v2.py --troubleshooting")

if __name__ == "__main__":
    asyncio.run(main())