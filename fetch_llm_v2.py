# /fetch_llm_v2.py
"""
CogniQuantum V2 CLI Entry Point
"""
import asyncio
import logging
import sys
import os

# プロジェクトルートをsys.pathに追加して、cliモジュールをインポート可能にする
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# main関数をcliモジュールからインポート
from cli.main import main

# ロギング設定
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.getLogger(__name__).critical(f"CLIの実行中に致命的なエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)