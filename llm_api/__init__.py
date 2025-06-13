# /llm_api/__init__.py
# タイトル: CogniQuantum統合LLM APIモジュール (Refactored and Fixed)
# 役割: モジュールの初期化とロギング設定を行う。循環インポートを回避し、設定読み込みを遅延実行。

__version__ = "2.1.0"
__author__ = "CogniQuantum Project"
__description__ = "CogniQuantum統合LLM CLI - 革新的認知量子推論システム"

import logging
import os

# ロギング設定
def setup_logging():
    """ロギングの設定"""
    # 環境変数から直接ログレベルを取得（循環インポートを回避）
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # CogniQuantum特有のロガー
    cq_logger = logging.getLogger('cogniquantum')
    cq_logger.setLevel(log_level)

# モジュール初期化時にロギングを設定
setup_logging()

# 設定オブジェクトは必要に応じて遅延インポート
def get_settings():
    """設定オブジェクトを遅延インポートで取得"""
    from llm_api.config import settings
    return settings