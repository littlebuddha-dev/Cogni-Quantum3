# /llm_api/__init__.py
# タイトル: CogniQuantum統合LLM APIモジュール (Refactored)
# 役割: モジュールの初期化とロギング設定を行う。設定読み込みはconfigモジュールに委譲。

__version__ = "2.1.0"
__author__ = "CogniQuantum Project"
__description__ = "CogniQuantum統合LLM CLI - 革新的認知量子推論システム"

import logging
import os
from llm_api.config import settings

# ロギング設定
def setup_logging():
    """ロギングの設定"""
    log_level_str = os.getenv("LOG_LEVEL", settings.LOG_LEVEL).upper()
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