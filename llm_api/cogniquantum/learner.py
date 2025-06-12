# /llm_api/cogniquantum/learner.py
# タイトル: Complexity Learner
# 役割: 実行結果からプロンプトの複雑性を学習し、知識を永続化する。

import json
import logging
import os
import re
from typing import Dict, Optional

from .enums import ComplexityRegime

logger = logging.getLogger(__name__)

class ComplexityLearner:
    """
    プロンプトの複雑性と成功した推論レジームの関係を学習し、
    将来の分析に活用するためのクラス。
    """
    def __init__(self, storage_path: str = 'complexity_learnings.json'):
        self.storage_path = storage_path
        self.learnings = self._load_learnings()

    def _load_learnings(self) -> Dict[str, str]:
        """保存された学習データをファイルから読み込む。"""
        if not os.path.exists(self.storage_path):
            return {}
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"学習データの読み込みに失敗しました: {e}")
            return {}

    def _save_learnings(self):
        """現在の学習データをファイルに保存する。"""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.learnings, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"学習データの保存に失敗しました: {e}")

    def _create_signature(self, prompt: str) -> str:
        """プロンプトから一意のシグネチャ（特徴）を生成する。"""
        # 簡単な正規化：小文字化、非英数字の削除
        normalized_prompt = re.sub(r'[^a-z0-9\s]', '', prompt.lower())
        # 単語に分割し、重複を排除してソート
        words = sorted(list(set(normalized_prompt.split())))
        # 重要な単語（5文字以上）を最大10個まで抽出
        significant_words = [word for word in words if len(word) > 4][:10]
        return "_".join(significant_words)

    def record_outcome(self, prompt: str, successful_regime: ComplexityRegime):
        """プロンプトと成功したレジームのペアを記録する。"""
        signature = self._create_signature(prompt)
        if not signature:
            return
            
        # 既存の学習内容と異なる場合のみ更新・保存
        if self.learnings.get(signature) != successful_regime.value:
            logger.info(f"新しい学びを記録します: signature='{signature}', regime='{successful_regime.value}'")
            self.learnings[signature] = successful_regime.value
            self._save_learnings()

    def get_suggestion(self, prompt: str) -> Optional[ComplexityRegime]:
        """
        新しいプロンプトに基づき、過去の学習から最適なレジームを提案する。
        """
        signature = self._create_signature(prompt)
        if not signature:
            return None
            
        learned_regime_str = self.learnings.get(signature)
        if learned_regime_str:
            logger.info(f"過去の学習データから提案を取得しました: signature='{signature}', regime='{learned_regime_str}'")
            try:
                return ComplexityRegime(learned_regime_str)
            except ValueError:
                return None
        return None