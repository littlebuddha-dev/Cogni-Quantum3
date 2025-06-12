# /llm_api/rag/retriever.py
# パス: littlebuddha-dev/cogni-quantum2.1/Cogni-Quantum2.1-fb17e3467b051803511a1506de5e02910bbae07e/llm_api/rag/retriever.py
# タイトル: Retriever with updated LangChain method
# 役割: LangChainの非推奨警告に対応し、新しい推奨メソッド 'invoke' を使用する。

import logging
from typing import List
from .knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

class Retriever:
    """ナレッジベースから情報を検索するクラス"""
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb_retriever = knowledge_base.get_retriever()

    def search(self, query: str) -> List[str]:
        """クエリに最も関連するドキュメントチャンクを検索する"""
        if not self.kb_retriever:
            logger.warning("Retrieverが初期化されていません。検索をスキップします。")
            return []
            
        try:
            logger.info(f"クエリで関連情報を検索中: '{query[:50]}...'")
            # 非推奨の get_relevant_documents から推奨の invoke に変更
            docs = self.kb_retriever.invoke(query)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"検索中にエラーが発生しました: {e}", exc_info=True)
            return []