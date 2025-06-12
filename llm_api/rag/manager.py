# /llm_api/rag/manager.py
# パス: littlebuddha-dev/cogni-quantum2.1/Cogni-Quantum2.1-fb17e3467b051803511a1506de5e02910bbae07e/llm_api/rag/manager.py
# タイトル: RAG Manager with Smart Query Extraction
# 役割: RAGプロセスを管理する。LLMを使い、プロンプトから最適な検索クエリを抽出する機能を追加。

import logging
from typing import Optional

from .knowledge_base import KnowledgeBase
from .retriever import Retriever
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..providers.base import LLMProvider # LLMProviderをインポート

logger = logging.getLogger(__name__)

class RAGManager:
    """RAGプロセスを管理するクラス"""
    def __init__(self,
                 provider: LLMProvider, # providerを受け取るように変更
                 use_wikipedia: bool = False,
                 knowledge_base_path: Optional[str] = None):
        
        self.provider = provider
        self.use_wikipedia = use_wikipedia
        self.knowledge_base_path = knowledge_base_path
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    async def _extract_search_query(self, prompt: str) -> str:
        """LLMを使ってプロンプトから検索クエリを抽出する"""
        extraction_prompt = f"""以下のユーザーの質問から、Wikipediaで検索するのに最も適した、簡潔なキーワードまたは固有表現を1つだけ抽出してください。

質問: "{prompt}"

検索キーワード:"""
        try:
            # 検索クエリ抽出のためにLLMを呼び出す
            response = await self.provider.call(extraction_prompt, "")
            # LLMの応答からキーワードをクリーンアップして抽出
            query = response.get('text', prompt).strip().replace("「", "").replace("」", "").replace("\"", "")
            logger.info(f"抽出されたWikipedia検索クエリ: '{query}'")
            return query
        except Exception as e:
            logger.error(f"検索クエリの抽出中にエラー: {e}")
            return prompt # 失敗した場合は元のプロンプトをクエリとする

    async def _retrieve_from_wikipedia(self, query: str) -> str:
        """Wikipediaから情報を検索してコンテキストを生成する"""
        logger.info(f"Wikipediaで検索中: '{query}'")
        try:
            docs = WikipediaLoader(query=query, lang="ja", load_max_docs=2, doc_content_chars_max=2000).load()
            if not docs:
                logger.warning("Wikipediaで関連情報が見つかりませんでした。")
                return ""
            
            chunks = self.text_splitter.split_documents(docs)
            return "\n\n".join([chunk.page_content for chunk in chunks])

        except Exception as e:
            logger.error(f"Wikipedia検索中にエラー: {e}", exc_info=True)
            return ""

    async def _retrieve_from_knowledge_base(self, query: str) -> str:
        """ファイル/URLベースのナレッジベースから情報を検索する"""
        try:
            kb = KnowledgeBase()
            kb.load_documents(self.knowledge_base_path)
            retriever = Retriever(kb)
            return "\n\n".join(retriever.search(query))
        except Exception as e:
            logger.error(f"ナレッジベースからの検索中にエラー: {e}", exc_info=True)
            return ""

    async def retrieve_and_augment(self, original_prompt: str) -> str:
        """情報を検索し、プロンプトを拡張する"""
        retrieved_context = ""
        if self.use_wikipedia:
            # 質問から検索クエリを抽出するステップを追加
            search_query = await self._extract_search_query(original_prompt)
            retrieved_context = await self._retrieve_from_wikipedia(search_query)
        elif self.knowledge_base_path:
            retrieved_context = await self._retrieve_from_knowledge_base(original_prompt)
        
        if not retrieved_context:
            logger.info("関連情報が見つからなかったため、プロンプトは拡張されません。")
            return original_prompt
        
        augmented_prompt = f"""以下の「コンテキスト情報」を最優先の根拠として利用し、「元の質問」に答えてください。

# コンテキスト情報
---
{retrieved_context}
---

# 元の質問
{original_prompt}
"""
        logger.info("プロンプトが検索されたコンテキストで拡張されました。")
        return augmented_prompt