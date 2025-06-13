# /llm_api/rag/manager.py
# タイトル: RAG Manager with Robust Query Extraction
# 役割: RAGプロセスを管理する。LLMから検索クエリを抽出するプロンプトを強化し、出力のサニタイズ処理を追加する。

import logging
import re # reモジュールをインポート
from typing import Optional

from .knowledge_base import KnowledgeBase
from .retriever import Retriever
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..providers.base import LLMProvider

logger = logging.getLogger(__name__)

class RAGManager:
    """RAGプロセスを管理するクラス"""
    def __init__(self,
                 provider: LLMProvider,
                 use_wikipedia: bool = False,
                 knowledge_base_path: Optional[str] = None):
        
        self.provider = provider
        self.use_wikipedia = use_wikipedia
        self.knowledge_base_path = knowledge_base_path
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # ★★★ このメソッドを修正 ★★★
    async def _extract_search_query(self, prompt: str) -> str:
        """LLMを使ってプロンプトから検索クエリを抽出し、サニタイズする"""
        # プロンプトをより厳格に修正
        extraction_prompt = f"""以下のユーザーの質問から、Wikipediaで検索するのに最適な「検索キーワード」だけを抽出してください。
説明や前置き、句読点などは一切含めず、キーワードのみを返してください。複数のキーワードはスペースで区切ってください。

例：
質問：「Llama.cppのvLLMとの違いは？」
出力：「Llama.cpp vLLM」

質問：「日本の首都はどこですか？」
出力：「東京」

---
質問: "{prompt}"
---
出力："""
        try:
            # 検索クエリ抽出のためにLLMを呼び出す
            response = await self.provider.call(extraction_prompt, "")
            
            # LLMの応答からキーワードをクリーンアップして抽出
            query = response.get('text', prompt).strip()
            
            # 万が一、LLMが余計なテキストを返した場合に備えて後処理を追加
            # 「出力：」や「検索キーワード：」のような接頭辞を削除
            query = re.sub(r'^(出力|検索キーワード)[:：\s]*', '', query).strip()
            # クォーテーションを削除
            query = query.replace("「", "").replace("」", "").replace("\"", "").replace("'", "")

            logger.info(f"抽出・サニタイズされたWikipedia検索クエリ: '{query}'")
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