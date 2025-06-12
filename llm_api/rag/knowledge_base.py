# /llm_api/rag/knowledge_base.py
# パス: littlebuddha-dev/cogni-quantum2.1/Cogni-Quantum2.1-fb17e3467b051803511a1506de5e02910bbae07e/llm_api/rag/knowledge_base.py
# タイトル: KnowledgeBase with updated LangChain components
# 役割: LangChainの非推奨警告に対応し、新しい推奨コンポーネントを使用する。

import logging
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# 新しい推奨ライブラリからインポート
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """ナレッジベースを管理するクラス"""
    def __init__(self, embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.vector_store: Optional[FAISS] = None
        # 新しいクラスを使用
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def load_documents(self, source: str):
        """ファイルパスまたはURLからドキュメントを読み込み、ベクトルストアを構築する"""
        logger.info(f"'{source}' からドキュメントを読み込んでいます...")
        loader = None
        try:
            if source.lower().startswith("http://") or source.lower().startswith("https://"):
                logger.info("URLとしてソースを処理します。")
                loader = WebBaseLoader(source)
            elif source.lower().endswith(".pdf"):
                logger.info("PDFファイルとしてソースを処理します。")
                loader = PyPDFLoader(source)
            else:
                logger.info("テキストファイルとしてソースを処理します。")
                loader = TextLoader(source, encoding='utf-8')

            if loader:
                documents = loader.load()
                chunks = self.text_splitter.split_documents(documents)
                
                logger.info(f"{len(chunks)}個のチャンクを作成し、ベクトルストアを構築します...")
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
                logger.info("ベクトルストアの構築が完了しました。")
            else:
                raise ValueError("対応していないソースタイプです。")
            
        except Exception as e:
            logger.error(f"ドキュメントの読み込みまたはベクトル化中にエラー: {e}", exc_info=True)
            raise

    def get_retriever(self, top_k: int = 3):
        """Retrieverを取得する際に、検索するドキュメント数を指定できるように変更"""
        if not self.vector_store:
            return None
        return self.vector_store.as_retriever(search_kwargs={'k': top_k})