# /llm_api/tools/image_retrieval.py
# タイトル: Image Retrieval Tool (Corrected)
# 役割: SerpApiを利用して画像を検索する。インポート文を正しい形式に修正済み。

import logging
import os
from typing import NamedTuple, Optional

# SerpApiの正しいインポート文に修正
from serpapi import GoogleSearch
from ..config import settings

logger = logging.getLogger(__name__)

class ImageResult(NamedTuple):
    """Represents a single image search result."""
    title: str
    source: str
    content_url: str
    thumbnail_url: str

def search(query: str) -> Optional[ImageResult]:
    """Performs an image search using SerpApi and returns the top result."""
    api_key = settings.SERPAPI_API_KEY
    if not api_key:
        logger.warning("SERPAPI_API_KEYが設定されていません。画像検索はスキップされます。")
        return None

    params = {
        "engine": "google_images",
        "q": query,
        "api_key": api_key,
        "tbm": "isch", # Specify image search
    }

    try:
        search_client = GoogleSearch(params)
        results = search_client.get_dict()
        
        image_results = results.get("images_results")
        if not image_results:
            logger.warning(f"画像検索で結果が見つかりませんでした: '{query}'")
            return None
        
        top_result = image_results[0]
        return ImageResult(
            title=top_result.get("title", ""),
            source=top_result.get("source", ""),
            content_url=top_result.get("original", ""),
            thumbnail_url=top_result.get("thumbnail", ""),
        )
    except Exception as e:
        logger.error(f"SerpApi経由の画像検索に失敗しました: {e}", exc_info=True)
        return None