# /llm_api/cogniquantum/pipelines/speculative.py
# タイトル: Speculative Thought Pipeline Handler
# 役割: 投機的思考パイプライン処理をsystem.pyから分離

import logging
from typing import Any, Dict, Optional
import httpx

from .adaptive import AdaptivePipeline
from ...rag import RAGManager
from ...providers import get_provider
from ...providers.base import LLMProvider

logger = logging.getLogger(__name__)

class SpeculativePipeline:
    """投機的思考パイプライン処理を担当するクラス"""
    
    def __init__(self, provider: LLMProvider, base_model_kwargs: Dict[str, Any]):
        self.provider = provider
        self.base_model_kwargs = base_model_kwargs
        self.adaptive_pipeline = AdaptivePipeline(provider, base_model_kwargs)
        logger.info("SpeculativePipeline を初期化しました")
    
    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        use_rag: bool = False,
        knowledge_base_path: Optional[str] = None,
        use_wikipedia: bool = False
    ) -> Dict[str, Any]:
        """投機的思考パイプラインの実行"""
        logger.info(f"思考レベルの投機的デコーディングパイプライン開始: {prompt[:80]}...")
        
        # RAG処理
        current_prompt = prompt
        rag_source = None
        if use_rag or use_wikipedia:
            rag_manager = RAGManager(provider=self.provider, use_wikipedia=use_wikipedia, knowledge_base_path=knowledge_base_path)
            current_prompt = await rag_manager.retrieve_and_augment(prompt)
            rag_source = 'wikipedia' if use_wikipedia else 'knowledge_base'

        # 1. ドラフト生成用モデルの自動選択
        draft_model_name = await self._find_lightweight_model()
        
        if not draft_model_name:
            logger.warning("適切な軽量モデルが見つかりませんでした。適応型パイプラインにフォールバックします。")
            return await self.adaptive_pipeline.execute(current_prompt, system_prompt, mode='balanced')
        
        try:
            # 2. ドラフト生成
            drafts = await self._generate_drafts(current_prompt, draft_model_name)
            
            if not drafts:
                logger.error("ドラフト生成に失敗しました。")
                return self._format_error_response("ドラフト生成に失敗しました。")
            
            # 3. 検証と統合
            final_solution = await self._verify_and_integrate(current_prompt, drafts, system_prompt)
            
            if not final_solution:
                logger.error("検証・統合に失敗しました。")
                return self._format_error_response("検証・統合に失敗しました。")
            
            # レスポンス構築
            thought_process = {
                'draft_generator': f"ollama/{draft_model_name}",
                'verifier_integrator': self.provider.provider_name,
                'drafts_generated': len(drafts.split('\n\n')) if drafts else 0,
                'speculative_method': 'draft_then_verify'
            }
            
            v2_improvements = {
                'regime': 'N/A (Speculative)',
                'reasoning_approach': 'speculative_thought',
                'speculative_execution_enabled': True,
                'rag_enabled': use_rag or use_wikipedia,
                'rag_source': rag_source,
                'draft_model': draft_model_name,
            }

            return {
                'success': True,
                'final_solution': final_solution,
                'image_url': None,
                'thought_process': thought_process,
                'v2_improvements': v2_improvements,
                'version': 'v2'
            }
            
        except Exception as e:
            logger.error(f"投機的思考パイプライン実行中にエラー: {e}", exc_info=True)
            return self._format_error_response(str(e))
    
    async def _find_lightweight_model(self) -> Optional[str]:
        """Ollamaから軽量モデルを取得"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:11434/api/tags")
                response.raise_for_status()
                available_models = [m['name'] for m in response.json().get('models', [])]
            
            # 軽量モデルの候補を検索
            lightweight_candidates = []
            for model in available_models:
                model_lower = model.lower()
                if any(k in model_lower for k in ['phi', 'gemma', '2b', '3b', '4b']):
                    lightweight_candidates.append(model)
            
            if lightweight_candidates:
                # 最も小さそうなモデルを選択
                selected_model = sorted(lightweight_candidates, key=len)[0]
                logger.info(f"Ollamaからドラフト生成用の軽量モデルを自動選択しました: {selected_model}")
                return selected_model
            
            logger.warning("適切な軽量モデルが見つかりませんでした")
            return None
            
        except Exception as e:
            logger.warning(f"Ollamaから利用可能なモデルの取得に失敗しました: {e}")
            return None
    
    async def _generate_drafts(self, prompt: str, model_name: str) -> Optional[str]:
        """軽量モデルでドラフトを生成"""
        try:
            draft_provider = get_provider('ollama', enhanced=False)
            draft_model_kwargs = {'model': model_name, 'temperature': 0.7}
            
            draft_prompt = f"""以下の質問に対して、考えられる答えのドラフトを3つ、多様な視点から簡潔に生成してください。

質問: {prompt}

回答形式:
1. [視点1からの回答]
2. [視点2からの回答]  
3. [視点3からの回答]"""
            
            response = await draft_provider.call(draft_prompt, "", **draft_model_kwargs)
            
            if response.get('error'):
                logger.error(f"ドラフト生成でエラー: {response['error']}")
                return None
            
            return response.get('text', '')
            
        except Exception as e:
            logger.error(f"ドラフト生成中にエラー: {e}")
            return None
    
    async def _verify_and_integrate(self, original_prompt: str, drafts: str, system_prompt: str) -> Optional[str]:
        """ドラフトを検証・統合"""
        try:
            verification_prompt = f"""以下の「元の質問」に対して、いくつかの「回答ドラフト」が提供されました。
あなたは専門家として、これらのドラフトを評価・検証し、最も正確で包括的な最終回答を1つに統合してください。
元の質問の意図を完全に満たすように、情報を取捨選択し、再構成してください。

# 元の質問
{original_prompt}

# 回答ドラフト
---
{drafts}
---

# 統合・検証済みの最終回答
（上記のドラフトを参考に、最も適切で完全な回答を生成してください）
"""
            
            response = await self.provider.call(verification_prompt, system_prompt, **self.base_model_kwargs)
            
            if response.get('error'):
                logger.error(f"検証・統合でエラー: {response['error']}")
                return None
            
            return response.get('text', '')
            
        except Exception as e:
            logger.error(f"検証・統合中にエラー: {e}")
            return None
    
    def _format_error_response(self, error_message: str) -> Dict[str, Any]:
        """エラーレスポンスの形式"""
        return {
            'success': False,
            'final_solution': None,
            'image_url': None,
            'thought_process': {'error': error_message},
            'v2_improvements': {'speculative_execution_enabled': True},
            'version': 'v2',
            'error': error_message
        }