# /llm_api/cogniquantum/pipelines/parallel.py
# タイトル: Parallel Pipeline Handler
# 役割: 並列推論パイプライン処理をsystem.pyから分離

import asyncio
import logging
from typing import Any, Dict, List

from .adaptive import AdaptivePipeline
from ..enums import ComplexityRegime
from ...rag import RAGManager
from ...providers.base import LLMProvider

logger = logging.getLogger(__name__)

class ParallelPipeline:
    """並列推論パイプライン処理を担当するクラス"""
    
    def __init__(self, provider: LLMProvider, base_model_kwargs: Dict[str, Any], shared_adaptive_pipeline: AdaptivePipeline = None):
        self.provider = provider
        self.base_model_kwargs = base_model_kwargs
        # 共有パイプラインがあれば使用、なければ新規作成
        self.adaptive_pipeline = shared_adaptive_pipeline or AdaptivePipeline(provider, base_model_kwargs)
        logger.info("ParallelPipeline を初期化しました")
    
    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        use_rag: bool = False,
        knowledge_base_path: str = None,
        use_wikipedia: bool = False
    ) -> Dict[str, Any]:
        """並列推論パイプラインの実行"""
        logger.info(f"並列推論パイプライン実行開始: {prompt[:80]}...")
        
        # RAG処理
        final_prompt = prompt
        rag_source = None
        if use_rag or use_wikipedia:
            rag_manager = RAGManager(provider=self.provider, use_wikipedia=use_wikipedia, knowledge_base_path=knowledge_base_path)
            final_prompt = await rag_manager.retrieve_and_augment(prompt)
            rag_source = 'wikipedia' if use_wikipedia else 'knowledge_base'
        
        # 3つの異なる複雑性レジームで並列実行（改善版）
        logger.info("3つの複雑性レジーム（低・中・高）で並列実行します")
        
        try:
            # 並列実行のタスクを作成
            tasks = [
                self._execute_regime_safely("low", final_prompt, system_prompt, ComplexityRegime.LOW),
                self._execute_regime_safely("medium", final_prompt, system_prompt, ComplexityRegime.MEDIUM),
                self._execute_regime_safely("high", final_prompt, system_prompt, ComplexityRegime.HIGH),
            ]
            
            # セマフォで同時実行数を制限（Ollamaサーバーの負荷軽減）
            semaphore = asyncio.Semaphore(2)  # 最大2つの同時実行
            
            async def limited_task(task, regime_name):
                async with semaphore:
                    logger.info(f"{regime_name}レジーム実行開始")
                    result = await task
                    logger.info(f"{regime_name}レジーム実行完了")
                    return result
            
            limited_tasks = [
                limited_task(tasks[0], "低複雑性"),
                limited_task(tasks[1], "中複雑性"), 
                limited_task(tasks[2], "高複雑性")
            ]
            
            results = await asyncio.gather(*limited_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"並列実行中にエラー: {e}")
            return self._format_error_response(str(e))
        
        # 有効な結果をフィルタリング
        valid_solutions = []
        for i, res in enumerate(results):
            regime_names = ["低複雑性", "中複雑性", "高複雑性"]
            if isinstance(res, Exception):
                logger.warning(f"{regime_names[i]}レジームでエラー: {res}")
                continue
                
            if res and res.get('success') and not res.get('error'):
                valid_solutions.append({
                    'solution': res.get('final_solution'),
                    'complexity_regime': res.get('v2_improvements', {}).get('regime'),
                    'reasoning_approach': res.get('v2_improvements', {}).get('reasoning_approach'),
                    'regime_name': regime_names[i],
                    'full_response': res
                })
            else:
                logger.warning(f"{regime_names[i]}レジームが無効な結果を返しました")
        
        if not valid_solutions:
            return self._format_error_response("全ての並列パイプラインが失敗しました。")
        
        logger.info(f"{len(valid_solutions)}/3 のレジームが成功しました")
            
        # 最良解選択（改善版）
        best_solution_info = await self._select_best_solution(valid_solutions, prompt)
        final_solution = best_solution_info['solution']
        
        # レスポンス構築
        thought_process = {
            'reasoning_approach': f"parallel_best_of_{len(valid_solutions)}",
            'candidates_considered': len(valid_solutions),
            'selected_regime': best_solution_info.get('complexity_regime'),
            'selection_reason': best_solution_info.get('selection_reason', 'First valid solution'),
            'all_candidates': [
                {
                    'regime': sol['regime_name'],
                    'approach': sol['reasoning_approach'],
                    'length': len(sol['solution'])
                } for sol in valid_solutions
            ]
        }
        
        v2_improvements = {
            'rag_enabled': use_rag or use_wikipedia,
            'rag_source': rag_source,
            'parallel_execution': True,
            'regimes_tested': len(valid_solutions),
            'selected_regime': best_solution_info.get('complexity_regime'),
        }

        return {
            'success': True,
            'final_solution': final_solution,
            'image_url': None,
            'thought_process': thought_process,
            'v2_improvements': v2_improvements,
            'version': 'v2'
        }
    
    async def _execute_regime_safely(self, regime_name: str, prompt: str, system_prompt: str, force_regime: ComplexityRegime) -> Dict[str, Any]:
        """安全な個別レジーム実行"""
        try:
            result = await self.adaptive_pipeline.execute(
                prompt=prompt,
                system_prompt=system_prompt,
                force_regime=force_regime,
                real_time_adjustment=False,  # 並列実行時は調整を無効化
                mode='parallel'
            )
            return result
        except Exception as e:
            logger.error(f"{regime_name}レジーム実行中にエラー: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _select_best_solution(self, solutions: List[Dict[str, Any]], original_prompt: str) -> Dict[str, Any]:
        """最良解の選択（簡易版）"""
        if len(solutions) == 1:
            return {**solutions[0], 'selection_reason': '唯一の有効な解'}
        
        # 長さとアプローチに基づく簡易選択
        scored_solutions = []
        for sol in solutions:
            score = 0
            solution_text = sol['solution']
            
            # 長さスコア（適度な長さを好む）
            length = len(solution_text)
            if 100 <= length <= 1000:
                score += 3
            elif 50 <= length <= 2000:
                score += 2
            else:
                score += 1
            
            # 複雑性レジームボーナス（中複雑性を好む）
            if sol['complexity_regime'] == 'medium':
                score += 2
            elif sol['complexity_regime'] in ['low', 'high']:
                score += 1
            
            scored_solutions.append({**sol, 'score': score})
        
        # 最高スコアの解を選択
        best_solution = max(scored_solutions, key=lambda x: x['score'])
        best_solution['selection_reason'] = f"スコア{best_solution['score']}で選択（長さ:{len(best_solution['solution'])}文字、レジーム:{best_solution['complexity_regime']}）"
        
        logger.info(f"並列実行結果から最良解を選択: {best_solution['regime_name']} (スコア: {best_solution['score']})")
        return best_solution
    
    def _format_error_response(self, error_message: str) -> Dict[str, Any]:
        """エラーレスポンスの形式"""
        return {
            'success': False,
            'final_solution': None,
            'image_url': None,
            'thought_process': {'error': error_message},
            'v2_improvements': {'parallel_execution': True},
            'version': 'v2',
            'error': error_message
        }