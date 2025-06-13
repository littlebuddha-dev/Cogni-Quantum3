# /llm_api/cogniquantum/pipelines/adaptive.py
# タイトル: Adaptive Pipeline Handler
# 役割: 適応型パイプライン処理をsystem.pyから分離

import logging
from typing import Any, Dict, Optional

from ..analyzer import AdaptiveComplexityAnalyzer
from ..engine import EnhancedReasoningEngine
from ..enums import ComplexityRegime
from ..learner import ComplexityLearner
from ...rag import RAGManager
from ...providers.base import LLMProvider

logger = logging.getLogger(__name__)

class AdaptivePipeline:
    """適応型パイプライン処理を担当するクラス"""
    
    def __init__(self, provider: LLMProvider, base_model_kwargs: Dict[str, Any]):
        self.provider = provider
        self.base_model_kwargs = base_model_kwargs
        self.learner = ComplexityLearner()
        self.complexity_analyzer = AdaptiveComplexityAnalyzer(learner=self.learner)
        self.reasoning_engine = EnhancedReasoningEngine(provider, base_model_kwargs, complexity_analyzer=self.complexity_analyzer)
        self.max_adjustment_attempts = 2
        logger.info("AdaptivePipeline 完全版を初期化しました")
    
    async def execute(
        self,
        prompt: str,
        system_prompt: str = "",
        force_regime: Optional[ComplexityRegime] = None,
        use_rag: bool = False,
        knowledge_base_path: Optional[str] = None,
        use_wikipedia: bool = False,
        real_time_adjustment: bool = True,
        mode: str = 'adaptive'
    ) -> Dict[str, Any]:
        """適応型パイプラインの実行"""
        logger.info(f"適応型パイプライン開始: {prompt[:80]}...")
        
        # Edge モードの特別処理
        is_edge_mode = (mode == 'edge')
        if is_edge_mode:
            logger.info("エッジデバイス最適化モードで実行。高度な機能は無効化。")
            use_rag = False
            use_wikipedia = False
            real_time_adjustment = False
            force_regime = ComplexityRegime.LOW
        
        # RAG処理
        current_prompt = prompt
        rag_source = None
        if use_rag or use_wikipedia:
            rag_manager = RAGManager(provider=self.provider, use_wikipedia=use_wikipedia, knowledge_base_path=knowledge_base_path)
            current_prompt = await rag_manager.retrieve_and_augment(prompt)
            rag_source = 'wikipedia' if use_wikipedia else 'knowledge_base'
        
        try:
            # 複雑性分析
            complexity_score, current_regime = self.complexity_analyzer.analyze_complexity(current_prompt, mode=mode)
            
            if force_regime:
                current_regime = force_regime
                logger.info(f"レジームを '{current_regime.value}' に強制設定しました。")
            
            initial_regime = current_regime
            final_reasoning_result = None
            
            # 自己調整ループ
            for attempt in range(self.max_adjustment_attempts):
                logger.info(f"推論試行 {attempt + 1}/{self.max_adjustment_attempts} (レジーム: {current_regime.value})")
                reasoning_result = await self.reasoning_engine.execute_reasoning(current_prompt, system_prompt, complexity_score, current_regime)
                final_reasoning_result = reasoning_result.copy()
                
                if reasoning_result.get('error'): 
                    return self._format_response(None, None, None, success=False, error=reasoning_result['error'])
                
                final_solution = reasoning_result.get('solution')
                if force_regime or not real_time_adjustment or (attempt + 1) >= self.max_adjustment_attempts: 
                    break
                
                evaluation = await self._self_evaluate_solution(final_solution, prompt, current_regime)
                if evaluation.get("is_sufficient"):
                    final_reasoning_result['self_evaluation'] = {'outcome': 'sufficient', 'reason': evaluation.get('reason')}
                    break 
                else:
                    final_reasoning_result['self_evaluation'] = {'outcome': 'insufficient', 'reason': evaluation.get('reason'), 'next_regime': evaluation.get("next_regime").value}
                    new_regime = evaluation.get("next_regime", current_regime)
                    if new_regime != current_regime:
                        logger.info(f"自己評価に基づき複雑性を再調整: {current_regime.value} -> {new_regime.value}")
                        current_regime = new_regime
                        current_prompt = f"前回の回答は不十分でした。より深く、包括的な分析を行ってください。\n元の質問: {prompt}\n前回の回答: {final_solution}\n"
                    else:
                        logger.info("同じ複雑性レジームが推奨されたため、調整を終了します。")
                        break
            
            # 学習記録
            if real_time_adjustment and current_regime != initial_regime: 
                self.learner.record_outcome(prompt, current_regime)
            
            # 最終改善
            final_solution = await self._evaluate_and_refine(final_reasoning_result['solution'], current_prompt, system_prompt, current_regime)
            
            # レスポンス構築
            thought_process = {
                'complexity_score': complexity_score,
                'initial_regime': initial_regime.value,
                'decomposition': final_reasoning_result.get('decomposition'),
                'sub_solutions': final_reasoning_result.get('sub_solutions'),
                'self_evaluation': final_reasoning_result.get('self_evaluation'),
            }
            
            v2_improvements = {
                'regime': current_regime.value,
                'reasoning_approach': final_reasoning_result.get('reasoning_approach'),
                'overthinking_prevention': final_reasoning_result.get('overthinking_prevention', False),
                'collapse_prevention': final_reasoning_result.get('collapse_prevention', False),
                'rag_enabled': use_rag or use_wikipedia,
                'rag_source': rag_source,
                'real_time_adjustment_active': real_time_adjustment and not force_regime,
                'learned_suggestion_used': self.learner.get_suggestion(prompt) is not None,
                'is_edge_optimized': is_edge_mode,
            }

            return self._format_response(final_solution, thought_process, v2_improvements)

        except Exception as e:
            logger.error(f"適応型パイプライン実行中にエラー: {e}", exc_info=True)
            return self._format_response(None, None, None, success=False, error=str(e))
    
    async def _self_evaluate_solution(self, solution: str, original_prompt: str, current_regime: ComplexityRegime) -> Dict[str, Any]:
        """解の自己評価"""
        if len(solution) < 150 and current_regime == ComplexityRegime.LOW:
            return {"is_sufficient": False, "reason": "Solution may be too brief for the question.", "next_regime": ComplexityRegime.MEDIUM}
        return {"is_sufficient": True, "reason": "Solution seems adequate."}
    
    async def _evaluate_and_refine(self, solution: str, original_prompt: str, system_prompt: str, regime: ComplexityRegime) -> str:
        """解の評価と改善"""
        if regime == ComplexityRegime.LOW:
            logger.info("低複雑性問題: refinementスキップ（overthinking防止）")
            return solution
        if regime in [ComplexityRegime.MEDIUM, ComplexityRegime.HIGH]:
            return await self._perform_limited_refinement(solution, original_prompt, system_prompt)
        return solution
    
    async def _perform_limited_refinement(self, solution: str, original_prompt: str, system_prompt: str) -> str:
        """限定的改善の実行"""
        logger.info("解の限定的改善プロセスを開始...")
        refinement_prompt = f"""以下の「元の質問」に対する「回答案」です。
内容をレビューし、以下の観点で改善してください。
- 明確さ: より分かりやすい表現か？
- 正確性: 事実誤認はないか？
- 完全性: 重要な情報が欠けていないか？

改善した最終版の回答のみを出力してください。自己評価や変更点の説明は不要です。

# 元の質問
{original_prompt}

# 回答案
---
{solution}
---

# 改善された最終回答
"""
        response = await self.provider.call(refinement_prompt, system_prompt, **self.base_model_kwargs)
        
        if response.get('error'):
            logger.warning(f"改善プロセス中にエラーが発生しました: {response['error']}。元の解を返します。")
            return solution
            
        logger.info("解の改善が完了しました。")
        return response.get('text', solution)
    
    def _format_response(self, solution, thought_process, v2_improvements, success=True, error=None):
        """統一されたレスポンス形式"""
        base_response = {
            'success': success,
            'final_solution': solution,
            'image_url': None,
            'thought_process': thought_process,
            'v2_improvements': v2_improvements,
            'version': 'v2',
        }
        if error:
            base_response['error'] = error
        return base_response