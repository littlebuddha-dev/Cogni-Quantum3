# /llm_api/cogniquantum/tracker.py
"""
推論プロセスの追跡とメトリクス関連のモジュール
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .enums import ComplexityRegime

@dataclass
class ReasoningMetrics:
    """推論プロセスのメトリクス"""
    complexity_score: float
    regime: ComplexityRegime
    thinking_tokens_used: int
    solution_positions: List[int]
    correct_solution_positions: List[int]
    first_correct_position: Optional[int]
    overthinking_detected: bool
    consistency_score: float

class SolutionTracker:
    """中間解の追跡と分析（論文のthinking trace分析に基づく）"""
    def __init__(self):
        self.solutions = []
        self.token_positions = []
        
    def track_solution(self, solution: str, token_position: int, is_correct: bool = None):
        self.solutions.append({
            'solution': solution,
            'token_position': token_position,
            'is_correct': is_correct,
            'timestamp': len(self.solutions)
        })
        self.token_positions.append(token_position)
    
    def analyze_solution_patterns(self) -> Dict[str, Any]:
        if not self.solutions:
            return {'pattern': 'no_solutions'}
        early_correct = any(
            sol['is_correct'] and sol['timestamp'] < len(self.solutions) * 0.3 
            for sol in self.solutions if sol['is_correct'] is not None
        )
        overthinking = early_correct and len(self.solutions) > 5
        return {
            'total_solutions': len(self.solutions),
            'early_correct_found': early_correct,
            'overthinking_detected': overthinking,
            'solution_distribution': self._analyze_distribution()
        }
    
    def _analyze_distribution(self) -> Dict[str, float]:
        if not self.token_positions:
            return {}
        positions = self.token_positions
        max_pos = max(positions) if positions else 1
        return {
            'mean_position': sum(positions) / len(positions) if positions else 0,
            'early_concentration': sum(1 for p in positions if p < max_pos * 0.3) / len(positions) if positions else 0,
            'late_concentration': sum(1 for p in positions if p > max_pos * 0.7) / len(positions) if positions else 0
        }