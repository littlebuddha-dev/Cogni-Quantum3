# /llm_api/utils/performance_monitor.py

import time
import logging
from collections import deque
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    システム全体のパフォーマンスを監視し、統計情報を収集するクラス。
    """
    def __init__(self, history_size: int = 100):
        """
        PerformanceMonitorを初期化します。

        Args:
            history_size (int): 保持する最近のAPIコールの履歴数。
        """
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        self.call_history = deque(maxlen=history_size)
        self.provider_metrics: Dict[str, Dict] = {}

    def record_call(self, provider_name: str, response: Dict[str, Any]):
        """
        APIコールの結果を記録します。

        Args:
            provider_name (str): 使用されたプロバイダーの名前。
            response (dict): プロバイダーからのレスポンス。
        """
        self.total_requests += 1
        
        provider_metrics = response.get('provider_metrics', {})
        execution_time = provider_metrics.get('execution_time', 0.0)
        is_error = response.get('error', False)

        if is_error:
            self.failed_requests += 1
        else:
            self.successful_requests += 1
        
        self.total_processing_time += execution_time
        
        # 履歴に追加
        self.call_history.append({
            'timestamp': time.time(),
            'provider': provider_name,
            'execution_time': execution_time,
            'success': not is_error,
            'enhanced': response.get('enhanced', False)
        })

        # プロバイダーごとの統計を更新
        if provider_name not in self.provider_metrics:
            self.provider_metrics[provider_name] = {
                'calls': 0, 'success': 0, 'total_time': 0.0
            }
        
        p_metrics = self.provider_metrics[provider_name]
        p_metrics['calls'] += 1
        p_metrics['success'] += 1 if not is_error else 0
        p_metrics['total_time'] += execution_time

        logger.debug(f"Performance recorded for provider '{provider_name}'.")

    def get_summary(self) -> Dict[str, Any]:
        """
        システム全体のパフォーマンスサマリーを返します。

        Returns:
            dict: パフォーマンスサマリー。
        """
        avg_time = (self.total_processing_time / self.total_requests) if self.total_requests > 0 else 0
        success_rate = (self.successful_requests / self.total_requests) if self.total_requests > 0 else 0

        provider_summary = {}
        for name, metrics in self.provider_metrics.items():
            provider_summary[name] = {
                'total_calls': metrics['calls'],
                'success_rate': (metrics['success'] / metrics['calls']) if metrics['calls'] > 0 else 0,
                'avg_response_time': (metrics['total_time'] / metrics['calls']) if metrics['calls'] > 0 else 0
            }

        return {
            'overall_statistics': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'overall_success_rate': success_rate,
                'average_processing_time': avg_time,
            },
            'provider_breakdown': provider_summary,
            'recent_calls_count': len(self.call_history)
        }