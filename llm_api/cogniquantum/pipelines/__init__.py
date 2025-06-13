# /llm_api/cogniquantum/pipelines/__init__.py
# タイトル: Pipeline Package Initializer
# 役割: 全てのパイプラインの統一インターフェース

from .adaptive import AdaptivePipeline
from .parallel import ParallelPipeline
from .quantum_inspired import QuantumInspiredPipeline
from .speculative import SpeculativePipeline

__all__ = [
    "AdaptivePipeline",
    "ParallelPipeline", 
    "QuantumInspiredPipeline",
    "SpeculativePipeline",
]