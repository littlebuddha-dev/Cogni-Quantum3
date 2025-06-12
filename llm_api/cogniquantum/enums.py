# /llm_api/cogniquantum/enums.py
"""
CogniQuantumシステムで使用するEnum定義
"""
from enum import Enum

class ComplexityRegime(Enum):
    """論文で特定された3つの複雑性体制"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"