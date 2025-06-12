# /tests/test_cogniquantum.py

import pytest
from unittest.mock import MagicMock, patch, call

# Test target modules
from llm_api.cogniquantum_v2 import (
    CogniQuantumSystemV2,
    ComplexityRegime,
    AdaptiveComplexityAnalyzer,
    EnhancedReasoningEngine
)
from llm_api.providers.base import LLMProvider

# --- Test Fixtures ---

@pytest.fixture
def mock_provider():
    """A pytest fixture that creates a mock LLMProvider."""
    provider = MagicMock(spec=LLMProvider)
    # Configure the mock to return a simple response when get_response is called
    provider.get_response.side_effect = lambda prompt, **kwargs: f"Mocked response for: {prompt}"
    provider.is_available.return_value = True
    return provider

@pytest.fixture
def cq_system(mock_provider):
    """A pytest fixture that creates an instance of CogniQuantumSystemV2 with a mock provider."""
    return CogniQuantumSystemV2(standard_provider=mock_provider, verbose=False)

# --- Test Cases ---

class TestAdaptiveComplexityAnalyzer:
    """Tests for the AdaptiveComplexityAnalyzer class."""

    def test_analyze_prompt_complexity(self, mock_provider):
        """Test the prompt complexity analysis."""
        analyzer = AdaptiveComplexityAnalyzer(standard_provider=mock_provider)
        
        # Mock the internal calculation methods to isolate the logic of analyze_prompt_complexity
        with patch.object(analyzer, '_calculate_syntactic_complexity', return_value=10):
            with patch.object(analyzer, '_calculate_semantic_complexity', return_value=10):
                regime, score = analyzer.analyze_prompt_complexity("Simple question")
                assert regime == ComplexityRegime.LOW
                assert score == 20

        with patch.object(analyzer, '_calculate_syntactic_complexity', return_value=30):
            with patch.object(analyzer, '_calculate_semantic_complexity', return_value=40):
                regime, score = analyzer.analyze_prompt_complexity("Medium question")
                assert regime == ComplexityRegime.MEDIUM
                assert score == 70

        with patch.object(analyzer, '_calculate_syntactic_complexity', return_value=50):
            with patch.object(analyzer, '_calculate_semantic_complexity', return_value=45):
                regime, score = analyzer.analyze_prompt_complexity("Very long and complex question asking multiple things")
                assert regime == ComplexityRegime.HIGH
                assert score == 95


class TestCogniQuantumSystemV2:
    """Tests for the main CogniQuantumSystemV2 class and its dispatch logic."""

    def test_process_prompt_mode_dispatch(self, cq_system):
        """Test that process_prompt correctly dispatches to the right engine method based on the mode."""
        prompt = "test prompt"
        
        # Mock the reasoning engine's methods to verify they are called correctly
        with patch.object(cq_system.reasoning_engine, '_execute_low_complexity_reasoning') as mock_low:
            cq_system.process_prompt(prompt, mode='efficient')
            mock_low.assert_called_once_with(prompt, mode='efficient')

        with patch.object(cq_system.reasoning_engine, '_execute_medium_complexity_reasoning') as mock_medium:
            cq_system.process_prompt(prompt, mode='balanced')
            mock_medium.assert_called_once_with(prompt, mode='balanced')
            
        with patch.object(cq_system.reasoning_engine, '_execute_high_complexity_reasoning') as mock_high:
            cq_system.process_prompt(prompt, mode='decomposed')
            mock_high.assert_called_once_with(prompt, mode='decomposed')

    @patch('llm_api.cogniquantum_v2.AdaptiveComplexityAnalyzer.analyze_prompt_complexity')
    def test_process_prompt_adaptive_mode(self, mock_analyze, cq_system):
        """Test the adaptive mode correctly uses the analyzer's result."""
        prompt = "An adaptive question"

        # Case 1: Analyzer returns LOW complexity
        mock_analyze.return_value = (ComplexityRegime.LOW, 25)
        with patch.object(cq_system.reasoning_engine, '_execute_low_complexity_reasoning') as mock_low:
            cq_system.process_prompt(prompt, mode='adaptive')
            mock_analyze.assert_called_once_with(prompt)
            mock_low.assert_called_once()

        # Case 2: Analyzer returns HIGH complexity
        mock_analyze.return_value = (ComplexityRegime.HIGH, 85)
        with patch.object(cq_system.reasoning_engine, '_execute_high_complexity_reasoning') as mock_high:
            cq_system.process_prompt(prompt, mode='adaptive')
            mock_high.assert_called_once()


class TestEnhancedReasoningEngine:
    """Tests for the core reasoning logic in EnhancedReasoningEngine."""

    def test_high_complexity_decomposes_and_integrates(self, mock_provider):
        """Verify the high complexity flow: decompose -> solve sub-problems -> integrate."""
        engine = EnhancedReasoningEngine(standard_provider=mock_provider, verbose=False)
        complex_prompt = "Explain quantum computing and its impact on cryptography."

        # Define the mock responses for each step of the high-complexity flow
        mock_provider.get_response.side_effect = [
            # 1. Response for the decomposition prompt
            "Sub-problem 1: Explain quantum bits (qubits).\nSub-problem 2: Explain Shor's algorithm.",
            # 2. Response for solving sub-problem 1
            "Qubits can exist in a superposition of 0 and 1.",
            # 3. Response for solving sub-problem 2
            "Shor's algorithm can factor large numbers, breaking RSA encryption.",
            # 4. Response for the integration prompt
            "Final integrated answer about quantum computing and cryptography."
        ]
        
        result = engine._execute_high_complexity_reasoning(complex_prompt)

        # Assert the final result is the integrated one
        assert result == "Final integrated answer about quantum computing and cryptography."
        
        # Assert that the provider was called 4 times (1 decompose + 2 solves + 1 integrate)
        assert mock_provider.get_response.call_count == 4

        # Optionally, check the prompts sent to the mock provider
        actual_calls = mock_provider.get_response.call_args_list
        assert "Decompose the following complex problem" in actual_calls[0].args[0]
        assert "Explain quantum bits (qubits)" in actual_calls[1].args[0]
        assert "Explain Shor's algorithm" in actual_calls[2].args[0]
        assert "Integrate the following solutions" in actual_calls[3].args[0]