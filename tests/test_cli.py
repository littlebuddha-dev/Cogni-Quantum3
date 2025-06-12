# /tests/test_cli.py

import pytest
from unittest.mock import patch, MagicMock
import sys

# The main entry point of the CLI
from fetch_llm_v2 import main

@patch('fetch_llm_v2.get_provider')
def test_cli_simple_prompt(mock_get_provider):
    """Test a simple CLI call with a provider and prompt."""
    mock_provider_instance = MagicMock()
    mock_provider_instance.get_response.return_value = "CLI test successful"
    mock_get_provider.return_value = mock_provider_instance
    
    test_args = ["fetch_llm_v2.py", "--provider", "openai", "--prompt", "Hello, world!"]
    with patch.object(sys, 'argv', test_args):
        main()

    # Assertions
    mock_get_provider.assert_called_with("enhanced_openai_v2", prefer_v2=True)
    mock_provider_instance.get_response.assert_called_once_with("Hello, world!", mode='adaptive')


@patch('fetch_llm_v2.get_provider')
def test_cli_v2_mode_argument(mock_get_provider):
    """Test that the --mode argument is passed correctly to the provider."""
    mock_provider_instance = MagicMock()
    mock_provider_instance.get_response.return_value = "Efficient mode test"
    mock_get_provider.return_value = mock_provider_instance
    
    test_args = ["fetch_llm_v2.py", "--provider", "enhanced_openai_v2", "--mode", "efficient", "--prompt", "Be quick"]
    with patch.object(sys, 'argv', test_args):
        main()

    mock_get_provider.assert_called_once_with("enhanced_openai_v2", prefer_v2=True)
    mock_provider_instance.get_response.assert_called_once_with("Be quick", mode='efficient')


@patch('fetch_llm_v2.CogniQuantumCLIV2Fixed.run_health_check')
@patch('fetch_llm_v2.get_provider')
def test_cli_health_check(mock_get_provider, mock_health_check):
    """Test that the --health-check argument calls the correct function and exits."""
    test_args = ["fetch_llm_v2.py", "--health-check"]
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit):
            main()
    
    mock_health_check.assert_called_once()
    mock_get_provider.assert_not_called()


def test_cli_missing_prompt():
    """Test that the CLI exits if --prompt is missing for a standard request."""
    test_args = ["fetch_llm_v2.py", "--provider", "openai"]
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit):
            main()


@patch('fetch_llm_v2.get_provider')
def test_cli_fallback_mechanism(mock_get_provider):
    """Test the provider fallback logic from V2 -> V1 -> standard."""
    mock_provider_instance = MagicMock()
    # Simulate get_provider failing for V2 and V1, but succeeding for standard
    mock_get_provider.side_effect = [
        ValueError("V2 provider not found"),
        ValueError("V1 provider not found"),
        mock_provider_instance 
    ]
    
    test_args = ["fetch_llm_v2.py", "--provider", "openai", "--prompt", "fallback test"]
    with patch.object(sys, 'argv', test_args):
        main()
        
    assert mock_get_provider.call_count == 3
    calls = mock_get_provider.call_args_list
    # The logic first tries the enhanced_v2 version
    assert calls[0].args[0] == "enhanced_openai_v2"
    # Then falls back to the enhanced (v1) version
    assert calls[1].args[0] == "enhanced_openai"
    # Finally falls back to the standard version
    assert calls[2].args[0] == "openai"
    # The successful provider should be used
    mock_provider_instance.get_response.assert_called_once()