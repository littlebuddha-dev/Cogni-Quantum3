# /tests/test_providers.py

import pytest
from unittest.mock import patch, MagicMock
import os

# The main function to test
from llm_api.providers import get_provider, _providers_cache
# The classes we expect to be returned
from llm_api.providers.openai import OpenAIProvider
from llm_api.providers.enhanced_openai_v2 import EnhancedOpenAIProviderV2

# Clean the cache before each test to ensure test isolation
@pytest.fixture(autouse=True)
def clear_provider_cache():
    _providers_cache.clear()
    yield

@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
class TestProviderLoading:
    """Test suite for the get_provider function."""

    def test_get_standard_provider(self):
        """Test that get_provider returns a standard provider instance."""
        provider = get_provider("openai")
        assert isinstance(provider, OpenAIProvider)
        assert provider.is_available()

    def test_get_enhanced_v2_provider(self):
        """Test that get_provider returns a V2 enhanced provider instance."""
        provider = get_provider("enhanced_openai_v2")
        assert isinstance(provider, EnhancedOpenAIProviderV2)

    def test_get_provider_not_found(self):
        """Test that a ValueError is raised for a non-existent provider."""
        with pytest.raises(ValueError, match="Provider 'non_existent_provider' not found."):
            get_provider("non_existent_provider")

    def test_get_provider_prefer_v2(self):
        """Test the prefer_v2 flag to get the V2 version of a standard provider."""
        provider = get_provider("openai", prefer_v2=True)
        assert isinstance(provider, EnhancedOpenAIProviderV2)

    def test_provider_caching(self):
        """Test that providers are cached and the same instance is returned on subsequent calls."""
        provider1 = get_provider("openai")
        provider2 = get_provider("openai")
        assert provider1 is provider2  # Check for object identity

def test_provider_availability_check():
    """Test the is_available method of a provider based on environment variables."""
    # Test without the required API key
    with patch.dict(os.environ, clear=True):
        # We need to re-import to re-evaluate the module-level client initializations
        from llm_api.providers import openai
        import importlib
        importlib.reload(openai)
        provider = openai.OpenAIProvider()
        assert not provider.is_available()
    
    # Test with the required API key
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"}):
        importlib.reload(openai)
        provider = openai.OpenAIProvider()
        assert provider.is_available()