"""Unit tests for the Exa search provider."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_response(items):
    """Build a fake exa-py search_and_contents response."""
    results = [SimpleNamespace(**item) for item in items]
    return SimpleNamespace(results=results)


@pytest.fixture
def mock_exa_class():
    """Patch the module-level Exa import with a mock class."""
    with patch("src.utils.web_utils.Exa") as MockExa:
        instance = MagicMock()
        instance.headers = {}
        MockExa.return_value = instance
        yield MockExa, instance


class TestExaProviderInit:
    def test_sets_integration_header(self, mock_exa_class):
        from src.utils.web_utils import ExaProvider

        _, instance = mock_exa_class
        ExaProvider(api_key="test-key")

        assert instance.headers["x-exa-integration"] == "deep-research-agent"

    def test_provider_name(self, mock_exa_class):
        from src.utils.web_utils import ExaProvider

        provider = ExaProvider(api_key="test-key")
        assert provider.name == "exa"

    def test_passes_api_key(self, mock_exa_class):
        from src.utils.web_utils import ExaProvider

        MockExa, _ = mock_exa_class
        ExaProvider(api_key="my-key")
        MockExa.assert_called_once_with(api_key="my-key")


class TestExaProviderSearch:
    def test_search_parses_results(self, mock_exa_class):
        from src.utils.web_utils import ExaProvider

        _, instance = mock_exa_class
        instance.search_and_contents.return_value = _make_response([
            {
                "title": "Example Title",
                "url": "https://example.com/a",
                "text": "Full article text here",
                "highlights": ["Key highlight one", "Key highlight two"],
                "summary": "An example summary",
            },
        ])

        provider = ExaProvider(api_key="test-key", max_results=5)
        results = asyncio.run(provider.search("test query"))

        assert len(results) == 1
        assert results[0].title == "Example Title"
        assert results[0].url == "https://example.com/a"
        assert results[0].query == "test query"
        assert results[0].content == "Full article text here"
        # Summary wins over highlights and text when present.
        assert results[0].snippet == "An example summary"

    def test_search_passes_request_kwargs(self, mock_exa_class):
        from src.utils.web_utils import ExaProvider

        _, instance = mock_exa_class
        instance.search_and_contents.return_value = _make_response([])

        provider = ExaProvider(
            api_key="test-key",
            max_results=7,
            search_type="neural",
            category="research paper",
            include_domains=["arxiv.org"],
            exclude_domains=["spam.example"],
            start_published_date="2025-01-01",
        )
        asyncio.run(provider.search("llm research"))

        instance.search_and_contents.assert_called_once()
        args, kwargs = instance.search_and_contents.call_args
        assert args == ("llm research",)
        assert kwargs["num_results"] == 7
        assert kwargs["type"] == "neural"
        assert kwargs["category"] == "research paper"
        assert kwargs["include_domains"] == ["arxiv.org"]
        assert kwargs["exclude_domains"] == ["spam.example"]
        assert kwargs["start_published_date"] == "2025-01-01"
        # Both content modes should be requested together.
        assert "highlights" in kwargs
        assert "text" in kwargs

    def test_rate_limit_error_mapped(self, mock_exa_class):
        from src.utils.web_utils import ExaProvider
        from src.exceptions import RateLimitError

        _, instance = mock_exa_class
        instance.search_and_contents.side_effect = Exception("429 rate limit exceeded")

        provider = ExaProvider(api_key="test-key")
        with pytest.raises(RateLimitError):
            asyncio.run(provider.search("query"))

    def test_generic_error_mapped_to_search_error(self, mock_exa_class):
        from src.utils.web_utils import ExaProvider
        from src.exceptions import SearchError

        _, instance = mock_exa_class
        instance.search_and_contents.side_effect = Exception("boom")

        provider = ExaProvider(api_key="test-key")
        with pytest.raises(SearchError):
            asyncio.run(provider.search("query"))


class TestSnippetFallback:
    """Content may arrive as any mix of text/highlights/summary -- the
    snippet extraction must cascade through them gracefully."""

    def _snippet_for(self, mock_exa_class, item):
        from src.utils.web_utils import ExaProvider

        _, instance = mock_exa_class
        instance.search_and_contents.return_value = _make_response([item])
        provider = ExaProvider(api_key="test-key")
        return asyncio.run(provider.search("q"))[0].snippet

    def test_highlights_used_when_no_summary(self, mock_exa_class):
        snippet = self._snippet_for(mock_exa_class, {
            "title": "t",
            "url": "https://x.com",
            "text": "full text body",
            "highlights": ["first", "second"],
        })
        assert "first" in snippet
        assert "second" in snippet

    def test_text_used_when_no_summary_or_highlights(self, mock_exa_class):
        snippet = self._snippet_for(mock_exa_class, {
            "title": "t",
            "url": "https://x.com",
            "text": "full text body only",
        })
        assert snippet.startswith("full text body only")

    def test_empty_when_nothing_present(self, mock_exa_class):
        snippet = self._snippet_for(mock_exa_class, {
            "title": "t",
            "url": "https://x.com",
        })
        assert snippet == ""


class TestProviderRegistration:
    """Verify _build_search_providers only returns Exa when configured."""

    def test_exa_not_registered_by_default(self, mock_exa_class):
        from src.utils import tools
        from src.utils.web_utils import ExaProvider

        with patch.object(tools.config, "search_provider", "duckduckgo"):
            providers = tools._build_search_providers()
        assert not any(isinstance(p, ExaProvider) for p in providers)

    def test_exa_registered_when_selected(self, mock_exa_class):
        from src.utils import tools
        from src.utils.web_utils import ExaProvider

        with patch.object(tools.config, "search_provider", "exa"), \
                patch.object(tools.config, "exa_api_key", "test-key"), \
                patch.object(tools.config, "exa_search_type", "auto"), \
                patch.object(tools.config, "exa_category", ""):
            providers = tools._build_search_providers()
        assert len(providers) == 1
        assert isinstance(providers[0], ExaProvider)
