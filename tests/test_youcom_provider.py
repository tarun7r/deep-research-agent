import pytest

from src.utils.tools import _build_search_providers
from src.utils.web_utils import DuckDuckGoProvider, YouComProvider


class _FakeResponse:
    def __init__(self, payload, status_code=200, text="ok"):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def get(self, url, params=None, headers=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self.response


class _FakeManager:
    def __init__(self, client):
        self.client = client

    async def get_client(self):
        return self.client


@pytest.mark.asyncio
async def test_youcom_provider_parses_web_results():
    provider = YouComProvider(
        api_key="test-key",
        base_url="https://ydc-index.io/v1/search",
        max_results=3,
    )
    client = _FakeClient(
        _FakeResponse(
            {
                "results": {
                    "web": [
                        {
                            "url": "https://example.com/a",
                            "title": "Result A",
                            "description": "Result A description",
                            "snippets": ["Snippet A1", "Snippet A2"],
                        },
                        {
                            "url": "https://example.com/b",
                            "title": "Result B",
                            "snippets": [],
                        },
                    ]
                }
            }
        )
    )
    provider.client_manager = _FakeManager(client)

    results = await provider.search("agentic search", max_results=2)

    assert len(results) == 2
    assert results[0].title == "Result A"
    assert results[0].snippet == "Snippet A1"
    assert results[1].title == "Result B"
    assert results[1].snippet == ""
    assert client.calls[0]["url"] == "https://ydc-index.io/v1/search"
    assert client.calls[0]["params"] == {"query": "agentic search", "count": 2}
    assert client.calls[0]["headers"]["X-API-Key"] == "test-key"
    assert client.calls[0]["headers"]["User-Agent"] == "youdotcom-integration/tarun7r-deep-research-agent"


def test_build_search_providers_prefers_youcom_with_duckduckgo_fallback():
    original_provider = _build_search_providers.__globals__["config"].search_provider
    original_key = _build_search_providers.__globals__["config"].youcom_api_key
    original_results = _build_search_providers.__globals__["config"].max_search_results_per_query
    original_url = _build_search_providers.__globals__["config"].youcom_base_url

    try:
        cfg = _build_search_providers.__globals__["config"]
        cfg.search_provider = "youcom"
        cfg.youcom_api_key = "test-key"
        cfg.youcom_base_url = "https://ydc-index.io/v1/search"
        cfg.max_search_results_per_query = 4

        providers = _build_search_providers()

        assert isinstance(providers[0], YouComProvider)
        assert isinstance(providers[1], DuckDuckGoProvider)
        assert providers[0].max_results == 4
        assert providers[1].max_results == 4
    finally:
        cfg = _build_search_providers.__globals__["config"]
        cfg.search_provider = original_provider
        cfg.youcom_api_key = original_key
        cfg.max_search_results_per_query = original_results
        cfg.youcom_base_url = original_url
