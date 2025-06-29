from urllib.parse import urljoin
import requests


class HeygemApiClient:
    def __init__(self, api_config: dict):
        self.base = api_config["api_base"].rstrip("/")
        self.api_key = api_config["api_key"]

    def _build_url(self, endpoint: str) -> str:
        return urljoin(self.base + "/", endpoint)

    def _build_headers(self) -> dict:
        return {
            "accept": "application/json",
            "x-api-key": self.api_key
        }

    def get(self, endpoint: str, **kwargs):
        url = self._build_url(endpoint)
        headers = kwargs.pop("headers", self._build_headers())
        return requests.get(url, headers=headers, **kwargs)

    def post(self, endpoint: str, **kwargs):
        url = self._build_url(endpoint)
        headers = kwargs.pop("headers", self._build_headers())
        return requests.post(url, headers=headers, **kwargs)
