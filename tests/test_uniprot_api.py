import unittest
from unittest.mock import patch

import sys
import types

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.RequestException = Exception
    requests_stub.Session = object
    requests_stub.Response = object
    requests_stub.request = lambda *a, **k: None
    sys.modules["requests"] = requests_stub

from perceptrome import uniprot_api


class FakeResponse:
    def __init__(self, status_code=200, headers=None, text="", json_payload=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self._json_payload = json_payload

    def json(self):
        if isinstance(self._json_payload, Exception):
            raise self._json_payload
        return self._json_payload


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def request(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("No fake responses left")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class UniProtApiTests(unittest.TestCase):
    def test_build_count_query_modes(self):
        self.assertEqual(
            uniprot_api.build_count_query("all", None, "taxonomy_id:2"),
            "(taxonomy_id:2) AND (fragment:false)",
        )
        self.assertEqual(
            uniprot_api.build_count_query("reviewed", "", "taxonomy_id:2"),
            "(taxonomy_id:2) AND (reviewed:true AND fragment:false)",
        )
        self.assertEqual(
            uniprot_api.build_count_query("unreviewed", None, "taxonomy_id:2"),
            "(taxonomy_id:2) AND (reviewed:false AND fragment:false)",
        )
        self.assertEqual(
            uniprot_api.build_count_query("all", " reviewed:true ", "taxonomy_id:2"),
            "reviewed:true",
        )

    def test_fetch_uniprot_count_prefers_header(self):
        session = FakeSession(
            [FakeResponse(status_code=200, headers={"x-total-results": "123"}, json_payload={"totalResults": 9})]
        )
        payload = uniprot_api.fetch_uniprot_count("q", session=session)
        self.assertEqual(payload["count"], 123)
        self.assertEqual(payload["count_source"], "header:x-total-results")

    def test_fetch_uniprot_count_falls_back_to_body(self):
        session = FakeSession([FakeResponse(status_code=200, headers={}, json_payload={"hits": {"total": {"value": 77}}})])
        payload = uniprot_api.fetch_uniprot_count("q", session=session)
        self.assertEqual(payload["count"], 77)
        self.assertEqual(payload["count_source"], "body:hits.total.value")

    def test_request_with_retry_transient_then_success(self):
        session = FakeSession(
            [
                FakeResponse(status_code=429, headers={"Retry-After": "0"}, text="rate limit"),
                FakeResponse(status_code=503, text="unavailable"),
                FakeResponse(status_code=200, text="ok"),
            ]
        )
        with patch("perceptrome.uniprot_api.time.sleep") as sleep_mock:
            response = uniprot_api.request_with_retry(
                "GET",
                "https://example.test",
                session=session,
                max_retries=3,
                backoff_seconds=0.01,
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(sleep_mock.call_count, 2)

    def test_request_with_retry_raises_after_retries_exhausted(self):
        session = FakeSession(
            [
                FakeResponse(status_code=500, text="boom1"),
                FakeResponse(status_code=502, text="boom2"),
                FakeResponse(status_code=504, text="boom3"),
            ]
        )
        with patch("perceptrome.uniprot_api.time.sleep"):
            with self.assertRaises(RuntimeError):
                uniprot_api.request_with_retry(
                    "GET",
                    "https://example.test",
                    session=session,
                    max_retries=2,
                    backoff_seconds=0,
                )


if __name__ == "__main__":
    unittest.main()
