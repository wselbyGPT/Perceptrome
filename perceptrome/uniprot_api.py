from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Iterator, Optional

import requests

UNIPROT_BASE_URL = "https://rest.uniprot.org"
UNIPROT_SEARCH_URL = f"{UNIPROT_BASE_URL}/uniprotkb/search"
UNIPROT_STREAM_URL = f"{UNIPROT_BASE_URL}/uniprotkb/stream"

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_ACCESSION_RE = re.compile(r"^>(?:sp|tr)\|([^|]+)\|")


def _normalize_base_url(base_url: Optional[str]) -> str:
    if base_url is None:
        return UNIPROT_BASE_URL
    normalized = str(base_url).strip().rstrip("/")
    return normalized or UNIPROT_BASE_URL


def build_uniprot_search_url(base_url: Optional[str] = None) -> str:
    return f"{_normalize_base_url(base_url)}/uniprotkb/search"


def build_uniprot_stream_url(base_url: Optional[str] = None) -> str:
    return f"{_normalize_base_url(base_url)}/uniprotkb/stream"


def build_count_query(
    mode: str,
    explicit_query: Optional[str],
    default_query: str,
) -> str:
    """Build UniProt query string for count/search operations.

    Modes:
      - all: full-length proteins (`fragment:false`)
      - reviewed: reviewed full-length (`reviewed:true AND fragment:false`)
      - unreviewed: unreviewed full-length (`reviewed:false AND fragment:false`)

    If ``explicit_query`` is provided, it is returned as-is.
    """
    if explicit_query and explicit_query.strip():
        return explicit_query.strip()

    base_query = default_query.strip()
    mode_filters = {
        "all": "fragment:false",
        "reviewed": "reviewed:true AND fragment:false",
        "unreviewed": "reviewed:false AND fragment:false",
    }
    if mode not in mode_filters:
        valid = ", ".join(sorted(mode_filters))
        raise ValueError(f"Unsupported UniProt mode {mode!r}. Expected one of: {valid}")

    mode_clause = mode_filters[mode]
    if base_query:
        return f"({base_query}) AND ({mode_clause})"
    return mode_clause


def request_with_retry(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    stream: bool = False,
    session: Optional[requests.Session] = None,
    logger: Optional[logging.Logger] = None,
) -> requests.Response:
    """Execute an HTTP request with exponential-backoff retries.

    Retries only on transient failures: HTTP 429 and HTTP 5xx, plus
    ``requests.RequestException`` transport errors.
    """
    http = session or requests
    log = logger or logging.getLogger("uniprot")

    last_error: Optional[str] = None
    response: Optional[requests.Response] = None

    for attempt in range(max_retries + 1):
        try:
            response = http.request(
                method=method,
                url=url,
                params=params,
                timeout=timeout,
                stream=stream,
            )
            if response.status_code < 400:
                return response

            status = response.status_code
            snippet = response.text[:160].replace("\n", " ") if not stream else "<streaming response body>"
            last_error = f"HTTP {status} from {url} (params={params!r}): {snippet!r}"

            if status not in _TRANSIENT_STATUS_CODES:
                raise RuntimeError(last_error)

        except requests.RequestException as exc:
            last_error = f"Request to {url} failed: {exc}"

        if attempt >= max_retries:
            break

        retry_after = 0.0
        if response is not None:
            try:
                retry_after = float(response.headers.get("Retry-After", 0.0))
            except ValueError:
                retry_after = 0.0

        delay = max(backoff_seconds * (2 ** attempt), retry_after)
        log.warning(
            "UniProt request attempt %s/%s failed (%s); retrying in %.1fs",
            attempt + 1,
            max_retries + 1,
            last_error,
            delay,
        )
        time.sleep(delay)

    raise RuntimeError(
        "UniProt request failed after "
        f"{max_retries + 1} attempts: {last_error or 'unknown error'}"
    )


def fetch_uniprot_count(
    query: str,
    *,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    session: Optional[requests.Session] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch live UniProtKB count for a query using `size=0`.

    Returns a structured payload suitable for CLI JSON output.
    """
    params: Dict[str, Any] = {
        "query": query,
        "size": 0,
        "format": "json",
    }
    search_url = build_uniprot_search_url(base_url)
    response = request_with_retry(
        "GET",
        search_url,
        params=params,
        timeout=timeout,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        session=session,
    )

    parsed_count: Optional[int] = None
    count_source = "unknown"

    header_count = response.headers.get("x-total-results")
    if header_count is not None:
        try:
            parsed_count = int(header_count)
            count_source = "header:x-total-results"
        except ValueError:
            parsed_count = None

    body: Dict[str, Any] = {}
    if parsed_count is None:
        try:
            body = response.json()
        except ValueError:
            body = {}

        for key in ("totalResults", "total", "count"):
            value = body.get(key)
            if isinstance(value, int):
                parsed_count = value
                count_source = f"body:{key}"
                break

        if parsed_count is None:
            hits_total = body.get("hits", {}).get("total", {}) if isinstance(body.get("hits"), dict) else {}
            value = hits_total.get("value") if isinstance(hits_total, dict) else None
            if isinstance(value, int):
                parsed_count = value
                count_source = "body:hits.total.value"

    if parsed_count is None:
        raise RuntimeError(
            "Unable to determine UniProt result count from response headers/body "
            f"for query {query!r}"
        )

    return {
        "provider": "uniprot",
        "endpoint": search_url,
        "query": query,
        "count": parsed_count,
        "count_source": count_source,
        "status_code": response.status_code,
    }


def stream_uniprot_fasta(
    query: str,
    *,
    include_isoform: bool = False,
    timeout: float = 60.0,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    session: Optional[requests.Session] = None,
    base_url: Optional[str] = None,
) -> Iterator[str]:
    """Stream FASTA from UniProt incrementally without full buffering."""
    params: Dict[str, Any] = {
        "query": query,
        "format": "fasta",
        "includeIsoform": str(include_isoform).lower(),
    }
    stream_url = build_uniprot_stream_url(base_url)
    response = request_with_retry(
        "GET",
        stream_url,
        params=params,
        timeout=timeout,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        stream=True,
        session=session,
    )

    for line in response.iter_lines(decode_unicode=True):
        if line is None:
            continue
        yield f"{line}\n"


def parse_accession_from_fasta_header(header_line: str) -> Optional[str]:
    """Extract accession from a FASTA header line.

    Supports canonical UniProt headers (e.g. `>sp|P12345|...`) and falls back
    to first token parsing for less-structured headers.
    """
    if not header_line.startswith(">"):
        return None

    match = _ACCESSION_RE.match(header_line)
    if match:
        return match.group(1)

    token = header_line[1:].strip().split()[0] if header_line[1:].strip() else ""
    if not token:
        return None

    if "|" in token:
        parts = token.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return token
