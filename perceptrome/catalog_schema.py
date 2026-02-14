import re
from typing import List, Tuple

_CANONICAL_ALIASES = {
    "plasmid": "plasmid",
    "plasmids": "plasmid",
    "virus": "virus",
    "viruses": "virus",
    "virus'": "virus",
    "eukaryote": "eukaryote",
    "eukaryotes": "eukaryote",
}

_CLAUSE_RE = re.compile(r"^(?P<count>[+-]?\d+)\s+(?P<category>[A-Za-z']+)\s*$")


def parse_catalog_schema(text: str) -> List[Tuple[str, int]]:
    """Parse schema text like '100 plasmids, 20 viruses' into ordered (category, count)."""
    if text is None:
        raise ValueError("Catalog schema is required. Example: '100 plasmids, 20 viruses'.")

    raw = text.strip()
    if not raw:
        raise ValueError("Catalog schema is empty. Example: '100 plasmids, 20 viruses'.")

    parsed: List[Tuple[str, int]] = []
    clauses = raw.split(",")
    for idx, clause in enumerate(clauses, start=1):
        piece = clause.strip()
        if not piece:
            raise ValueError(
                f"Clause #{idx} is empty. Remove extra commas and use '<count> <category>' format."
            )

        match = _CLAUSE_RE.match(piece)
        if not match:
            raise ValueError(
                f"Invalid clause #{idx}: {piece!r}. Expected '<count> <category>' like '100 plasmids'."
            )

        count_text = match.group("count")
        category_text = match.group("category").lower()

        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid count in clause #{idx}: {count_text!r}. Count must be an integer."
            ) from exc

        if count < 0:
            raise ValueError(
                f"Invalid count in clause #{idx}: {count}. Count must be >= 0."
            )

        canonical = _CANONICAL_ALIASES.get(category_text)
        if canonical is None:
            allowed = ", ".join(sorted(_CANONICAL_ALIASES))
            raise ValueError(
                f"Unknown category in clause #{idx}: {category_text!r}. "
                f"Supported aliases: {allowed}."
            )

        parsed.append((canonical, count))

    return parsed
