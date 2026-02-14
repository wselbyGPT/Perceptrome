from typing import Dict

# Semantic nuccore categories for discovery/count queries.
# Terms are passed directly to ESearch `term=`.
NUCCORE_CATEGORY_QUERY_MAP: Dict[str, str] = {
    "plasmid": "plasmid[Title] AND biomol_genomic[PROP]",
    "virus": "Viruses[Organism] AND biomol_genomic[PROP]",
    "eukaryote": "Eukaryota[Organism] AND biomol_genomic[PROP]",
}


def get_nuccore_category_query(category: str) -> str:
    """Return the ESearch term for a semantic nuccore category."""
    key = category.strip().lower()
    if key not in NUCCORE_CATEGORY_QUERY_MAP:
        known = ", ".join(sorted(NUCCORE_CATEGORY_QUERY_MAP))
        raise ValueError(f"Unknown nuccore category: {category!r}. Known categories: {known}")
    return NUCCORE_CATEGORY_QUERY_MAP[key]
