from typing import Any, Dict, Optional


def _normalize_protein_opts(protein_opts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Supports BOTH newer keys and older compatibility keys.

    New keys:
      strict_cds, require_translation, x_free, require_start_m, reject_partial_cds, max_protein_aa

    Old keys that may appear in YAML/older code:
      protein_strict_cds_only, protein_require_translation, protein_x_free,
      protein_require_start_m, protein_reject_partial, protein_max_aa
    """
    d = dict(protein_opts or {})

    def pick(new_key: str, old_key: str, default):
        if new_key in d and d[new_key] is not None:
            return d[new_key]
        if old_key in d and d[old_key] is not None:
            return d[old_key]
        return default

    out = {
        "strict_cds": bool(pick("strict_cds", "protein_strict_cds_only", False)),
        "require_translation": bool(pick("require_translation", "protein_require_translation", False)),
        "x_free": bool(pick("x_free", "protein_x_free", False)),
        "require_start_m": bool(pick("require_start_m", "protein_require_start_m", False)),
        "reject_partial_cds": bool(pick("reject_partial_cds", "protein_reject_partial", False)),
    }

    maxv = pick("max_protein_aa", "protein_max_aa", 0)
    if maxv in (None, "", 0):
        out["max_protein_aa"] = 0
    else:
        out["max_protein_aa"] = int(maxv)

    return out


