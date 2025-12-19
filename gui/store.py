from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .utils import now_epoch


@dataclass
class Preset:
    name: str
    args: Dict[str, Any]
    created: int = 0
    updated: int = 0
    last_used: int = 0
    use_count: int = 0
    favorite: bool = False


@dataclass
class HistoryEntry:
    id: int
    ts: str
    command_key: str
    label: str
    args: Dict[str, Any]


class PresetHistoryStore:
    """
    settings schema (current):
      presets: {
        command_key: [
          { name: str, args: dict, meta: { created, updated, last_used, use_count, favorite } }
        ]
      }
      history: [
        { id: int, ts: str, command_key: str, label: str, args: dict }
      ]
      history_next_id: int
      history_limit: int
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings
        self.settings.setdefault("presets", {})
        self.settings.setdefault("history", [])
        self.settings.setdefault("history_limit", 200)
        self.settings.setdefault("history_next_id", 1)
        self._migrate_presets()
        self._migrate_history()

    def _migrate_presets(self) -> None:
        presets = self.settings.get("presets", {})
        if not isinstance(presets, dict):
            self.settings["presets"] = {}
            return
        now = now_epoch()
        for ck, plist in list(presets.items()):
            if not isinstance(plist, list):
                presets[ck] = []
                continue
            new_list = []
            for p in plist:
                if not isinstance(p, dict):
                    continue
                name = p.get("name")
                args = p.get("args")
                if not isinstance(name, str) or not isinstance(args, dict):
                    continue
                meta = p.get("meta")
                if not isinstance(meta, dict):
                    meta = {"created": now, "updated": now, "last_used": 0, "use_count": 0, "favorite": False}
                    p = {"name": name, "args": args, "meta": meta}
                else:
                    meta.setdefault("created", now)
                    meta.setdefault("updated", now)
                    meta.setdefault("last_used", 0)
                    meta.setdefault("use_count", 0)
                    meta.setdefault("favorite", False)
                    p["meta"] = meta
                new_list.append(p)
            presets[ck] = new_list
        self.settings["presets"] = presets

    def _migrate_history(self) -> None:
        hist = self.settings.get("history", [])
        if not isinstance(hist, list):
            self.settings["history"] = []
            return
        next_id = int(self.settings.get("history_next_id", 1) or 1)
        max_seen = 0
        for item in hist:
            if isinstance(item, dict) and isinstance(item.get("id"), int):
                max_seen = max(max_seen, int(item["id"]))
        if max_seen >= next_id:
            next_id = max_seen + 1

        for item in hist:
            if not isinstance(item, dict):
                continue
            if not isinstance(item.get("id"), int):
                item["id"] = next_id
                next_id += 1
            else:
                max_seen = max(max_seen, int(item["id"]))
        self.settings["history_next_id"] = max(next_id, max_seen + 1)

    # ---- presets ----
    def list_presets(self, command_key: str) -> List[Preset]:
        raw = self.settings.get("presets", {}).get(command_key, [])
        out: List[Preset] = []
        for item in raw:
            try:
                meta = item.get("meta", {}) if isinstance(item, dict) else {}
                out.append(
                    Preset(
                        name=str(item["name"]),
                        args=dict(item["args"]),
                        created=int(meta.get("created", 0) or 0),
                        updated=int(meta.get("updated", 0) or 0),
                        last_used=int(meta.get("last_used", 0) or 0),
                        use_count=int(meta.get("use_count", 0) or 0),
                        favorite=bool(meta.get("favorite", False)),
                    )
                )
            except Exception:
                continue
        out.sort(key=lambda p: (not p.favorite, -(p.last_used or 0), p.name.lower()))
        return out

    def _find_preset_index(self, command_key: str, name: str) -> Optional[int]:
        presets = self.settings["presets"].setdefault(command_key, [])
        for i, p in enumerate(presets):
            if str(p.get("name", "")) == name:
                return i
        return None

    def add_or_replace_preset(self, command_key: str, name: str, args: Dict[str, Any]) -> None:
        presets = self.settings["presets"].setdefault(command_key, [])
        now = now_epoch()
        idx = self._find_preset_index(command_key, name)
        if idx is None:
            presets.append(
                {"name": name, "args": args, "meta": {"created": now, "updated": now, "last_used": 0, "use_count": 0, "favorite": False}}
            )
            return
        meta = presets[idx].get("meta", {}) if isinstance(presets[idx], dict) else {}
        meta.setdefault("created", now)
        meta.setdefault("favorite", False)
        meta["updated"] = now
        presets[idx] = {"name": name, "args": args, "meta": meta}

    def delete_preset(self, command_key: str, name: str) -> None:
        presets = self.settings["presets"].setdefault(command_key, [])
        self.settings["presets"][command_key] = [p for p in presets if str(p.get("name", "")) != name]

    def rename_preset(self, command_key: str, old: str, new: str) -> None:
        idx = self._find_preset_index(command_key, old)
        if idx is None:
            return
        presets = self.settings["presets"].setdefault(command_key, [])
        presets[idx]["name"] = new
        meta = presets[idx].get("meta", {})
        if isinstance(meta, dict):
            meta["updated"] = now_epoch()

    def touch_preset_used(self, command_key: str, name: str) -> None:
        idx = self._find_preset_index(command_key, name)
        if idx is None:
            return
        presets = self.settings["presets"].setdefault(command_key, [])
        meta = presets[idx].get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        meta["last_used"] = now_epoch()
        meta["use_count"] = int(meta.get("use_count", 0) or 0) + 1
        meta.setdefault("favorite", False)
        presets[idx]["meta"] = meta

    def toggle_preset_favorite(self, command_key: str, name: str) -> Optional[bool]:
        idx = self._find_preset_index(command_key, name)
        if idx is None:
            return None
        presets = self.settings["presets"].setdefault(command_key, [])
        meta = presets[idx].get("meta", {})
        if not isinstance(meta, dict):
            meta = {"created": now_epoch(), "updated": now_epoch(), "last_used": 0, "use_count": 0, "favorite": False}
        cur = bool(meta.get("favorite", False))
        meta["favorite"] = not cur
        meta["updated"] = now_epoch()
        presets[idx]["meta"] = meta
        return bool(meta["favorite"])

    # ---- history ----
    def add_history(self, ts: str, command_key: str, label: str, args: Dict[str, Any]) -> int:
        hist = self.settings.setdefault("history", [])
        next_id = int(self.settings.get("history_next_id", 1) or 1)
        hist.append({"id": next_id, "ts": ts, "command_key": command_key, "label": label, "args": args})
        self.settings["history_next_id"] = next_id + 1

        limit = int(self.settings.get("history_limit", 200) or 200)
        if len(hist) > limit:
            del hist[: len(hist) - limit]
        return next_id

    def list_history(self) -> List[HistoryEntry]:
        raw = self.settings.get("history", [])
        out: List[HistoryEntry] = []
        for item in raw:
            try:
                out.append(
                    HistoryEntry(
                        id=int(item["id"]),
                        ts=str(item["ts"]),
                        command_key=str(item["command_key"]),
                        label=str(item["label"]),
                        args=dict(item["args"]),
                    )
                )
            except Exception:
                continue
        return out

    def delete_history_by_id(self, entry_id: int) -> None:
        hist = self.settings.setdefault("history", [])
        self.settings["history"] = [h for h in hist if not (isinstance(h, dict) and int(h.get("id", -1)) == entry_id)]

    def clear_history(self) -> None:
        self.settings["history"] = []

    # ---- import/export ----
    def export_presets_only(self) -> Dict[str, Any]:
        return {"version": 3, "presets": self.settings.get("presets", {})}

    def import_presets_only(self, payload: Dict[str, Any], *, replace: bool) -> None:
        presets = payload.get("presets", {})
        if not isinstance(presets, dict):
            raise ValueError("Invalid presets format")
        if replace:
            self.settings["presets"] = presets
        else:
            existing: Dict[str, Any] = self.settings.setdefault("presets", {})
            for ck, plist in presets.items():
                if not isinstance(plist, list):
                    continue
                existing.setdefault(ck, [])
                for p in plist:
                    if not isinstance(p, dict):
                        continue
                    name = p.get("name")
                    args = p.get("args")
                    if not isinstance(name, str) or not isinstance(args, dict):
                        continue
                    meta = p.get("meta", {})
                    if not isinstance(meta, dict):
                        meta = {"created": now_epoch(), "updated": now_epoch(), "last_used": 0, "use_count": 0, "favorite": False}
                    meta.setdefault("favorite", False)
                    replaced = False
                    for i, ep in enumerate(existing[ck]):
                        if str(ep.get("name", "")) == name:
                            existing[ck][i] = {"name": name, "args": args, "meta": meta}
                            replaced = True
                            break
                    if not replaced:
                        existing[ck].append({"name": name, "args": args, "meta": meta})
        self._migrate_presets()
