"""
Structured lookup for airport locations/services — not a trained model. The
vision, text and fusion modules turn a recognised category/intent/entity
into a concrete answer by querying this. JSON keeps it easy to hand-edit and
easy to swap for SQLite later without touching any calling code.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"


@dataclass
class KBRecord:
    id: str
    name: str
    category: str
    terminal: str
    floor_zone: str
    description: str
    opening_hours: str
    directions: str
    accessibility: str
    related_facilities: list
    emergency_contact: str

    def to_response_text(self) -> str:
        return (
            f"**{self.name}** ({self.terminal}, {self.floor_zone})\n\n"
            f"{self.description}\n\n"
            f"**Directions:** {self.directions}\n"
            f"**Opening hours:** {self.opening_hours}\n"
            f"**Accessibility:** {self.accessibility}"
        )


class KnowledgeBase:
    """In-memory lookup over the airport records, indexed by category for
    fast routing from vision/text predictions to a concrete KB entry."""

    def __init__(self, path: Path = KB_PATH):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.records = [KBRecord(**r) for r in raw]
        self._by_category = {}
        self._by_id = {}
        for r in self.records:
            self._by_category.setdefault(r.category, []).append(r)
            self._by_id[r.id] = r

    def get_by_id(self, record_id: str) -> Optional[KBRecord]:
        return self._by_id.get(record_id)

    def get_by_category(self, category: str, terminal: Optional[str] = None) -> Optional[KBRecord]:
        """Return the best-matching record for a predicted category, optionally
        narrowed by a terminal entity extracted from the query (e.g. 'Terminal 2')."""
        candidates = self._by_category.get(category, [])
        if not candidates:
            return None
        if terminal:
            for c in candidates:
                if terminal.lower() in c.terminal.lower():
                    return c
        return candidates[0]

    def get_by_gate(self, gate_code: str) -> Optional[KBRecord]:
        gate_code = gate_code.upper().strip()
        for r in self.records:
            if r.category == "gate" and gate_code in r.name.upper():
                return r
        return None

    def categories(self):
        return sorted(self._by_category.keys())


if __name__ == "__main__":
    kb = KnowledgeBase()
    print(f"Loaded {len(kb.records)} records across {len(kb.categories())} categories")
    print(kb.get_by_category("baggage_claim").to_response_text())
