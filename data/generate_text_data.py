"""
Generates a synthetic passenger query dataset from hand-written templates
per intent, each with an entity slot (gate number, terminal, service name)
filled from a small value list. Keeps the corpus realistic — it mirrors
real airport FAQ phrasing — while staying fully reproducible.

Run: python data/generate_text_data.py
Output: data/text_queries.csv
"""
import csv
import random

random.seed(42)

GATES = ["A05", "B12", "A11", "B03", "C07"]
TERMINALS = ["Terminal 1", "Terminal 2"]

# intent -> list of (template, entity_type or None)
TEMPLATES = {
    "find_gate": [
        "Where is gate {gate}?",
        "How do I get to gate {gate}?",
        "Which way is gate {gate} from here?",
        "Can you direct me to gate {gate}?",
        "I need directions to gate {gate}.",
        "Excuse me, where can I find gate {gate}?",
    ],
    "baggage_claim": [
        "How do I get to baggage claim?",
        "Where is the baggage claim hall?",
        "Which belt is my baggage on?",
        "Where do I collect my luggage?",
        "Can you show me the way to the baggage reclaim area?",
    ],
    "check_in": [
        "Where is the check-in desk for international flights?",
        "Where can I check in for my flight?",
        "How do I find the check-in area in {terminal}?",
        "Where is the check-in counter for {terminal}?",
        "I need to check in, where should I go?",
    ],
    "security": [
        "Where is the security checkpoint?",
        "How do I get through security?",
        "Where is the security control area?",
        "Which way to airport security?",
    ],
    "lounge": [
        "Is there a lounge near {terminal}?",
        "Where is the airport lounge?",
        "How do I get to the business lounge?",
        "Is there a lounge close to gate {gate}?",
    ],
    "restaurant": [
        "Where can I get something to eat?",
        "Is there a restaurant near here?",
        "Where is the food court?",
        "I'm hungry, where can I find a restaurant?",
    ],
    "transport": [
        "Where can I find airport transport?",
        "How do I get to the train station?",
        "Where is the taxi pickup zone?",
        "How do I get into the city centre from here?",
        "Where can I rent a car?",
    ],
    "information_desk": [
        "Where is the nearest information desk?",
        "Who can I ask for general information?",
        "Where can I get help with directions?",
        "Is there an information point nearby?",
    ],
    "lost_and_found": [
        "I lost my bag, where do I report it?",
        "Where is the lost and found office?",
        "I lost an item at the airport, who do I contact?",
        "Where do I go to report lost property?",
    ],
    "prayer_room": [
        "Is there a prayer room in the airport?",
        "Where can I find a quiet room to pray?",
        "Where is the multi-faith prayer room?",
    ],
    "accessibility": [
        "I need a wheelchair, where do I go?",
        "Where is the special assistance desk?",
        "Can someone help me with mobility assistance?",
        "Where do I request accessibility support?",
    ],
    "flight_status": [
        "Is my flight delayed?",
        "What is the status of flight {flight}?",
        "Has my gate changed?",
        "Is flight {flight} on time?",
    ],
    "shopping": [
        "Where is the duty free shop?",
        "Where can I buy souvenirs?",
        "Is there a shopping area past security?",
    ],
    "medical": [
        "Is there a pharmacy in the airport?",
        "Where is the first aid point?",
        "I need medical assistance, where do I go?",
    ],
}

FLIGHTS = ["LH441", "BA218", "AF1023", "EW772"]


def fill(template: str) -> tuple[str, dict]:
    entities = {}
    text = template
    if "{gate}" in text:
        g = random.choice(GATES)
        text = text.replace("{gate}", g)
        entities["gate"] = g
    if "{terminal}" in text:
        t = random.choice(TERMINALS)
        text = text.replace("{terminal}", t)
        entities["terminal"] = t
    if "{flight}" in text:
        f = random.choice(FLIGHTS)
        text = text.replace("{flight}", f)
        entities["flight"] = f
    return text, entities


def main():
    rows = []
    qid = 1
    for intent, templates in TEMPLATES.items():
        # 6 samples per intent (some templates reused with different entity values)
        for _ in range(6):
            template = random.choice(templates)
            text, entities = fill(template)
            rows.append({
                "id": f"Q{qid:03d}",
                "text": text,
                "intent": intent,
                "entities": entities,
            })
            qid += 1

    random.shuffle(rows)
    split_point = int(len(rows) * 0.8)
    for i, row in enumerate(rows):
        row["split"] = "train" if i < split_point else "val"

    out_path = "data/text_queries.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "intent", "entities", "split"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} queries ({split_point} train / {len(rows)-split_point} val) to {out_path}")
    print(f"Intents: {len(TEMPLATES)}")


if __name__ == "__main__":
    main()
