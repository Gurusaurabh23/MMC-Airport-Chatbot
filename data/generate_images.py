"""
Generates a synthetic airport sign dataset instead of using real photos,
which would raise copyright/privacy issues (people, boarding passes, live
flight data in frame). Each image is a programmatically drawn wayfinding
sign (pictogram + label + directional arrow) in the ISO 7001 style used at
most international airports, with randomised colour, arrow direction and
camera-style jitter/rotation to simulate photos taken at different angles
and lighting levels.

Run: python data/generate_images.py
Output: data/images/<category>_<n>.png  (+ data/image_labels.csv)
"""
import csv
import math
import random

from PIL import Image, ImageDraw, ImageFont, ImageFilter

random.seed(7)

CANVAS = 384
IMAGES_PER_CLASS = 8

# (category, background colour, pictogram drawer name, label text)
CATEGORIES = [
    ("gate", (0, 90, 160), "pictogram_gate", "GATE"),
    ("baggage_claim", (0, 120, 70), "pictogram_bag", "BAGGAGE CLAIM"),
    ("check_in", (0, 90, 160), "pictogram_checkin", "CHECK-IN"),
    ("security", (150, 20, 20), "pictogram_security", "SECURITY"),
    ("information_desk", (0, 90, 160), "pictogram_info", "INFORMATION"),
    ("lost_and_found", (120, 90, 0), "pictogram_lost", "LOST & FOUND"),
    ("lounge", (90, 40, 130), "pictogram_lounge", "LOUNGE"),
    ("restaurant", (150, 90, 0), "pictogram_restaurant", "RESTAURANT"),
    ("transport", (0, 90, 160), "pictogram_transport", "TRANSPORT"),
    ("prayer_room", (0, 110, 90), "pictogram_prayer", "PRAYER ROOM"),
    ("accessibility", (0, 90, 160), "pictogram_wheelchair", "ASSISTANCE"),
    ("shopping", (120, 0, 90), "pictogram_shop", "DUTY FREE"),
    ("medical", (170, 0, 0), "pictogram_medical", "MEDICAL"),
]


def _white(draw, xy, **kw):
    draw.line(xy, fill="white", **kw)


def pictogram_gate(d, cx, cy, s):
    # simple aircraft silhouette (fuselage + wings + tail)
    d.line([(cx - s, cy), (cx + s, cy)], fill="white", width=int(s * 0.16))
    d.polygon([(cx + s * 0.2, cy - s * 0.7), (cx + s * 0.55, cy), (cx + s * 0.2, cy + s * 0.7)], fill="white")
    d.polygon([(cx - s * 0.8, cy - s * 0.35), (cx - s * 0.35, cy), (cx - s * 0.8, cy + s * 0.35)], fill="white")


def pictogram_bag(d, cx, cy, s):
    d.rounded_rectangle([cx - s * 0.7, cy - s * 0.4, cx + s * 0.7, cy + s * 0.8], radius=s * 0.15, outline="white", width=int(s * 0.12))
    d.arc([cx - s * 0.35, cy - s * 0.9, cx + s * 0.35, cy - s * 0.2], 180, 360, fill="white", width=int(s * 0.12))


def pictogram_checkin(d, cx, cy, s):
    d.rectangle([cx - s * 0.8, cy - s * 0.1, cx + s * 0.8, cy + s * 0.5], outline="white", width=int(s * 0.12))
    d.ellipse([cx - s * 0.3, cy - s * 0.9, cx + s * 0.3, cy - s * 0.3], outline="white", width=int(s * 0.1))


def pictogram_security(d, cx, cy, s):
    d.polygon([(cx, cy - s), (cx + s * 0.8, cy - s * 0.4), (cx + s * 0.8, cy + s * 0.4), (cx, cy + s),
               (cx - s * 0.8, cy + s * 0.4), (cx - s * 0.8, cy - s * 0.4)], outline="white", width=int(s * 0.12))
    d.line([(cx - s * 0.3, cy), (cx - s * 0.05, cy + s * 0.3), (cx + s * 0.4, cy - s * 0.35)], fill="white", width=int(s * 0.12))


def pictogram_info(d, cx, cy, s):
    d.ellipse([cx - s, cy - s, cx + s, cy + s], outline="white", width=int(s * 0.14))
    d.ellipse([cx - s * 0.12, cy - s * 0.55, cx + s * 0.12, cy - s * 0.3], fill="white")
    d.rectangle([cx - s * 0.1, cy - s * 0.15, cx + s * 0.1, cy + s * 0.55], fill="white")


def pictogram_lost(d, cx, cy, s):
    d.ellipse([cx - s, cy - s, cx + s, cy + s], outline="white", width=int(s * 0.14))
    d.line([(cx - s * 0.4, cy - s * 0.4), (cx + s * 0.4, cy + s * 0.4)], fill="white", width=int(s * 0.14))
    d.line([(cx + s * 0.4, cy - s * 0.4), (cx - s * 0.4, cy + s * 0.4)], fill="white", width=int(s * 0.14))


def pictogram_lounge(d, cx, cy, s):
    d.rounded_rectangle([cx - s * 0.7, cy - s * 0.1, cx + s * 0.7, cy + s * 0.6], radius=s * 0.2, outline="white", width=int(s * 0.12))
    d.line([(cx - s * 0.7, cy + 0.1 * s), (cx - s * 0.9, cy - s * 0.3)], fill="white", width=int(s * 0.12))
    d.line([(cx + s * 0.7, cy + 0.1 * s), (cx + s * 0.9, cy - s * 0.3)], fill="white", width=int(s * 0.12))


def pictogram_restaurant(d, cx, cy, s):
    d.line([(cx - s * 0.5, cy - s), (cx - s * 0.5, cy + s)], fill="white", width=int(s * 0.1))
    d.line([(cx - s * 0.65, cy - s), (cx - s * 0.65, cy - s * 0.3)], fill="white", width=int(s * 0.06))
    d.line([(cx - s * 0.35, cy - s), (cx - s * 0.35, cy - s * 0.3)], fill="white", width=int(s * 0.06))
    d.ellipse([cx + s * 0.2, cy - s, cx + s * 0.7, cy - s * 0.4], outline="white", width=int(s * 0.1))
    d.line([(cx + s * 0.45, cy - s * 0.4), (cx + s * 0.45, cy + s)], fill="white", width=int(s * 0.1))


def pictogram_transport(d, cx, cy, s):
    d.rounded_rectangle([cx - s * 0.9, cy - s * 0.4, cx + s * 0.9, cy + s * 0.3], radius=s * 0.15, outline="white", width=int(s * 0.12))
    d.ellipse([cx - s * 0.55, cy + s * 0.15, cx - s * 0.15, cy + s * 0.55], fill="white")
    d.ellipse([cx + s * 0.15, cy + s * 0.15, cx + s * 0.55, cy + s * 0.55], fill="white")


def pictogram_prayer(d, cx, cy, s):
    d.arc([cx - s * 0.7, cy - s * 0.9, cx + s * 0.7, cy + s * 0.9], 200, 340, fill="white", width=int(s * 0.14))
    d.polygon([(cx, cy - s * 0.9), (cx - s * 0.12, cy - s * 0.6), (cx + s * 0.12, cy - s * 0.6)], fill="white")


def pictogram_wheelchair(d, cx, cy, s):
    d.ellipse([cx - s * 0.1, cy - s, cx + s * 0.25, cy - s * 0.65], fill="white")
    d.arc([cx - s * 0.5, cy - s * 0.2, cx + s * 0.5, cy + s * 0.8], 0, 180, fill="white", width=int(s * 0.15))
    d.line([(cx - s * 0.1, cy - s * 0.4), (cx + s * 0.4, cy - s * 0.4), (cx + s * 0.6, cy + s * 0.2)], fill="white", width=int(s * 0.1))


def pictogram_shop(d, cx, cy, s):
    d.polygon([(cx - s * 0.7, cy - s * 0.3), (cx + s * 0.7, cy - s * 0.3), (cx + s * 0.55, cy + s * 0.8), (cx - s * 0.55, cy + s * 0.8)], outline="white", width=int(s * 0.1))
    d.arc([cx - s * 0.35, cy - s * 0.8, cx + s * 0.35, cy - s * 0.1], 180, 360, fill="white", width=int(s * 0.1))


def pictogram_medical(d, cx, cy, s):
    d.rectangle([cx - s * 0.25, cy - s, cx + s * 0.25, cy + s], fill="white")
    d.rectangle([cx - s, cy - s * 0.25, cx + s, cy + s * 0.25], fill="white")


PICTOGRAMS = {name: fn for name, fn in list(globals().items()) if name.startswith("pictogram_")}


def draw_sign(category_idx, category, bg_color, drawer_name, label, variant):
    img = Image.new("RGB", (CANVAS, CANVAS), bg_color)
    draw = ImageDraw.Draw(img)

    # border
    draw.rectangle([6, 6, CANVAS - 6, CANVAS - 6], outline="white", width=6)

    # pictogram
    PICTOGRAMS[drawer_name](draw, CANVAS // 2, int(CANVAS * 0.4), CANVAS * 0.22)

    # directional arrow (varies)
    arrow_dir = random.choice(["left", "right", "straight"])
    ay = int(CANVAS * 0.68)
    if arrow_dir == "left":
        draw.polygon([(CANVAS // 2 - 60, ay), (CANVAS // 2 - 20, ay - 18), (CANVAS // 2 - 20, ay + 18)], fill="white")
        draw.line([(CANVAS // 2 - 20, ay), (CANVAS // 2 + 60, ay)], fill="white", width=8)
    elif arrow_dir == "right":
        draw.polygon([(CANVAS // 2 + 60, ay), (CANVAS // 2 + 20, ay - 18), (CANVAS // 2 + 20, ay + 18)], fill="white")
        draw.line([(CANVAS // 2 - 60, ay), (CANVAS // 2 + 20, ay)], fill="white", width=8)
    else:
        draw.polygon([(CANVAS // 2, ay - 40), (CANVAS // 2 - 18, ay), (CANVAS // 2 + 18, ay)], fill="white")
        draw.line([(CANVAS // 2, ay), (CANVAS // 2, ay + 40)], fill="white", width=8)

    # label text
    try:
        font = ImageFont.truetype("arialbd.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    text = label
    if category == "gate":
        text = f"GATE {random.choice(['A05','B12','A11','B03','C07'])}"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((CANVAS - tw) / 2, CANVAS * 0.85), text, fill="white", font=font)

    # --- camera-style augmentation to simulate real photo conditions ---
    angle = random.uniform(-8, 8)
    img = img.rotate(angle, expand=False, fillcolor=bg_color)
    brightness = random.uniform(0.75, 1.25)
    img = Image.eval(img, lambda p: min(255, max(0, int(p * brightness))))
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
    # slight crop jitter to mimic imperfect framing
    crop_px = random.randint(0, 14)
    img = img.crop((crop_px, crop_px, CANVAS - crop_px, CANVAS - crop_px)).resize((CANVAS, CANVAS))

    return img, text


def main():
    rows = []
    for cat_idx, (category, bg, drawer, label) in enumerate(CATEGORIES):
        for i in range(IMAGES_PER_CLASS):
            img, text = draw_sign(cat_idx, category, bg, drawer, label, i)
            fname = f"data/images/{category}_{i:02d}.png"
            img.save(fname)
            rows.append({"filename": fname, "category": category, "label_text": text})

    with open("data/image_labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "category", "label_text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} images across {len(CATEGORIES)} categories -> data/image_labels.csv")


if __name__ == "__main__":
    main()
