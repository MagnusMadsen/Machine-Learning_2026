from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ENCODINGS_PATH = Path("output/encodings.pkl")
DEFAULT_TRAINING_DIR = Path("training")
DEFAULT_VALIDATION_DIR = Path("validation")
DEFAULT_OUTPUT_DIR = Path("output")


@dataclass
class MatchResult:
    name: str
    box: Tuple[int, int, int, int]  # top, right, bottom, left


def iter_image_files(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def encode_known_faces(
    training_dir: Path,
    model: str = "hog",
    encodings_path: Path = ENCODINGS_PATH,
) -> None:
    encodings_path.parent.mkdir(parents=True, exist_ok=True)

    known_encodings: List[np.ndarray] = []
    known_names: List[str] = []

    for person_dir in training_dir.iterdir():
        if not person_dir.is_dir():
            continue

        for img_path in iter_image_files(person_dir):
            image = face_recognition.load_image_file(str(img_path))
            locations = face_recognition.face_locations(image, model=model)

            if len(locations) != 1:
                continue

            encoding = face_recognition.face_encodings(
                image, known_face_locations=locations
            )[0]

            known_encodings.append(encoding)
            known_names.append(person_dir.name)

    data = {"names": known_names, "encodings": known_encodings}
    encodings_path.write_bytes(pickle.dumps(data))
    print(f"[OK] Saved {len(known_names)} encodings")


def load_encodings(encodings_path: Path) -> Dict[str, List]:
    data = pickle.loads(encodings_path.read_bytes())
    data["encodings"] = [np.asarray(e) for e in data["encodings"]]
    return data


def recognize_faces_in_image(
    image_path: Path,
    data: Dict[str, List],
    model: str = "hog",
    tolerance: float = 0.6,
) -> List[MatchResult]:
    image = face_recognition.load_image_file(str(image_path))
    locations = face_recognition.face_locations(image, model=model)
    encodings = face_recognition.face_encodings(image, known_face_locations=locations)

    results: List[MatchResult] = []

    for box, face_encoding in zip(locations, encodings):
        matches = face_recognition.compare_faces(
            data["encodings"], face_encoding, tolerance=tolerance
        )

        name = "Unknown"
        if any(matches):
            distances = face_recognition.face_distance(
                data["encodings"], face_encoding
            )
            best = min(
                (i for i, m in enumerate(matches) if m),
                key=lambda i: distances[i],
            )
            name = data["names"][best]

        results.append(MatchResult(name=name, box=box))

    return results


def draw_results(
    image_path: Path,
    matches: List[MatchResult],
    out_path: Path,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=32)
    except Exception:
        font = ImageFont.load_default()

    for m in matches:
        top, right, bottom, left = m.box

        pad = 25
        top = max(0, top - pad)
        left = max(0, left - pad)
        bottom = min(img.height, bottom + pad)
        right = min(img.width, right + pad)

        draw.rectangle(((left, top), (right, bottom)), outline="red", width=6)

        text = m.name
        text_w, text_h = draw.textbbox((0, 0), text, font=font)[2:]
        label_pad = 10

        label_top = max(0, top - text_h - 2 * label_pad)
        label_bottom = top

        draw.rectangle(
            ((left, label_top), (left + text_w + 2 * label_pad, label_bottom)),
            outline="red",
            width=6,
        )

        draw.text(
            (left + label_pad, label_top + label_pad),
            text,
            fill="red",
            font=font,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[OK] Wrote {out_path}")


def make_suffix(matches: List[MatchResult]) -> str:
    """Generate filename suffix based on who is detected."""
    names_detected = {m.name.lower() for m in matches}
    suffixes = []
    for person in ["daniel", "magnus"]:
        suffixes.append(person if person in names_detected else f"no_{person}")
    return "_".join(suffixes)


def validate_folder(
    validation_dir: Path,
    output_dir: Path,
    data: Dict[str, List],
    model: str,
    tolerance: float,
) -> None:
    for img_path in iter_image_files(validation_dir):
        matches = recognize_faces_in_image(img_path, data, model, tolerance)

        suffix = make_suffix(matches)
        out_path = output_dir / f"validated_{img_path.stem}_{suffix}.jpg"
        draw_results(img_path, matches, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("-f", "--file")

    parser.add_argument("--model", default="hog")
    parser.add_argument("--tolerance", type=float, default=0.6)

    parser.add_argument("--training-dir", default=str(DEFAULT_TRAINING_DIR))
    parser.add_argument("--validation-dir", default=str(DEFAULT_VALIDATION_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--encodings-path", default=str(ENCODINGS_PATH))

    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    validation_dir = Path(args.validation_dir)
    output_dir = Path(args.output_dir)
    encodings_path = Path(args.encodings_path)

    if args.train:
        encode_known_faces(training_dir, args.model, encodings_path)
        return

    data = load_encodings(encodings_path)

    if args.validate:
        validate_folder(validation_dir, output_dir, data, args.model, args.tolerance)
        return

    if args.test:
        image_path = Path(args.file)
        matches = recognize_faces_in_image(image_path, data, args.model, args.tolerance)

        suffix = make_suffix(matches)
        out_path = output_dir / f"tested_{image_path.stem}_{suffix}.jpg"
        draw_results(image_path, matches, out_path)
        return

    raise SystemExit("Pick --train, --validate, or --test")


if __name__ == "__main__":
    main()
