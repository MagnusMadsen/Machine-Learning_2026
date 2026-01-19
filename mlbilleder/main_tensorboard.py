from __future__ import annotations

import argparse
import re
import time
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorboardX import SummaryWriter


ENCODINGS_PATH = Path("output/encodings.pkl")
DEFAULT_TRAINING_DIR = Path("training")
DEFAULT_VALIDATION_DIR = Path("validation")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_LOG_DIR = Path("runs")

# Keeps TensorBoard x-axis clean (1,2,3...) instead of huge timestamps
RUN_COUNTER_PATH = Path("output/validate_run_counter.txt")


@dataclass
class MatchResult:
    name: str
    box: Tuple[int, int, int, int]  # top, right, bottom, left
    best_distance: float
    margin: float  # second_best - best


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def next_validate_run_step() -> int:
    RUN_COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RUN_COUNTER_PATH.exists():
        try:
            n = int(RUN_COUNTER_PATH.read_text(encoding="utf-8").strip())
        except Exception:
            n = 0
    else:
        n = 0
    n += 1
    RUN_COUNTER_PATH.write_text(str(n), encoding="utf-8")
    return n


def iter_image_files(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def encode_known_faces(
    training_dir: Path,
    model: str = "hog",
    encodings_path: Path = ENCODINGS_PATH,
    writer: SummaryWriter | None = None,
) -> None:
    encodings_path.parent.mkdir(parents=True, exist_ok=True)

    known_encodings: List[np.ndarray] = []
    known_names: List[str] = []

    total = used = skipped = 0
    t0 = time.time()

    for person_dir in training_dir.iterdir():
        if not person_dir.is_dir():
            continue

        for img_path in iter_image_files(person_dir):
            total += 1
            image = face_recognition.load_image_file(str(img_path))
            locations = face_recognition.face_locations(image, model=model)

            if len(locations) != 1:
                skipped += 1
                continue

            encoding = face_recognition.face_encodings(
                image, known_face_locations=locations
            )[0]

            known_encodings.append(encoding)
            known_names.append(person_dir.name)
            used += 1

    data = {"names": known_names, "encodings": known_encodings}
    encodings_path.write_bytes(pickle.dumps(data))
    print(f"[OK] Saved {len(known_names)} encodings")

    if writer:
        dt = time.time() - t0
        writer.add_scalar("train/total_images", total, 0)
        writer.add_scalar("train/used_images", used, 0)
        writer.add_scalar("train/skipped_images", skipped, 0)
        writer.add_scalar("train/seconds", dt, 0)
        writer.flush()


def load_encodings(encodings_path: Path) -> Dict[str, List]:
    if not encodings_path.exists():
        raise FileNotFoundError(
            f"Encodings file not found: {encodings_path}. Run: python main.py --train"
        )
    if encodings_path.stat().st_size == 0:
        raise ValueError(
            f"Encodings file is empty: {encodings_path}. Delete it and run: python main.py --train"
        )

    data = pickle.loads(encodings_path.read_bytes())
    data["encodings"] = [np.asarray(e) for e in data["encodings"]]
    return data


def _best_distance_and_margin(distances: np.ndarray) -> Tuple[float, float, int]:
    if distances.size == 0:
        return 1.0, 0.0, -1
    order = np.argsort(distances)
    best_i = int(order[0])
    best = float(distances[best_i])
    second = float(distances[order[1]]) if distances.size > 1 else 1.0
    margin = second - best
    return best, margin, best_i


def recognize_faces_in_image(
    image_path: Path,
    data: Dict[str, List],
    model: str = "hog",
    tolerance: float = 0.6,
) -> List[MatchResult]:
    image = face_recognition.load_image_file(str(image_path))
    locations = face_recognition.face_locations(image, model=model)
    encodings = face_recognition.face_encodings(image, known_face_locations=locations)

    known_encs = data["encodings"]
    known_names = data["names"]

    results: List[MatchResult] = []
    for box, face_encoding in zip(locations, encodings):
        distances = face_recognition.face_distance(known_encs, face_encoding)
        best_dist, margin, best_i = _best_distance_and_margin(distances)

        if best_i >= 0 and best_dist <= tolerance:
            name = known_names[best_i]
        else:
            name = "Unknown"

        results.append(
            MatchResult(
                name=name,
                box=box,
                best_distance=best_dist,
                margin=margin,
            )
        )

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

        text = f"{m.name}  d={m.best_distance:.3f}"
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
    names_detected = {m.name.lower() for m in matches}
    suffixes = []
    for person in ["daniel", "magnus"]:
        suffixes.append(person if person in names_detected else f"no_{person}")
    return "_".join(suffixes)


def _log_image(writer: SummaryWriter, tag: str, img_path: Path, step: int) -> None:
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).transpose(2, 0, 1)  # CHW
    writer.add_image(tag, arr, step)


def _infer_label_from_filename(
    img_path: Path,
    pos_re: re.Pattern,
    neg_re: re.Pattern,
) -> Optional[bool]:
    """
    Returns:
      True  => labeled as Daniel
      False => labeled as not-Daniel
      None  => unlabeled/unknown
    """
    name = img_path.stem.lower()
    if neg_re.search(name):
        return False
    if pos_re.search(name):
        return True
    return None


def _confusion_image(tp: int, fp: int, fn: int, tn: int) -> Image.Image:
    """
    Make a simple 2x2 confusion matrix image you can view in TensorBoard.
    Layout:
        Pred Daniel   Pred Not
    True D     TP        FN
    True !D    FP        TN
    """
    w, h = 800, 600
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    try:
        font_big = ImageFont.truetype("arial.ttf", 36)
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font_big = ImageFont.load_default()
        font = ImageFont.load_default()

    # grid
    left, top = 50, 80
    cell_w, cell_h = 325, 200

    # headers
    draw.text((left + 80, 20), "Pred: Daniel", fill="black", font=font)
    draw.text((left + cell_w + 80, 20), "Pred: Not", fill="black", font=font)
    draw.text((10, top + 70), "True: Daniel", fill="black", font=font)
    draw.text((10, top + cell_h + 70), "True: Not", fill="black", font=font)

    # cells: (x0,y0,x1,y1)
    cells = {
        "TP": (left, top, left + cell_w, top + cell_h),
        "FN": (left + cell_w, top, left + 2 * cell_w, top + cell_h),
        "FP": (left, top + cell_h, left + cell_w, top + 2 * cell_h),
        "TN": (left + cell_w, top + cell_h, left + 2 * cell_w, top + 2 * cell_h),
    }
    for _, box in cells.items():
        draw.rectangle(box, outline="black", width=3)

    total = tp + fp + fn + tn
    def pct(x: int) -> str:
        return f"{(x / total * 100):.1f}%" if total else "0.0%"

    # write counts
    def center_text(box, title: str, value: int):
        x0, y0, x1, y1 = box
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        draw.text((cx - 40, cy - 40), title, fill="black", font=font_big)
        draw.text((cx - 60, cy + 5), f"{value}", fill="black", font=font_big)
        draw.text((cx - 50, cy + 55), pct(value), fill="black", font=font)

    center_text(cells["TP"], "TP", tp)
    center_text(cells["FN"], "FN", fn)
    center_text(cells["FP"], "FP", fp)
    center_text(cells["TN"], "TN", tn)

    return img


def validate_folder(
    validation_dir: Path,
    output_dir: Path,
    data: Dict[str, List],
    model: str,
    tolerance: float,
    writer: SummaryWriter | None = None,
    log_images: bool = True,
    topk: int = 10,
    target_name: str = "daniel",
    # filename label patterns:
    pos_pattern: str = r"(^|[_-])daniel([_-]|$)",
    neg_pattern: str = r"(^|[_-])(notdaniel|no_daniel|random)([_-]|$)",
) -> None:
    target = target_name.lower()
    img_paths = iter_image_files(validation_dir)
    if not img_paths:
        print(f"[WARN] No images found in {validation_dir}")
        return

    pos_re = re.compile(pos_pattern, re.IGNORECASE)
    neg_re = re.compile(neg_pattern, re.IGNORECASE)

    images_total = len(img_paths)

    # Batch-level counters (easy graphs)
    images_no_faces = 0
    images_with_faces = 0
    images_with_target = 0
    images_unknown_only = 0
    target_confidences: List[float] = []

    # Confusion matrix (per image)
    tp = fp = fn = tn = 0
    unlabeled = 0

    # For useful examples
    closest_unknowns: List[Tuple[float, Path]] = []  # (distance, out_image)
    weak_targets: List[Tuple[float, Path]] = []      # (distance, out_image)
    false_pos: List[Path] = []
    false_neg: List[Path] = []

    t0 = time.time()

    for img_path in img_paths:
        matches = recognize_faces_in_image(img_path, data, model, tolerance)

        # Predicted "Daniel present?" per image
        pred_has_target = any(m.name.lower() == target for m in matches)

        # Batch stats
        if len(matches) == 0:
            images_no_faces += 1
        else:
            images_with_faces += 1
            if pred_has_target:
                images_with_target += 1
                for m in matches:
                    if m.name.lower() == target:
                        conf = clamp01(1.0 - (m.best_distance / float(tolerance)))
                        target_confidences.append(conf)
            else:
                images_unknown_only += 1

        # Save output image with boxes
        suffix = make_suffix(matches)
        out_path = output_dir / f"validated_{img_path.stem}_{suffix}.jpg"
        draw_results(img_path, matches, out_path)

        # Example collection
        for m in matches:
            if m.name == "Unknown":
                closest_unknowns.append((m.best_distance, out_path))
            if m.name.lower() == target:
                weak_targets.append((m.best_distance, out_path))

        # Confusion matrix labeling from filename
        gt = _infer_label_from_filename(img_path, pos_re, neg_re)
        if gt is None:
            unlabeled += 1
        else:
            # gt True => Daniel
            # gt False => not Daniel
            if gt is True and pred_has_target is True:
                tp += 1
            elif gt is True and pred_has_target is False:
                fn += 1
                false_neg.append(out_path)
            elif gt is False and pred_has_target is True:
                fp += 1
                false_pos.append(out_path)
            else:
                tn += 1

    dt = time.time() - t0

    # Four easy batch metrics
    pct_images_with_target = images_with_target / images_total if images_total else 0.0
    pct_images_unknown_only = images_unknown_only / images_total if images_total else 0.0
    pct_images_no_faces = images_no_faces / images_total if images_total else 0.0
    avg_confidence_target = float(np.mean(target_confidences)) if target_confidences else 0.0

    # Optional: combined score
    batch_score = pct_images_with_target * avg_confidence_target

    # Confusion derived metrics (only on labeled subset)
    labeled_total = tp + fp + fn + tn
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = ((tp + tn) / labeled_total) if labeled_total else 0.0

    print("\n=== VALIDATE BATCH SUMMARY ===")
    print(f"Images total:                 {images_total}")
    print(f"Images with NO faces:         {images_no_faces} ({pct_images_no_faces*100:.1f}%)")
    print(f"Images predicted {target}:    {images_with_target} ({pct_images_with_target*100:.1f}%)")
    print(f"Images w/ faces, no {target}: {images_unknown_only} ({pct_images_unknown_only*100:.1f}%)")
    print(f"Avg confidence ({target}):    {avg_confidence_target:.3f} (0..1)")
    print(f"Batch score:                  {batch_score:.3f} (0..1)")
    print("")
    print("=== CONFUSION (from filename labels) ===")
    print(f"Labeled images: {labeled_total}   Unlabeled images: {unlabeled}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}  Acc={accuracy:.3f}")
    print(f"Seconds: {dt:.2f}")
    print("=======================================\n")

    if writer:
        step = next_validate_run_step()

        # Batch graphs (easy)
        writer.add_scalar("batch/images_total", images_total, step)
        writer.add_scalar("batch/pct_images_no_faces", pct_images_no_faces, step)
        writer.add_scalar(f"batch/pct_images_with_{target}", pct_images_with_target, step)
        writer.add_scalar(f"batch/pct_images_faces_no_{target}", pct_images_unknown_only, step)
        writer.add_scalar(f"batch/avg_confidence_{target}", avg_confidence_target, step)
        writer.add_scalar("batch/batch_score", batch_score, step)
        writer.add_scalar("batch/seconds", dt, step)

        # Confusion matrix + metrics
        writer.add_scalar("confusion/labeled_total", labeled_total, step)
        writer.add_scalar("confusion/unlabeled", unlabeled, step)

        writer.add_scalar("confusion/TP", tp, step)
        writer.add_scalar("confusion/FP", fp, step)
        writer.add_scalar("confusion/FN", fn, step)
        writer.add_scalar("confusion/TN", tn, step)

        writer.add_scalar("metrics/precision", precision, step)
        writer.add_scalar("metrics/recall", recall, step)
        writer.add_scalar("metrics/f1", f1, step)
        writer.add_scalar("metrics/accuracy", accuracy, step)

        # Confusion image
        cm_img = _confusion_image(tp, fp, fn, tn)
        cm_arr = np.array(cm_img).transpose(2, 0, 1)  # CHW
        writer.add_image("confusion/matrix", cm_arr, step)

        # Example images
        if log_images:
            # Near-misses in general
            closest_unknowns.sort(key=lambda x: x[0])  # low distance first
            for i, (dist, outp) in enumerate(closest_unknowns[:topk]):
                _log_image(writer, f"examples/closest_unknowns/{i}_d={dist:.3f}", outp, step * 1000 + i)

            weak_targets.sort(key=lambda x: x[0], reverse=True)  # high distance first
            for i, (dist, outp) in enumerate(weak_targets[:topk]):
                _log_image(writer, f"examples/weak_{target}/{i}_d={dist:.3f}", outp, step * 1000 + 100 + i)

            # Confusion examples (most actionable)
            for i, outp in enumerate(false_pos[:topk]):
                _log_image(writer, f"errors/false_positive/{i}", outp, step * 1000 + 200 + i)
            for i, outp in enumerate(false_neg[:topk]):
                _log_image(writer, f"errors/false_negative/{i}", outp, step * 1000 + 300 + i)

        writer.flush()


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

    # TensorBoard options
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--no-tb", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument("--no-tb-images", action="store_true", help="Don't log images to TensorBoard")
    parser.add_argument("--tb-topk", type=int, default=10, help="Top-K example images to log")
    parser.add_argument("--target-name", default="daniel", help="Name to track in batch metrics")

    # filename label patterns (advanced, but you likely won't touch these)
    parser.add_argument("--pos-pattern", default=r"(^|[_-])daniel([_-]|$)")
    parser.add_argument("--neg-pattern", default=r"(^|[_-])(notdaniel|no_daniel|random)([_-]|$)")

    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    validation_dir = Path(args.validation_dir)
    output_dir = Path(args.output_dir)
    encodings_path = Path(args.encodings_path)

    writer: Optional[SummaryWriter] = None
    if not args.no_tb:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        run_dir = log_dir / time.strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(logdir=str(run_dir))
        print(f"[TB] Logging to {run_dir}")
        writer.add_scalar("debug/hello", 1, 0)
        writer.flush()

    try:
        if args.train:
            encode_known_faces(training_dir, args.model, encodings_path, writer=writer)
            return

        data = load_encodings(encodings_path)

        if args.validate:
            validate_folder(
                validation_dir,
                output_dir,
                data,
                args.model,
                args.tolerance,
                writer=writer,
                log_images=not args.no_tb_images,
                topk=args.tb_topk,
                target_name=args.target_name,
                pos_pattern=args.pos_pattern,
                neg_pattern=args.neg_pattern,
            )
            return

        if args.test:
            image_path = Path(args.file)
            matches = recognize_faces_in_image(image_path, data, args.model, args.tolerance)

            suffix = make_suffix(matches)
            out_path = output_dir / f"tested_{image_path.stem}_{suffix}.jpg"
            draw_results(image_path, matches, out_path)

            if writer and not args.no_tb_images:
                _log_image(writer, "test/result", out_path, int(time.time()))
                writer.flush()

            return

        raise SystemExit("Pick --train, --validate, or --test")
    finally:
        if writer:
            writer.close()


if __name__ == "__main__":
    main()
