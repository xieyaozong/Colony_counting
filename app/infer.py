from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


IMAGE_SIZE = 1280
TILE_SIZE = 1536
TILE_OVERLAP = 0.30
CONF_THRESHOLD = 0.205
TILE_IOU_THRESHOLD = 0.35
GLOBAL_MERGE_THRESHOLD = 0.525
MAX_DETECTIONS_PER_TILE = 1000
MAX_DETECTIONS = 2000
DISH_RADIUS_SCALE = 0.985

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".heic", ".webp", ".tif", ".tiff"}
Detection = tuple[tuple[int, int, int, int], float]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image

    try:
        from PIL import Image
        import pillow_heif

        pillow_heif.register_heif_opener()
        with Image.open(path) as pil_image:
            rgb = np.array(pil_image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise RuntimeError(f"Failed to read image: {path}") from exc


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(path), image):
        return

    try:
        from PIL import Image
        import pillow_heif

        pillow_heif.register_heif_opener()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to write image: {path}") from exc


def letterbox(image: np.ndarray, size: int = IMAGE_SIZE) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)

    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, scale, pad_x, pad_y


def preprocess(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    padded, scale, pad_x, pad_y = letterbox(image)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return tensor[None], scale, pad_x, pad_y


def make_session(model_path: Path) -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def iter_tiles(image: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
    height, width = image.shape[:2]
    stride = max(1, round(TILE_SIZE * (1.0 - TILE_OVERLAP)))
    tiles = []
    for y1 in tile_starts(height, TILE_SIZE, stride):
        for x1 in tile_starts(width, TILE_SIZE, stride):
            y2 = min(y1 + TILE_SIZE, height)
            x2 = min(x1 + TILE_SIZE, width)
            tiles.append((image[y1:y2, x1:x2], x1, y1))
    return tiles


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def overlap_values(box: np.ndarray, others: np.ndarray, metric: str) -> np.ndarray:
    xx1 = np.maximum(box[0], others[:, 0])
    yy1 = np.maximum(box[1], others[:, 1])
    xx2 = np.minimum(box[2], others[:, 2])
    yy2 = np.minimum(box[3], others[:, 3])

    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    other_area = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(
        0.0, others[:, 3] - others[:, 1]
    )

    if metric == "containment":
        return inter / (np.minimum(box_area, other_area) + 1e-9)
    return inter / (box_area + other_area - inter + 1e-9)


def nms_indices(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    metric: str,
    max_detections: int,
) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []

    while order.size and len(keep) < max_detections:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        overlaps = overlap_values(boxes[current], boxes[rest], metric)
        order = rest[overlaps <= threshold]

    return keep


def postprocess_tile(
    prediction: np.ndarray,
    tile_shape: tuple[int, int],
    scale: float,
    pad_x: int,
    pad_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.squeeze(prediction)
    if pred.ndim != 2:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    if pred.shape[0] == 5:
        pred = pred.T

    scores = pred[:, 4]
    keep = scores >= CONF_THRESHOLD
    if not np.any(keep):
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    scores = scores[keep].astype(np.float32)
    boxes = xywh_to_xyxy(pred[keep, :4]).astype(np.float32)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

    height, width = tile_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height - 1)

    wh = boxes[:, 2:4] - boxes[:, 0:2]
    valid = (wh[:, 0] > 1) & (wh[:, 1] > 1)
    if not np.any(valid):
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes = boxes[valid]
    scores = scores[valid]
    keep_indices = nms_indices(boxes, scores, TILE_IOU_THRESHOLD, "iou", MAX_DETECTIONS_PER_TILE)
    return boxes[keep_indices], scores[keep_indices]


def filter_by_dish_circle(
    boxes: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if boxes.size == 0:
        return boxes, scores

    height, width = image_shape
    center_x = width * 0.5
    center_y = height * 0.5
    radius = min(width, height) * 0.5 * DISH_RADIUS_SCALE
    box_center_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
    box_center_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
    keep = (box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2 <= radius**2
    return boxes[keep], scores[keep]


def detect(session: ort.InferenceSession, image: np.ndarray) -> list[Detection]:
    input_name = session.get_inputs()[0].name
    image_height, image_width = image.shape[:2]
    collected_boxes = []
    collected_scores = []

    for tile, offset_x, offset_y in iter_tiles(image):
        tensor, scale, pad_x, pad_y = preprocess(tile)
        prediction = session.run(None, {input_name: tensor})[0]
        boxes, scores = postprocess_tile(prediction, tile.shape[:2], scale, pad_x, pad_y)
        if boxes.size == 0:
            continue

        boxes[:, [0, 2]] += offset_x
        boxes[:, [1, 3]] += offset_y
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_width - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_height - 1)
        collected_boxes.append(boxes)
        collected_scores.append(scores)

    if not collected_boxes:
        return []

    boxes = np.concatenate(collected_boxes, axis=0)
    scores = np.concatenate(collected_scores, axis=0)
    boxes, scores = filter_by_dish_circle(boxes, scores, image.shape[:2])
    if boxes.size == 0:
        return []

    keep = nms_indices(
        boxes,
        scores,
        GLOBAL_MERGE_THRESHOLD,
        "containment",
        MAX_DETECTIONS,
    )
    return [
        ((int(boxes[idx, 0]), int(boxes[idx, 1]), int(boxes[idx, 2]), int(boxes[idx, 3])), float(scores[idx]))
        for idx in keep
    ]


def draw_boxes(image: np.ndarray, boxes: list[Detection]) -> np.ndarray:
    output = image.copy()
    thickness = max(1, round(min(image.shape[:2]) / 900))
    for (x1, y1, x2, y2), _score in boxes:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), thickness)

    label = f"colonies: {len(boxes)}"
    cv2.rectangle(output, (0, 0), (260, 44), (255, 255, 255), -1)
    cv2.putText(output, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return output


def collect_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_images(model_path: Path, input_dir: Path, output_dir: Path) -> None:
    image_output_dir = output_dir / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    session = make_session(model_path)
    rows = []

    for image_path in collect_images(input_dir):
        image = read_image(image_path)
        boxes = detect(session, image)
        save_image(image_output_dir / image_path.name, draw_boxes(image, boxes))
        rows.append({"filename": image_path.name, "number of colonies": len(boxes)})

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "number of colonies"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count bacterial colonies with tiled ONNX inference.")
    parser.add_argument("--input", type=Path, default=Path("data/inference"), help="Folder containing input images.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Folder for CSV and annotated images.")
    parser.add_argument("--model", type=Path, default=Path("app/models/best.onnx"), help="ONNX model path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_images(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
