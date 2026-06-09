# Linux and macOS Compatibility

The inference code uses ONNX Runtime with the CPU execution provider, so CUDA is
not required. The project is intended to run on Windows, Linux, and macOS with
Python `3.10` through `3.13`.

## Setup

```bash
git clone https://github.com/xieyaozong/Colony_counting.git
cd Colony_counting
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

## Inference Paths

Use forward-slash paths on Linux and macOS:

```bash
python app/infer.py --input app/data --output app/outputs --model app/models/best.onnx
```

## Notes

- Place the ONNX model at `app/models/best.onnx`.
- Place input images under `app/data/` or pass a different `--input` folder.
- HEIC image support is handled through `pillow-heif`.
- If OpenCV installation differs by platform, test `python -c "import cv2"` in
  the activated environment before running a large batch.
