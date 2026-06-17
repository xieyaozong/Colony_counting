# Inference App

This folder contains the standalone ONNX inference entry point.

## Expected Layout

```text
app/
  infer.py
  data/
  models/
  outputs/
```

`data/`, `models/`, and `outputs/` are local working folders and are ignored by Git.

## Run

```powershell
python app\infer.py --input app\data --output app\outputs --model app\models\best.onnx
```

The script writes:

```text
app\outputs\results.csv
app\outputs\images\
```

`results.csv` contains one row per image with the predicted colony count.

## Notes

- The detector runs through ONNX Runtime `CPUExecutionProvider`.
- CUDA is not required for inference.
- Model files are ignored by git; place your own ONNX export under `app/models/`.
- Input and output images are ignored by git to keep the repository lightweight.
