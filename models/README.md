# Models

Model files are local artifacts and are not tracked by git.

Recommended layout:

```text
models/
  pretrained/   # YOLO initialization weights
```

For standalone inference, place the selected ONNX export at:

```text
app/models/best.onnx
```

Before publishing weights, confirm that the training data and model license allow
redistribution.
