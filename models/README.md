# Models

Model files are local artifacts and are not tracked by git.

Trained model weights and ONNX files are not included in this repository due to
file size and redistribution considerations.

Recommended layout:

```text
models/
  pretrained/   # YOLO initialization weights
```

For standalone inference, place the selected ONNX export at:

```text
app/models/best.onnx
```

If a demo weight is published later, GitHub Releases is preferred over committing
large binary files directly to the repository. Before publishing weights, confirm
that the training data and model license allow redistribution.
