# Data

Local datasets are intentionally not tracked by git.

Recommended layout:

```text
data/
  source/       # raw images and source annotations
  prepared/     # generated YOLO datasets
  archives/     # original compressed datasets, if any
```

The preparation scripts expect source annotations and images to be placed here,
but the exact filenames can be configured through script arguments.
