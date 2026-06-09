# Failure Cases

This file lists expected failure modes for a bounding-box colony-counting
pipeline. These cases are useful for error analysis, interview discussion, and
future dataset expansion.

## Dense Colonies

Very dense regions can cause under-counting because nearby colonies may appear
as one visual cluster. Tiled inference helps preserve local detail, but it cannot
fully separate colonies that are visually merged.

## Overlapping Colonies

Bounding boxes are less expressive when colonies strongly overlap. Segmentation
or instance segmentation may be a better fit if pixel-level masks are available.

## Edge of Dish

Colonies near the Petri dish edge can be difficult because dish boundaries,
refraction, labels, or cropping artifacts may look colony-like. The circular
dish filter reduces false positives outside the plate, but it assumes the dish
is approximately centered.

## Lighting and Reflection

Uneven illumination, reflections, and glare can create bright or dark regions
that change colony contrast. These cases should be represented in validation
data before deploying to a new microscope or camera setup.

## Small Colonies

Very small colonies may be missed after image resizing or compression. Tiled
inference improves local resolution, but detector confidence thresholds still
need to be tuned against count-level metrics.

## False Positives Outside the Petri Dish

Background dust, labels, bubbles, or table texture can create false detections.
Dish-region filtering is a simple guardrail, but it should be revalidated when
plate placement or framing changes.

## Suggested Review Set

When validating a new dataset, include examples with dense growth, sparse
growth, strong overlap, edge colonies, glare, off-center plates, and small early
colonies. The review set should include both annotated images and count-level
CSV results.
