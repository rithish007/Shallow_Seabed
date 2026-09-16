"""Shared helper for resolving a data.yaml train/val image path to its
matching labels path.
"""
import os


def images_to_labels_rel_path(images_rel_path):
    """Mirror Ultralytics' own img2label_paths: swap the last 'images' path
    segment for 'labels', so this matches whatever labels dir YOLO itself
    would actually use for a given data.yaml train/val path."""
    parts = images_rel_path.replace("\\", "/").split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return os.path.join(*parts)
    raise ValueError(f"Could not find an 'images' path segment in {images_rel_path!r}")
