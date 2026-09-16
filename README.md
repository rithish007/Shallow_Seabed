# ShallowSeabed

Unreal Engine 5.8 synthetic underwater scene simulator, built as part of an MSc Robotics
dissertation at the University of Sheffield. Descoped in favour of the diffusion-model
generation pipeline in [ImgGen_AI](https://github.com/rithish007/ImgGen_AI); retained here as
the record of that pivot, and as a working reference implementation of a stencil-based,
foliage-instance-aware synthetic annotation pipeline for underwater scenes.

## What's here

- `Content/Maps/` — survey scenes (`Dataset_Survey_01`, `Deep_Blue_Water`, `Swamp_Water`,
  `BP_UnderWater_Main`).
- `Content/Fish/`, `Content/FX/Fish/` — fish meshes and VFX (Arowana, Betta, Mackerel, and
  others — see `Content/Textures/`).
- `Content/Materials/`, `Content/StaticMesh/`, `Content/ReefAssets/`, `Content/LUT_Texture/` —
  reef/seafloor materials, meshes, and colour-grading LUTs.
- `Content/Blurprint/` — Blueprint logic for the underwater scene and camera survey rig.
- `Content/Dataset/` — the stencil-based instance annotation rig: `M_StencilMask`, per-channel
  render targets (`RT_Capture_RGB`/`RT_Capture_Mask`/`RT_Capture_Depth`), and the
  `BP_SurveyPath` camera-survey Blueprint.
- `Content/Python/dataset/` — the in-editor capture pipeline driving that rig: `capture_rig.py`
  (drives the survey and grabs RGB/mask/depth per frame), `pose_sampling.py` (spline-based
  camera pose generation), `tag_stencil_classes.py` (assigns per-actor stencil IDs),
  `label_derivation.py` (turns captured masks into per-instance boxes), plus `config.py` /
  `path_validation.py` / `example_full_run.py`.
- `Tools/` — a standalone (non-Unreal-Python) post-processing pipeline that turns those raw
  per-instance foliage/rock annotations into a clean YOLO dataset:
  1. `merge_labels.py` — greedy IoU merge of oversegmented per-instance boxes (e.g. a "carpet"
     of adjacent coral instances collapsed into one label).
  2. `cluster_candidates.py` — a 3D-proximity clustering preview, aimed at the cases the 2D
     IoU merge can't fix because it only sees per-frame screen-space overlap.
  3. `export_yolo_dataset.py` — exports the merged labels into a standard Ultralytics YOLO
     train/val layout, with a **block-based** (not per-frame-random) split — frames are 500
     evenly-spaced samples along one continuous survey spline (~8m apart, ~12.4m visibility
     range), so a naive random split would leak near-duplicate content across train/val.
  4. `render_debug_overlays.py` — draws the exported YOLO boxes back onto the RGB frames for
     visual sanity-checking. Run with a regular system Python (not Unreal's embedded
     interpreter), needs Pillow.
- `Dataset/Baseline_Set/scripts/` — the downstream YOLO training/eval pipeline run on the
  exported dataset: `02_train_no_aug.py` / `03_train_with_aug.py` (the augmentation
  comparison), `04_evaluate.py`, `07_predict_real_test_images.py` (the sim-to-real check),
  `08_predict_duo_test.py`, plus preflight/smoke-test/plotting scripts and `scripts/utils/`.

The rest of `Dataset/` (raw survey captures, the ~2000-image derived `coral/kelp/rock/sponge`
YOLO dataset itself, training logs, prediction outputs) is not published here on account of
size — see [Related resources](#related-resources) for the published subset (two 500-frame
image sets + the four trained checkpoints) on Hugging Face.

## Related resources

- **Generation pipeline that shipped:** [ImgGen_AI](https://github.com/rithish007/ImgGen_AI)
- **Dataset and model checkpoints:** [huggingface.co/datasets/Rithish007/MScDissertation](https://huggingface.co/datasets/Rithish007/MScDissertation) (CC-BY-NC-4.0)

## License

Code in this repository is released under the [LICENSE](LICENSE) (MIT). Some third-party
Content assets may carry their own Marketplace/Fab licenses; check individual asset sources
before reuse.

## Citation

If you use this code, please cite it:

```bibtex
@misc{ramamoorthysathya2026shallowseabed,
  author       = {Ramamoorthy Sathya, Rithish},
  title        = {{ShallowSeabed: Unreal Engine 5 Synthetic Underwater Scene Simulator}},
  year         = {2026},
  howpublished = {\url{https://github.com/rithish007/Shallow_Seabed}},
  note         = {Code repository. University of Sheffield MSc Robotics Dissertation.}
}
```

See [CITATION.cff](CITATION.cff) for the machine-readable citation record.
