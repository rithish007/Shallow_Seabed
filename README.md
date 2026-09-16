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
- `Tools/` — a standalone (non-Unreal-Python) post-processing pipeline that turns raw
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

A local `Dataset/` folder (raw survey captures + the derived `coral/kelp/rock/sponge` YOLO
dataset this pipeline produces) is not published here — see [Related resources](#related-resources).

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
