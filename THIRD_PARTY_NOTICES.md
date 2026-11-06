# Third-party code, models and data

## Included reference implementation

`src/rtabmap_hloc_reprocess/_exchange/` contains the reconstruction-description
and scene-bundle reference shared with [Restruct](https://github.com/HomeSpatial/restruct).
It is included under the project's Apache-2.0 core grant, with attribution in
[NOTICE](NOTICE). The file hashes in [SOURCE.json](src/rtabmap_hloc_reprocess/_exchange/SOURCE.json)
identify the exact vendored revision; release checks verify these hashes.
The GPL-licensed Blender component of Restruct is not included here.

## Installed separately

The wheel and source archive do not bundle HLOC, COLMAP, PyTorch, model
checkpoints or third-party datasets. Installation resolves external dependencies.
Their own notices and licenses remain applicable.

| Component | Source and terms |
| --- | --- |
| HLOC | [Pinned source license](https://github.com/cvg/Hierarchical-Localization/blob/c13273bd0ecc2917a35910fd843712a1c6243193/LICENSE), Apache-2.0 |
| LightGlue | [Locked source license](https://github.com/cvg/LightGlue/blob/5a650c444718caffc2669dc0094cde908f42dd73/LICENSE), Apache-2.0 |
| COLMAP / pycolmap | [Upstream repository and license](https://github.com/colmap/colmap) |
| PyTorch and torchvision | [PyTorch](https://github.com/pytorch/pytorch), [torchvision](https://github.com/pytorch/vision) |
| Feature and retrieval models | [HLOC extractor sources](https://github.com/cvg/Hierarchical-Localization/tree/c13273bd0ecc2917a35910fd843712a1c6243193/hloc/extractors), [ALIKED](https://github.com/Shiaoming/ALIKED), [MegaLoc](https://github.com/gmberton/MegaLoc) |

This table identifies major dependencies, not a relicensing of their transitive
dependencies. The complete Python dependency set is in `uv.lock`. Alternative
extractors and weights can have different terms; a selectable model name does
not imply that every associated weight is available under Apache-2.0 or for
unrestricted commercial use. Review the exact upstream model and weight terms
before using or redistributing them.

The repository contains synthetic test fixtures generated in temporary
directories. User-supplied databases, photographs and outputs are not covered
by this project's license. Contributors must have permission to share any data
attached to an issue or pull request.
