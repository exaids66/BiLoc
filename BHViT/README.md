# BHViT Runtime Dependency

This directory keeps the minimal BHViT modules required by BiLoc's
`ImageFeatureExtractor(backbone="bhvit")`. Standalone BHViT training and
dataset code is not included in this release package.

The retained runtime files are under `transformer/`:

- `BHViT.py`
- `utils_quant.py`

Please cite the original BHViT work when using this backbone.
