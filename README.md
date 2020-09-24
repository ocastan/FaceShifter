# FaceShifter
Forked from taotaonice fantastic job on FaceShifter.
Made the following modifications:
- Moved from Visdom to Tensorboard to monitor the various loss contributors
- Moved to Pytorch native AMP implementation (Pytorch>=1.6): less memory consumption, increased batch size,
- Various corrections: generator hinge loss,...
- Added DiffAugment algorithm to increase the discriminator perceived diversity,
- Tested gradient accumulation, replacing Batch norm with GroupNorm: unsuccessful due to poor attributes transfer.

So far, not satisfied with id loss (0.25 on validation dataset)
