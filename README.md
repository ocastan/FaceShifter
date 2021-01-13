# FaceShifter
Forked from taotaonice fantastic job on FaceShifter.
Made the following modifications:
- Moved from Visdom to Tensorboard to monitor the various loss contributors
- Moved to Pytorch native AMP implementation (Pytorch>=1.6): less memory consumption, increased batch size,
- Various corrections: generator hinge loss, dataset logic,...
- ~~Added DiffAugment algorithm to increase the discriminator perceived diversity,~~
- Tested gradient accumulation, replacing Batch norm with GroupNorm: unsuccessful due to poor attributes transfer,
- Reduced the adversarial loss and the Generator learning rate to achieve better source id transfer,
- Changed the face cropping algorithm to get a better chin coverage.

## TODO
Rework the dataset generation to keep FFHQ native images and to crop the others images using FFHQ algorithm (using landmarks) to preserve the whole chin and achieve better alignment.

## Requirements installation using Conda
```
conda create -n FaceShifter -c pytorch -c conda-forge 'pytorch>=1.6' torchvision tensorboard opencv cudnn
conda activate FaceShifter
python train_AEI.py
```
You may reduce or increase the `batch_size` in `train_AEI.py` according to your graphic card memory.
