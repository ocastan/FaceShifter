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

## Installation using Conda
```
conda create -n FaceShifter -c pytorch -c conda-forge 'pytorch>=1.6' torchvision tensorboard opencv cudnn
conda activate FaceShifter
git clone https://github.com/ocastan/FaceShifter
cd FaceShifter
```
Go to https://github.com/TreB1eN/InsightFace_Pytorch and download `model_ir_se50.pth` to `face_modules` directory.
## Prepare data
Get face sources (you can look here https://github.com/mindslab-ai/faceshifter#preparing-data for datasets)
```
cd face_modules
python preprocess_images.py unarchived_source_directory cropped_faces_destination_directory
```
Modify `train_face_sources` in `train_AEI.py` accordingly.
## Train
```
python train_AEI.py
```
You may reduce or increase the `batch_size` in `train_AEI.py` according to your graphic card memory.

Monitor losses and generated images, running from another terminal:
```
tensorboard --logdir runs/
```
