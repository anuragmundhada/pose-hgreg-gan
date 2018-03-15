# GANs for weak supervision of depth

This is adopted from the codebase of [Zhou et al](https://github.com/xingyizhou/pytorch-pose-hg-3d).
Uses a a GAN approach to provide weak supervision in the absence of 3D ground truth.

## Requirements
- cudnn
- [PyTorch](http://pytorch.org/)
- Python with h5py, opencv and [progress](https://anaconda.org/conda-forge/progress)
- Optional: [tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) 