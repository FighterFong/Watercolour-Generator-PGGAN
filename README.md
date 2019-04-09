# PGGAN-tensorflow
The Tensorflow implementation of [PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION](https://arxiv.org/abs/1710.10196).

### The generative process of PG-GAN

<p align="center">
  <img src="/images/samples.png">
</p>


## Setup

### Prerequisites

- TensorFlow >= 1.4
- python 2.7 or 3

### Getting Started

- Train the model on Jellyfish dataset

```bash
python main.py --path=your celeba data-path --celeba=True

python main.py --path=/Users/xufeng/Data/CelebA/images --celeba=True

python main.py --path=/home/xufengchen/Data/CelebA/images --celeba=True

python main.py --path=/Users/xufeng/Data/Jellyfish/recognizer/true --celeba=True

python main.py --path=/home/xufengchen/Data/Jellyfish/recognizer/true --celeba=True

python main.py --path=/Users/xufeng/Data/Jellyfish/recognizer/true --celeba=True --batch_size=32 --max_iters=40000 --step_by_save_sample=5000 --step_by_save_weights=10000

```

- Train the model on CelebA-HQ dataset

```bash
python main.py --path=your celeba-hq data-path --celeba=False
```

## Results on celebA dataset
Here is the generated 64x64 results(Left: generated; Right: Real):

<p align="center">
  <img src="/images/sample.png">
</p>

Here is the generated 128x128 results(Left: generated; Right: Real):
<p align="center">
  <img src="/images/sample_128.png">
</p>


## Results on CelebA-HQ dataset
Here is the generated 64x64 results(Left: Real; Right: Generated):

<p align="center">
  <img src="/images/hs_sample_64.jpg">
</p>

Here is the generated 128x128 results(Left: Real; Right: Generated):
<p align="center">
  <img src="/images/hs_sample_128.jpg">
</p>

## Issue
 If you find some bugs, Thanks for your issue to propose it.
    
## Reference code
[From](https://github.com/zhangqianhui/progressive_growing_of_gans_tensorflow)

[PGGAN Theano](https://github.com/tkarras/progressive_growing_of_gans)

[PGGAN Pytorch](https://github.com/github-pengge/PyTorch-progressive_growing_of_gans)
