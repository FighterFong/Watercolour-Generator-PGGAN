# PGGAN-tensorflow
The Tensorflow implementation of [PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION](https://arxiv.org/abs/1710.10196).

### The generative process of PG-GAN

<p align="center">
  <img src="/images/samples.jpg">
</p>


## Setup
### Prerequisites
- TensorFlow >= 1.4
- python 2.7

Run following command in the terminal
```shell
conda create -n py27 python=2.7
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirments.txt
```


### Getting Started
- Train on Jellyfish dataset
The path parameters need modify your path prefix + 'Data/Jellyfish/recognizer/true'
	* OPER_FLAG:			Flag of opertion, 0 is for training, 1 for testing.
	* step_by_save_sample:	the step of saving sample
	* step_by_save_weights:	the step of saving weights

_Just for reference only, specific command will show in Iterations section_
```bash
python main.py --path=your celeba data-path 

python main.py --path=/Users/xufeng/Data/CelebA/images 

python main.py --path=/home/xufengchen/Data/CelebA/images 

# jellyfish watercolour
python main.py --path=/Users/xufeng/Data/Jellyfish/recognizer/true 
python main.py --path=/home/xufengchen/Data/Jellyfish/recognizer/true 
python main.py --path=/Users/xufeng/Data/Jellyfish/recognizer/true --OPER_FLAG=0  --batch_size=32 --max_iters=40000 --step_by_save_sample=5000 --step_by_save_weights=10000

# hkust sea view
python main.py --path=/Users/xufeng/Data/HKUST-SEA/selected-hkust-sea-view 
python main.py --path=/home/xufengchen/Data/HKUST-SEA/selected-hkust-sea-view  

# TinyMind Calligraphy
python main.py --path=/Users/xufeng/Data/TinyMind/Calligraphy/train 
python main.py --path=/home/xufengchen/Data/TinyMind/Calligraphy/train/且/  --OPER_NAME=Calligraphy --max_iters=300 --flag=7 --step_by_save_sample=50 --step_by_save_weights 299

```

- Generate image
modify main.py OPER_FLAG=1

```
eg.
python main.py --path=/Users/xufeng/Data/Jellyfish/recognizer/true --OPER_FLAG=1
```

## Iterations
- 2018.4.9
	v1.0.0
	
- 2018.4.15
	v1.0.1

In this version, we need to run the model with following parameters, also update jellyfish dataset and other new datasets:
you can download from 143.89.131.29:/home/ustlab/Fong_Dir/jellyfish/Data/2018-04-15/ folder.
Three datasets: Jellyfish.zip, Landscape.zip, Calligraphy.zip

The path parameters need modify to your path which saved datasets.
	- path                  Datasets saved path
	- OPER_FLAG:			Flag of opertion, 0 is for training, 1 for testing.
	- step_by_save_sample:	the step of saving sample
	- step_by_save_weights:	the step of saving weights

```shell
1. For Jellyfish
a. python main.py --path=/Path/To/Jellyfish --OPER_NAME=Jellyfish-1671-10000-32 --OPER_FLAG=0  --batch_size=32 --max_iters=10000 --step_by_save_sample=5000 --step_by_save_weights=5000

b. python main.py --path=/Path/To/Jellyfish --OPER_NAME=Jellyfish-1671-20000-32 --OPER_FLAG=0  --batch_size=32 --max_iters=20000 --step_by_save_sample=10000 --step_by_save_weights=10000

2. For Landscape
a. Modify utils.py, comment line 100 and uncomment line 101

b. python main.py --path=/Path/To/Landscape --OPER_NAME=Landscape-1426-10000-32 --OPER_FLAG=0  --batch_size=32 --max_iters=10000 --step_by_save_sample=5000 --step_by_save_weights=5000

3. For Calligraphy
a. Modify utils.py, comment line 101 and uncomment line 100

b. python main.py --path=/Path/To/Calligraphy/train/且/ --OPER_NAME=Calligraphy-Qie-10000-16 --OPER_FLAG=0  --batch_size=16 --max_iters=10000 --step_by_save_sample=5000 --step_by_save_weights=5000

c. python main.py --path=/Path/To/Calligraphy/train/世/ --OPER_NAME=Calligraphy-Shi-10000-32 --OPER_FLAG=0  --batch_size=32 --max_iters=10000 --step_by_save_sample=5000 --step_by_save_weights=5000

```

_**Note:Please refer Prerequisites section to setup your environment before you run the script.If the script running quickly and no images generated,pls check your environment if python2.7**_


## Back
Need back output folder and Logs folder to us
zip -r output.zip output/
zip -r Logs.zip Logs/
    

## Reference code
[From](https://github.com/zhangqianhui/progressive_growing_of_gans_tensorflow)

[PGGAN Theano](https://github.com/tkarras/progressive_growing_of_gans)

[PGGAN Pytorch](https://github.com/github-pengge/PyTorch-progressive_growing_of_gans)
