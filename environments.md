
## Environments
python 2.7
TensorFlow = 1.4

should run the following command in the terminal
conda create -n py27 python=2.7
pip install -r requirments.txt

## Data
In pggan_v1.0.0.zip data folder

## Usage
### Training
the path parameters need modify your path prefix + 'Data/Jellyfish/recognizer/true'
- OPER_FLAG	Flag of opertion: 0 is for training
- step_by_save_sample	the step of save sample
- step_by_save_weights	the step of save weights
```
eg.
python main.py --path=/Users/xufeng/Data/Jellyfish/recognizer/true --celeba=True

python main.py --path=/Users/xufeng/Data/Jellyfish/recognizer/true --OPER_FLAG=0 --celeba=True --batch_size=32 --max_iters=40000 --step_by_save_sample=5000 --step_by_save_weights=10000
```

### Generate
modify main.py OPER_FLAG=1
```
eg.
python main.py --path=/Users/xufeng/Data/Jellyfish/recognizer/true --OPER_FLAG=1 --celeba=True
```

## Back
need back output folder to us
zip -r output.zip output/
