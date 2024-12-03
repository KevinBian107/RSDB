Create a environment to work on:
```python
conda env create
```

Running a training job with `blf` model(basic latent factor model):
```python
python rsdb/train.py --action "train" --model "blf"
```

Running a training job with `tdlf` model:
```python
python rsdb/train.py --action "train" --model "tdlf"
```

Running a tunning job with `fpmc` model:
```python
python rsdb/train.py --action "tune" --model "fpmc"
```

Our system supports customized tunning through our yaml configs system, so all hyperparamters of tunning and training job can be tracked in the configs system. With the [config](https://github.com/KevinBian107/RSDB/tree/main/rsdb/configs) system, we can tune and choose the hyperparameter that we want to use.

We have created two notebooks for a clear visualization of our [training](https://github.com/KevinBian107/RSDB/blob/main/demo_notebooks/train.ipynb), [evaluations, and downstream applications](https://github.com/KevinBian107/RSDB/blob/main/demo_notebooks/eval.ipynb)