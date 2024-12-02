Running a training job with `tdlf` model:
```bash
python rsdb/train.py --action "train" --model "tdlf"
```

Running a tunning job with `fpmc` model:
```bash
python rsdb/train.py --action "tune" --model "fpmc"
```

Our system supports customized tunning through our yaml configs system, so all hyperparamters of tunning and training job can be tracked in the configs system.