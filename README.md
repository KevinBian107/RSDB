# Recommendation based on Sequential Dynamics for Business owners (RSDB)
[Running Meeting Note](https://docs.google.com/document/d/1wip-kDJHyLVldHFIrES-p2NLOI2Qk7_ww8qfhiIvoc4/edit?usp=sharing)

## Milestones
1. Four hours per day work:
2. Milestones:
    - `Dec 2, 2024` Start doing writeup
    - `Dec 1, 2024` Finish project
    - `Nov 30, 2024` Basemodel, can be bad models, ready to fine tune.
    - `Nov 28, 2024` EDA all finish, start doing model(EDA简单粗暴)
        - Prioritize cleaning and standardization
        - Dataset can use Hawaii if no choice
        - After this, everyone focus on modeling
    - `Nov 26, 2024` Model 1 & 2 start (Jason & Kevin), make sure not just mathematically, but practically work.
    - `Nov 25, 2024` Data cleaning finish (at least good dat, doesn’t need to be best data)
        - Have a Python file with clean code that preprocesses and give clean data

## Structure of System

```bash
rsdb/
├── data/
├── configs/
├── features/
│   ├── featuring.py
├── math_formulation/
├── models/
│   ├── fpmc/
│   ├── tldf/
├── preprocess/
│   ├── data_preprocessing.py
├── recommendation.py
├── run.ipynb
├── train.py
```

## Running RSDB System
Running a training job with `tdlf` model:
```bash
python rsdb/train.py --action "train" --model "tdlf"
```

Running a tunning job with `fpmc` model:
```bash
python rsdb/train.py --action "tune" --model "fpmc"
```

Our system supports customized tunning through our yaml configs system, so all hyperparamters of tunning and training job can be tracked in the configs system.

## Reference Sources
1. Dataset:
    - General Information: https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_local
    - Full Dataset: https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/#files
2. Models:
    - General models from textbook: https://cseweb.ucsd.edu/~jmcauley/pml/pml_book.pdf
    - Translational Model: https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18a.pdf
    - Advance LSTM Model: https://github.com/nijianmo/fit-rec

## Mathamatics Formulation of Model
1. Summary from FitRec: https://github.com/KevinBian107/RSDB/blob/main/math/TBR%20Mathamatical%20Formulation.pdf
2. Intro to Sequential Modeling: https://github.com/KevinBian107/RSDB/blob/main/math/Intro%20to%20Sequential%20Modeling.pdf

