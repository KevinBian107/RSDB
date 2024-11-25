# Recommendation based on Sequential Dynamics for Business owners (RSDB)

[Running Meeting Note](https://docs.google.com/document/d/1wip-kDJHyLVldHFIrES-p2NLOI2Qk7_ww8qfhiIvoc4/edit?usp=sharing)

## Reference Sources
1. Dataset:
    - General Information: https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_local
    - Full Dataset: https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/#files
2. Models:
    - General models from textbook: https://cseweb.ucsd.edu/~jmcauley/pml/pml_book.pdf
    - Translational Model: https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18a.pdf
    - Advance LSTM Model: https://github.com/nijianmo/fit-rec
    - Model charcteristics
        - BIM (Bias-injected model) manually define long-short-term bias
        - FPMC (Factorized Personalized Markov Chain) finds automaticlaly short temporal pattern, but fails in long term temporal pattern.
        - RNN/LSTM (Recurrent) cares long term and short term and finds them automatically.
3. Instructions:
    - A2 Slides: https://cseweb.ucsd.edu/classes/fa24/cse258-b/slides/assignment2_fa24.pdf

## Mathamatics Formulation
1. Summary of the math from FitRec: https://github.com/KevinBian107/TBR/blob/main/math/TBR%20Mathamatical%20Formulation.pdf
2. Intro to Sequential Modeling: https://github.com/KevinBian107/SBRB/blob/main/math/Intro%20to%20Sequential%20Modeling.pdf

## Explorative Data Analysis
Refer to https://github.com/KevinBian107/RSDB/blob/main/explorative%20data%20analysis/eda.ipynb
