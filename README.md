# Temporal Bias-based Recommandation

[Running Meeting Note](https://docs.google.com/document/d/1wip-kDJHyLVldHFIrES-p2NLOI2Qk7_ww8qfhiIvoc4/edit?usp=sharing)


Running Sample:
- `data_split.py` - First run this to split the dataset into train/valid/test. Or you can directly download the files here `endomondoHR_proper_temporal_dataset.pkl` and `endomondoHR_proper_metaData.pkl`.
- `heart_rate_aux.py` - Run this file to predict the heart rate given the route and target time.
- `speed_aux.py` - Run this file to predict the speed given the route and target time.
- `data_interpreter_Keras_aux.py` - This is the dataloader file.

Reference Source:
1. Data: https://github.com/MengtingWan/marketBias
2. Data: https://cseweb.ucsd.edu/~jmcauley/datasets.html#market_bias
3. Model: https://github.com/nijianmo/fit-rec
4. Summary of the math from FitRec: https://github.com/KevinBian107/TBR/blob/main/related%20works%20and%20math/TBR%20Mathamatical%20Formulation.pdf
5. A2 Slides: https://cseweb.ucsd.edu/classes/fa24/cse258-b/slides/assignment2_fa24.pdf
