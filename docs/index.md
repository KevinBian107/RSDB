
# A Sequential Recommendation System

**This is a Beta version system. We are currently working on imrpoving model abilities and data flows.**

We are constructing a hybrid recommender system with collaborative filters and content-based filtering. The model will use features of locations, the interaction of location and user, and the temporal evolution of the interaction to predict a user's rating of a business.

## System Flow of RSDB
This is the main flow of our pipeline:

![system](img/SystemWorkFlow.png)

## Reference Sources
1. Dataset:
    - [General Information](https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_local)
    - [Full Dataset](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/#files)
2. Models:
    - [General models from textbook](https://cseweb.ucsd.edu/~jmcauley/pml/pml_book.pdf)
    - [Translational Model](https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18a.pdf)
    - [Advance LSTM Model](https://github.com/nijianmo/fit-rec)