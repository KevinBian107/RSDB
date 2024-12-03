Remanber that `intelligence comes from the data`, you should use a temporal model if your data tells you so, not just by imaginations. We have nice engineering lessons from the Netflix price model, `build models that is particularly designed and shaped particularly your data` (extract intelligence form the data), the temporal user bias is specifically designed as a parametric function to follow the frame of the data.

## Structure of Models
```bash
fpmc (factorized personalized markov chain)/
├── fpmc.py
│   └── (factorized personalized markov chain)
├── fpmc_variants.py
│   └── (our variants factorized personalized markov chain)
├── baseline.ipynb
│   └── (run script of non-variants fpmc baseline)
tdlf (temporal dynamic latent factor)/
├── latent_factor.py
│   └── (baseline latent factor + neural correlative)
├── temporal_static.py
│   └── (Netflix price model with static user embeddings)
├── temporal_dynamics.py
│   └── (Netflix price model with dynamic user embeddings)
├── temporal_dynamics_variants.py
│   └── (Variants Netflix price model with dynamic user embeddings)
├── baseline.ipynb
│   └── (run scripts of non-variants tlfm baseline)
```

## Question In Interest
We want to do reconmandation for business owner, doing the inverse from traditional reconmandations for users. We want to see what does the users like, then we want to develop our business towards that (thinking from the business perspective).

- We are modeling interaction by connection through rating (though we can also use `num_reviews`, `token_counts_in_review`, or `token_counts_pos_words`)
- Techniqually, teh `user` for us is business owners and `item` for us is the actual users, but because the structure of the dataset, we still use the traditional perspective of deeming users as `user`.
- We want to predict unknown interaction between business and users, modeling `interaction` through `metric` (counts/rating/...) and `features`, reasoning about **what business has close relationship for what user**.

## Model Architectures
Based on the pros and cons of the model, the effect would be different and what we model is different. All model have characteristics differences, they are good for different cases as they care about different properties.

### Temporal Latent Factor Model Variants (`TLF-V`):
1. It manually define long-short-term bias term for user and item by `binning` or `parametric functions`, many `hand-crafted` things that is specific to the data set working with (original model for Netflix dataset).
2. Temporal modeling based on time-stamp.
3. Very dependent and over-enginnering on a specific domain of dataset.

### Factorized Personalized Markov Chain Variants (`FPMC-V`):
1. It finds automaticlaly short temporal pattern (first order Markovian property), but fails in long term temporal pattern.
2. Sequential modeling, not caring about timestamps but ordering. It is the more trivial form of token prediction.
    - Item should match user preferences.
    - Next item should be consistent with previous item.
    - (i.e. if users watches Harry Potter 1 and really likes it + they have the ability to watch sequel of Harry Potter 1, we can say they might like Harry Potter 2)
3. Knowing what a user did just last time is much more predictive than knowing what the user did for all the history.
4. Different from traditional FPMC model, we need to not only distibguish perfered and non-perfered, so the vanilla model need to have use MSE instead of BPR lost.

### Further Implementations:
1. Transformer/Recurrent Neural Network Based Models (LSTM):
    - Surpass first order Markovian property, it includes more history and recency.
    - Finding these patterns automatically agonist of the dataset.
2. Changing latent inner product (related to SVD, but can be changed) to eucledian distance measure (Metric embeddings for sequential recommendation).

## Testing
- Use a different state? based on a region?
- Output should be given region, check with how many ranking degree.
- Use RMSE, ACC, and R^2 for now
- Study the real business success of the predictions