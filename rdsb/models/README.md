# Keys to Keep in Mind 💡
Remanber that `intelligence comes from the data`, you should use a temporal model if your data tells you so, not just by imaginations. We have nice engineering lessons from the Netflix price model, `build models that is particularly designed and shaped particularly your data` (extract intelligence form the data), the temporal user bias is specifically designed as a parametric function to follow the frame of the data.

# Structure of Models
```bash
- fpmc
    - fpmc.py
        - (factorized personalized markov chain)
    - fpmc_variants.py
        - (our variants factorized personalized markov chain)
    - run.ipynb
        - (run script of non-varaiants fpmc)
- tlfm
    - latent_factor.py
        - (baseline latent factor + neural corrolative)
    - temporal_static.py
        - (Netflex price model with static user imbeddings)
    - temporal_dynamics.py
        - (Netflex price model with dynamics user imbeddings)
    - temporal_dynamics_variants.py
        - (Variants Netflex price model with dynamics user imbeddings)
    - run.ipynb
        - (run scripts of non-variants tlfm)
- recommendation.py
    - (down stream task)
- run.ipynb
    - (full model of the variants)
```

# Question In Interest 🤔
We want to do reconmandation for business owner:
- We are doing the inverse thing: see what user like to make/develop our business towards that (先看客户，然后看开什么店好)
- Think from the business perspective.
- We can do one perspective then reverse it.
- If we are modeling interaction, what interaction are we modeling? All of these serve for **dynamic attributes modeling**
    - What is our `user`: gmap_id (lontitude and latitude)
    - What is our `item`: user_id (?)
    - What is `interactions`: Build connection by rating (num_reviews? token_counts_in_review? token_counts_pos_words?)

# Features 🤪
**Static attributes**:
(shove in a Factorized Machine to model latent between features):

1. `Category` (not one-hot, combination give sparse)
    - Count (i.e. 4 categories and see how many does id satisfied)
    - Number of categories having.
2. `Bined Locations`
    - Use longitude + latitude
3. `Gmap popularity score`
    - Monthly visits
    - Temporal, does not construct data lekage
4. `Hours` (Open or not? + total opening times)
    - Time interval of when it is open (opening period)
    - Does it open during weekend 
    - Total time
5. `Text/reviews` (text mining), definately useful to do, but require long time development

**Dynamic attributes**:
1. Models the interaction across time -> a latent representation (it is a feature)
    - `Gmap ID`
    - `Reviewer ID`
    - `Rating`
2. **Need to study how to model temproal better, sticking with current status for now, working on featuers**

# Prediction Models (Let's not do Dark Magic 🪄🧙）
**Predict unknown interaction between business and users.** New business in certain area, we predict the `overall` user metric (i.e. rating) for this new business (scoring function of your business).
- Predict the overall user metric of certain user with the item that does not exist.
- **If score of business is high, interacting with lcoal user (captured in data, think about A1 book_id and user_id), high rating tells about insights of successful business in theis region**.

1. user = `gmap_id`
2. model `interaction` with item = `user_id` through `metric` (counts/rating/...) and `features`.
    - Which business has close relationship for what user.

# Model Architectures 🌉
Based on the pros and cons of the model, the effect would be different and what we model is different.

- TLF-V (Temporal Latent Factor Model  Variants):
    - It manually define long-short-term bias term for user and item by `binning` or `parametric functions`, many `hand-crafted` things that is specific to the data set working with (original model for Netflix dataset).
    - Temporal modeling based on time-stamp.
- FPMC-V (Factorized Personalized Markov Chain Variants):
    - It finds automaticlaly short temporal pattern, but fails in long term temporal pattern.
    - Sequential modeling.
    - Different from traditional FPMC model, we need to not only distibguish perfered and non-perfered, so the vanilla model need to have use MSE instead of BPR lost.
- LSTM (Recurrent Neural Network):
    - It cares long term and short term and finds them automatically agonist of the dataset.

# Downstream Application 🌊
Given a `location (lattitude, longitute)` -> binning -> look at all business rating predictions in this bin -> predict the best business for this location bin given all user rating in this location (implicit inm the recommander system).

```python
bining by locations (same as feature engineering)
for all users in location Bin + Hours want to operate:
    query all the needed info (temporal info + gmap popularity)
    predict ratings for all type of business x all user in such location
    aggregate all ratings grouoby business category
    ranking

```

# Testing 🔧
- Use a different state? based on a region?
- Output should be given region, check with how many ranking degree.
- Use RMSE, ACC, and R^2 for now
- Study the real business success of the predictions