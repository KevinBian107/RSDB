# Keys to Keep in Mind 💡

Remanber that `intelligence comes from the data`, you should use a temporal model if your data tells you so, not just by imaginations. We have nice engineering lessons from the Netflix price model, `build models that is particularly designed and shaped particularly your data` (extract intelligence form the data), the temporal user bias is specifically designed as a parametric function to follow the frame of the data.

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
1. location (longitude/latitude) (address?)
2. category (one hot)
3. price (discrete, need one hot)
4. hours (int)
5. MISC (one hot)
6. ...Text/reviews (text mining), definately useful to do, but require long time development

**Dynamic attributes**:
1. Models the interaction across time -> a latent representation (it is a feature)

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