# Question In Interest🤔
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