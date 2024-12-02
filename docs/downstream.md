## Expand Businesses & Advertisements
We want to look at how user would behave like in the future, not predicting the past, **we use the classical way of reconmendation**.
- How uninteracted `users` would inteacr with `gmap_id` then reconmand to high rating users.
- Assuming that the `gmap_id` want to expand business to certain location, we are modeling how `users` in this region would interact with this `gmap_id` ().

```python
- given gmap_id + location_range
- for all users in this region:
    - predict rating for each uninteracted users
    - reconmand business to predicted high rating users
```

## Start New Businesses
Given a `location (lattitude, longitute)` -> binning -> look at all business rating predictions in this bin -> predict the best business for this location bin given all user rating in this location (implicit inm the recommander system).
- If score of business is high, interacting with lcoal user (captured in data, think about A1 book_id and user_id), high rating tells about insights of successful business in theis region.

```python
- bining all locations (same as feature engineering)  
- for all users in (location_bin + hours_want_to_operate):
    - query all the needed info (temporal info + gmap popularity) based on user info in such location
        - all user in such location has history of interacting with certain business category
    - predict ratings for all type of business x all user in such location
    - aggregate all ratings grouoby business location Bin
    - ranking
```