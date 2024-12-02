## Static Attributes
Shove vectorized static attributees to model latent between features, constructing a numeric representation:

1. `Category`
    - Bining first (if we do all combination, data is sparse)
    - Count (i.e. 4 categories and see how many does id satisfied)
    - Number of categories having.
2. `Bined Locations`
    - Use longitude + latitude
3. `Gmap popularity score`
    - Monthly visits
    - Temporal, does not construct data lekage
4. `Hours`
    - Open or not? + total opening times
    - Time interval of when it is open
    - Does it open during weekend 
    - Total time

## Dynamic Attributes
We want to model the interaction across time, building a latent representation (it is a feature) with interactions in `Gmap ID`, `Reviewer ID`, and `Rating`.
