# Schematic of EDA

EDA + preprocessing for each model (many EDA stuff is reusable, pair up for EDA first)
1. Data Cleaning:
    - Traditional data science cleaning (systematic and rigorous)
        - Type conversion.
        - Outlier (make it N/A?).
        - Evaluate the outlier impact (use BoxPlot).
        - Column split if needed.
        - Narrow dataset to what we need (filter).
    - Missing value imputation. How do we deal with it? Depending on the case, look at the impact.
        - If missing value less than 5%, drop it directly.
        - If missing value more, do dpendent random imputation.
    - **Contraints on data**
        - Standardization
        - Constraint function (data integrity!!! Do we have functional dependency!!!🤪)
            - `The data is where the key lays in (Justin Eldridge)`
            - Ensure atomic typing
    - Look at data distribution (rating,...), beware of imbalance issues.
    - Prevent dropping features, return as much feature as possible.

2. EDA (Find some intelligence in data)
    - Dataset too big problem (how to deal with large data set) ✅
        - Narrow the data scope: we will do `entertainment` + `food` + `retail` using the `California dataset`
            - Choose a state.
            - Choose a few types of business.
            - Choose a specific time stamp (a few year)
            - Random samples may cause issues: need to ensure this sample represents the population.
    - Find relevant features (correlation study)

3. Feature engineering/preprocess:
    - Static attributes (shove in a Factorized Machine to model latent between features), construct a numeric representation/efficient:
        - location (longitude/latitude) (address?)
        - category (one hot)
        - price (discrete, need one hot)
        - hours (int)
        - MISC (one hot)

4. Models
    - Try to avoid python for loop, use vectorization.
    - Two model have characteristics differences, need to analyze
    - Ideally feature must be the same
    - Models: BIM and FPMC
    - Need evaluation metric
