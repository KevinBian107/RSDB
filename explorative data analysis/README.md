# Schematic of EDA

EDA + preprocessing for each model (many EDA stuff is reusable, pair up for EDA first)
1. Data Cleaning:
    - Traditional data science cleaning (systematic rigorous) + outlier (make it N/A?) + evaluate the outlier impact (use BoxPlot).
    - Missing value imputation. How do we deal with it? Depending on the case, look at the impact.
    - **Standardization + constraint function (data integrity!!! Do we have functional dependency!!!🤪) The data is where the key lays in (Justin Eldridge)’**
    - Look at data distribution (rating,...), beware of imbalance issues.
    - Prevent dropping features (atomic typing), return as much feature as possible.

2. EDA (Find some intelligence in data)
    - Dataset too big problem (how to deal with large data set)
        - Narrow the data scope:
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
