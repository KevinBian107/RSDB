## Data Cleaning
Notice that EDA here uses a toy `hawaii_dataset`, different from the actual dataset we are using. **The data size used is extremely large (fourty million rows of data), dealing with large dataset is one main challenge in this project.**

- Traditional data science cleaning (systematic and rigorous)
    - Type conversion
    - Outlier
    - Evaluate the outlier impact (use BoxPlot).
    - Column split if needed.
    - Narrow dataset to what we need (filter).
- Missing value imputation. How do we deal with it? Depending on the case, look at the impact.
    - If missing value less than 5%, drop it directly.
    - If missing value more, do dpendent random imputation.
- Contraints on data
    - Standardization
    - Constraint function (data integrity!!! Do we have functional dependency!!!🤪)
        - `The data is where the key lays in (Justin Eldridge)`
        - Ensure atomic typing
- Look at data distribution (rating,...), beware of imbalance issues.
- Prevent dropping features, return as much feature as possible.

## Eexplorations of Data
Let's try to find some intelligence in the dataset.

- Dataset too big problem (how to deal with large data set)
- We do not use the subset, subsetting makes data space sparse. We use the k-core for using only the dense data set and thinking about the more active user and items.
    - Narrow the data scope (i.e. `entertainment` + `food` + `retail` using the `California dataset`)
        - Choose a state.
        - Choose a few types of business.
        - Choose a specific time stamp (a few year)
        - Random samples may cause issues: need to ensure this sample represents the population.
- Find relevant features (correlation study)