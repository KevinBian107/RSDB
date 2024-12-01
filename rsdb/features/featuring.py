# responsible for feature engineering
# feature that we are going to use
## category analysis (check if the location belong to certain category)
## Bin locations ()
## consider yearly and monthly reviews
## Hours (TODO)
import pandas as pd
import numpy as np


def featuring_category(df: pd.DataFrame, featuring_category: list) -> pd.DataFrame:
    """
    creating features out of category
    takes around 1 min and 54 sec

    Args:
        df: input dataframe

    return: dataframe with category list encode into different columns

    """
    assert "category" in df.columns, "column category does not exist in df"

    ## helper functions
    def hot_encode_categories(location_categories: pd.Series) -> np.ndarray:
        """
        helper function for featuring_category that that convert
        location_category into multiple columns
        """
        # creating location category_matrix filled with zeros
        location_category_feature = np.zeros(
            (len(location_categories), len(featuring_category)), dtype=int
        )

        location_categories_str_rep = location_categories.apply(
            lambda categories: " ".join(categories).lower()
        )
        # check with location_categories that specified by user
        for i, feature in enumerate(featuring_category):
            location_category_feature[:, i] = [
                1 if feature in category_str_rep else 0
                for category_str_rep in location_categories_str_rep
            ]

        return location_category_feature

    category_feature_matrix = hot_encode_categories(df["category"])

    # convert features into dataframe
    category_feature_df = pd.DataFrame(
        category_feature_matrix,
        columns=[f"isin_category_{name}" for name in featuring_category],
        index=df.index,
    )

    return pd.concat([df, category_feature_df], axis=1).drop(columns=["category"])


def featuring_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    takes in a dataframe and bin locations
    """
    pass


def featuring_review_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    takes in a dataframe and count the reviews
    """
    pass


def featuring_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    extracting time interval features out of the dataframe
    """
    pass


def featuring_engineering(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    takings in a clean dataframe and export dataframe with avaliable features

    Args:
        cleaned_df: claned reviews dataframe
    """
    feaured_df = clean_df

    featuring_category = ["restaurant", "park", "store"]
    feaured_df = featuring_category(feaured_df, featuring_category)

    feaured_df = featuring_locations(feaured_df)

    feaured_df = featuring_review_counts(feaured_df)

    feaured_df = featuring_hours(feaured_df)

    return feaured_df
