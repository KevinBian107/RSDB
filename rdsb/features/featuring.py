# responsible for feature engineering
# feature that we are going to use
## category analysis (check if the location belong to certain category)
## Bin locations ()
## consider yearly and monthly reviews
## Hours (TODO)
import pandas as pd
import numpy as np


def featuring_category(df: pd.DataFrame, featuring_category) -> pd.DataFrame:
    """
    creating features out of category
    """
    pass


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
    featuring_category = ["restaurant", "park", "glocery"]

    feaured_df = clean_df

    feaured_df = featuring_category(
        feaured_df,
    )

    feaured_df = featuring_locations(feaured_df)

    feaured_df = featuring_review_counts(feaured_df)

    feaured_df = featuring_hours(feaured_df)

    return feaured_df
