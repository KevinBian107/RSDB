# responsible for feature engineering
# feature that we are going to use
## category analysis (check if the location belong to certain category)
## Bin locations ()
## consider yearly and monthly reviews
## Hours (TODO)
import pandas as pd
import numpy as np


def milliseconds_to_years(milliseconds: int) -> float:
    """
    turn milliseconds into years

    Args:
        milliseconds

    return: number that convert milliseconds into years
    """
    seconds = milliseconds / 1000
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24
    years = days / 365.25  # Account for leap years
    return years


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


def featuring_locations(df: pd.DataFrame, lon_bins=20, lat_bins=20) -> pd.DataFrame:
    """
    takes in a dataframe and divide longitude and latitude into equally
    distributed bins

    Args:
        df: input dataframe
        lon_bins: number of bins for longitude
        lat_bins: number of bins for latitude

    return: dataframe with bins encoded into categories

    """
    assert "longitude" in df.columns, "longitude not in the dataframe"
    assert "latitude" in df.columns, "latitude not in the dataframe"

    # Calculate bin edges for longitude and latitude
    lon_edges = np.linspace(df["longitude"].min(), df["longitude"].max(), lon_bins + 1)
    lat_edges = np.linspace(df["latitude"].min(), df["latitude"].max(), lat_bins + 1)

    lon_bins = pd.cut(
        df["longitude"], bins=lon_edges, labels=False, include_lowest=True
    )
    lat_bins = pd.cut(df["latitude"], bins=lat_edges, labels=False, include_lowest=True)

    lon_feature_df = pd.get_dummies(lon_bins, prefix="lon_bin", dtype=int)
    lat_feature_df = pd.get_dummies(lat_bins, prefix="lat_bin", dtype=int)

    return pd.concat([df, lon_feature_df, lat_feature_df], axis=1).drop(
        columns=["longitude", "latitude"]
    )


def featuring_review_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    takes in a dataframe and count the average review per year of each gmapid

    Args:
        pd: input dataframe

    return: dataframe with bins encoded into categories
    """
    assert "review_time(unix)" in df.columns, "no review time"
    assert "gmap_id" in df.columns, "needs location identifier"
    assert "num_of_reviews" in df.columns, "needs total review counts"

    # find review duration of each store and calculate the avg review per year
    latest_review_time = df["review_time(unix)"].max()
    location_earliest_review = df.groupby(["gmap_id"])["review_time(unix)"].min()
    location_duration_ms = latest_review_time - location_earliest_review
    location_duration_yr = location_duration_ms.apply(milliseconds_to_years)
    location_duration_yr_reviws = df[["gmap_id", "num_of_reviews"]].merge(
        location_duration_yr, left_on="gmap_id", right_index=True
    )

    assert location_duration_yr_reviws.shape[0] == df.shape[0], "merging issue"

    return df.assign(
        **{
            "avg_review(per year)": location_duration_yr_reviws["num_of_reviews"]
            / location_duration_yr_reviws["review_time(unix)"]
        }
    )


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
