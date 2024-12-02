# responsible for feature engineering
# feature that we are going to use
## category analysis (check if the location belong to certain category)
## Bin locations ()
## consider yearly and monthly reviews
## Hours (TODO)
import pandas as pd
import numpy as np
from datetime import datetime
import time
from functools import lru_cache

DROP_COLS = [
    "reviewer_name",
    "text",
    "resp",
    "has_rep",
    "gmap_name",
    "description",
    "avg_rating",
    "num_of_reviews",
    "price",
    "hours",
    "MISC",
    "address",
    "relative_results",
]

time_format = ["%I%p", "%I:%M%p", "%I:%M", "%H", "%I%p"]


# caching the funciton so input will be remembered
@lru_cache(None)
def parse_time(time_str):
    for fmt in time_format:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Time format for '{time_str}' not recognized")


def parsing_hourList(hourList: list) -> dict:
    if not isinstance(hourList, list):
        return np.nan
    return {ind_hour[0]: ind_hour[1] for ind_hour in hourList}


def preprocess_time(hour_dict):
    if not isinstance(hour_dict, dict):
        return np.nan

    preprocessed = dict()

    for day, hours in hour_dict.items():
        if hours == "Closed":
            preprocessed[day] = "Closed"
        elif hours == "Open 24 hours":
            preprocessed[day] = (0, 24)
        else:
            try:
                hours = hours.replace("–", "-")
                if "-" not in hours:
                    preprocessed[day] = None  # Invalid format
                    continue

                start_time_str, end_time_str = hours.split("-")
                start_hour = parse_time(start_time_str).hour
                end_hour = parse_time(end_time_str).hour
                preprocessed[day] = (start_hour, end_hour)
            except Exception as e:
                print(f"Error parsing time for {day}: {hours}, Error: {e}")
                preprocessed[day] = None

    return preprocessed


def calculate_total_hours_vectorized(processed_hours):
    if not isinstance(processed_hours, dict):
        return np.nan
    total_hours = 0
    for _, hours in processed_hours.items():
        if hours == "Closed" or not hours:
            continue
        start, end = hours
        total_hours += (end - start) % 24  # Handles overnight cases
    return total_hours


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


# def featuring_review_counts(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     takes in a dataframe and count the average review per year of each gmapid

#     Args:
#         pd: input dataframe

#     return: dataframe with bins encoded into categories
#     """
#     assert "review_time(unix)" in df.columns, "no review time"
#     assert "gmap_id" in df.columns, "needs location identifier"
#     assert "num_of_reviews" in df.columns, "needs total review counts"

#     # find review duration of each store and calculate the avg review per year
#     latest_review_time = df["review_time(unix)"].max()
#     location_earliest_review = df.groupby(["gmap_id"])["review_time(unix)"].min()
#     location_duration_ms = latest_review_time - location_earliest_review
#     location_duration_yr = location_duration_ms.apply(milliseconds_to_years)
#     location_duration_yr_reviws = df[["gmap_id", "num_of_reviews"]].merge(
#         location_duration_yr, left_on="gmap_id", right_index=True
#     )

#     assert location_duration_yr_reviws.shape[0] == df.shape[0], "merging issue"

#     return df.assign(
#         **{
#             "avg_review(per year)": location_duration_yr_reviws["num_of_reviews"]
#             / location_duration_yr_reviws["review_time(unix)"]
#         }
#     )


def featuring_hours(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(hours_dict=df["hours"].apply(parsing_hourList))
    df["closed_on_weekend"] = df["hours_dict"].apply(
        lambda hour_dict: isinstance(hour_dict, dict)
        and hour_dict.get("Saturday") == "Closed"
        and hour_dict.get("Sunday") == "Closed"
    )
    df["operating_hours"] = df["hours_dict"].apply(preprocess_time)
    df["weekly_operating_hours"] = df["operating_hours"].apply(
        calculate_total_hours_vectorized
    )

    return df


def featuring_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    creating ml specific features. These features include:
    * weekly bin for gmap_id
    * average review time that user left the comment (normalized)
    * normalized review time

    Args:
        cleaned_df: claned reviews dataframe

    return: dataframe with
    """

    df = df.assign(time_bin=df["review_time(unix)"] // (7 * 24 * 3600))
    df["user_mean_time"] = df.groupby("reviewer_id")["review_time(unix)"].transform(
        "mean"
    )

    time_mean, time_std = (
        df["review_time(unix)"].mean(),
        df["review_time(unix)"].std(),
    )
    user_mean_time_mean, user_mean_time_std = (
        df["user_mean_time"].mean(),
        df["user_mean_time"].std(),
    )

    df["review_time(unix)"] = (df["review_time(unix)"] - time_mean) / time_std
    df["user_mean_time"] = (
        df["user_mean_time"] - user_mean_time_mean
    ) / user_mean_time_std

    df = df.sort_values(by=["reviewer_id", "review_time(unix)"])
    df["prev_item_id"] = df.groupby("reviewer_id")["gmap_id"].shift(1)
    df = df.dropna(subset=["prev_item_id"])

    return df


def featuring_engineering(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    takings in a clean dataframe and export dataframe with avaliable features

    Args:
        cleaned_df: claned reviews dataframe
    """
    start_time = time.time()
    checking_category = ["restaurant", "park", "store"]
    featured_df = clean_df

    featured_df = featuring_category(featured_df, checking_category)
    print(f"finished finding generalized categories. Takes {time.time() - start_time}")
    start_time = time.time()

    featured_df = featuring_locations(featured_df)
    print(f"finished bining locations. Takes {time.time() - start_time}")
    start_time = time.time()

    featured_df = featuring_hours(featured_df).dropna(
        subset=["operating_hours", "weekly_operating_hours"]
    )
    print(f"finished featuring hours. Takes {time.time() - start_time}")
    start_time = time.time()

    featured_df = featuring_model(
        featured_df,
    )
    print(
        f"finished creating model specalizied feature. Takes {time.time() - start_time}"
    )

    # featured_df = (
    #     clean_df.pipe(featuring_category, featuring_category=checking_category)
    #     .pipe(featuring_locations)
    #     .pipe(featuring_hours)
    #     .pipe(featuring_model)
    # )

    # this will drop 7 percent of the dataset
    return featured_df  # .drop(columns=DROP_COLS)  # [OUTPUT_COLS]
