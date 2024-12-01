# responsible for feature engineering
# feature that we are going to use
## category analysis (check if the location belong to certain category)
## Bin locations ()
## consider yearly and monthly reviews
## Hours (TODO)
import pandas as pd
import numpy as np
from datetime import datetime

OUTPUT_COLS = [
    "review_time(unix)",
    "reviewer_id",
    "gmap_id",
    "rating",
    "isin_category_restaurant",
    "isin_category_park",
    "isin_category_store",
    "lon_bin_0",
    "lon_bin_1",
    "lon_bin_2",
    "lon_bin_3",
    "lon_bin_4",
    "lon_bin_5",
    "lon_bin_6",
    "lon_bin_7",
    "lon_bin_8",
    "lon_bin_9",
    "lon_bin_10",
    "lon_bin_11",
    "lon_bin_12",
    "lon_bin_13",
    "lon_bin_14",
    "lon_bin_15",
    "lon_bin_16",
    "lon_bin_17",
    "lon_bin_18",
    "lon_bin_19",
    "lat_bin_0",
    "lat_bin_1",
    "lat_bin_2",
    "lat_bin_3",
    "lat_bin_4",
    "lat_bin_5",
    "lat_bin_6",
    "lat_bin_7",
    "lat_bin_8",
    "lat_bin_9",
    "lat_bin_10",
    "lat_bin_11",
    "lat_bin_12",
    "lat_bin_13",
    "lat_bin_14",
    "lat_bin_15",
    "lat_bin_16",
    "lat_bin_17",
    "lat_bin_18",
    "lat_bin_19",
    "closed_on_weekend",
    "weekly_operating_hours",
    "time_bin",
    "user_mean_time",
    "prev_item_id",
]


def is_closed_on_weekend(entry):
    # can swtich to single day if possible

    if not isinstance(entry, list):
        return np.nan
    return entry[2][1] == "Closed" and entry[3][1] == "Closed"


def parse_time(time_str):
    # Define the possible time formats
    time_formats = [
        "%I%p",  # 8AM, 8:00AM
        "%I:%M%p",  # 3:00, 6PM
        "%I:%M",  # 8:00AM
        "%H",  # 3
        "%I%p",  # 3PM, 6PM
    ]
    for fmt in time_formats:
        try:
            # Try parsing the time with the given format
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue

    # If no format matches, return None or raise an error
    raise ValueError(f"Time format for '{time_str}' not recognized")


def calculate_total_hours(hours_list):  # for each row
    total_hours = 0
    if not isinstance(hours_list, list):
        return np.nan

    for week_day in hours_list:
        # Take the second entry
        daily_hour = week_day[1]
        daily_hour = daily_hour.replace("–", "-")
        if daily_hour == "Closed":
            continue  # Skip if the value is "Closed"

        if daily_hour == "Open 24 hours":
            total_hours += 24
            continue

        if not isinstance(daily_hour, str):
            print(f"Unexpected format in daily_hour: {daily_hour}")
            continue  # Skip if the format is unexpected

        try:
            # Split the string into start and end times
            if "-" not in daily_hour:
                print(f"Invalid time format: {daily_hour}")
                continue

            start_time_str, end_time_str = daily_hour.split("-")

            # Parse the start and end times
            start_time = parse_time(start_time_str)
            end_time = parse_time(end_time_str)

            # Calculate duration in hours
            duration = (end_time - start_time).seconds / 3600
            total_hours += duration
        except ValueError as e:
            print(f"Error parsing time: {daily_hour}, Error: {e}")
            continue  # Skip malformed entries

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
    """
    extracting time interval features and open in weekend features

    Args:
        df: clean reviews dataframe

    return: cleaned dataframe
    """
    assert "hours" in df.columns, "hours does not exist"

    df = df.assign(closed_on_weekend=df["hours"].apply(is_closed_on_weekend))
    df = df.assign(weekly_operating_hours=df["hours"].apply(calculate_total_hours))

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

    df = df.assign(time_bin=df["review_time(unix)"])
    user_avg_time = df.groupby("reviewer_id")["review_time(unix)"].mean()
    df = df.assign(user_mean_time=df["reviewer_id"].map(user_avg_time))

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
    checking_category = ["restaurant", "park", "store"]

    featured_df = clean_df

    featured_df = featuring_category(featured_df, checking_category)

    featured_df = featuring_locations(featured_df)

    # feaured_df = featuring_review_counts(feaured_df)

    featured_df = featuring_hours(featured_df)

    featured_df = featuring_model(featured_df)

    # this will drop 7 percent of the dataset
    return featured_df[OUTPUT_COLS].dropna()
