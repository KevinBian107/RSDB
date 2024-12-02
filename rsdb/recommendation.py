import numpy as np
import pandas as pd
import tensorflow as tf
from models.tdlf.temporal_dynamic_v import TemporalDynamicVariants
from models.fpmc.fpmc_v import FPMCVariants
from features.featuring import featuring_engineering

class Recommendation():
    """
    Recommend potential customers for business owner
    """

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        # the number of users recommend to business
        self.N = 20
        # feature engineering dataset
        self.featured_df = featuring_engineering(self.dataset)
        

    def recommend(self, gmap_id):
        """
        Given gmap_id, recommend potential customers that are
        more likely to give high rating to the business
        """
        user_df = self.prepare_data(gmap_id)

        tf_data = None
        if isinstance(self.model, TemporalDynamicVariants):
            tf_data = Recommendation.tdlf_df_to_tf(user_df).batch(1024)
        elif isinstance(self.model, FPMCVariants):
            tf_data = Recommendation.fpmc_df_to_tf(user_df).batch(1024)

        # predict ratings for all user for a specific business in a location
        pred_rating = []
        for batch in tf_data:
            pred_rating.extend(self.model(batch).numpy())

        user_df["pred_rating"] = pred_rating

        # rank predicted rating
        # These are the top users that are more likely to give high rating to the business
        top_users = user_df.sort_values(by="pred_rating", ascending=False)[
            ["reviewer_id", "pred_rating"]
        ].iloc[:self.N]

        return top_users

    def prepare_data(self, gmap_id):
        """
        Prepare data for tensorflow model
        """
        # query data for users 
        category = set(self.dataset[self.dataset['gmap_id'] == gmap_id].iloc[0]['category'])

        query_data = (self.dataset[(self.dataset['category'].apply(
            lambda x: any(item in category for item in x))) & 
            (self.dataset['gmap_id'] != gmap_id)
            ])

        user_ids = query_data['reviewer_id']
        # get user ids that's available in featured dataframe
        user_ids = np.unique(self.featured_df[self.featured_df['reviewer_id'].isin(user_ids)]['reviewer_id'])
        gmap_ids = np.repeat(gmap_id, len(user_ids))

        # create user-item dataset
        empty_df = pd.DataFrame({"reviewer_id": user_ids, "gmap_id": gmap_ids})

        # find the time of latests review for each user
        user_time_mapping = (self.featured_df[
            self.featured_df["reviewer_id"].isin(user_ids)
            ].loc[
                lambda df: df.groupby("reviewer_id")["review_time(unix)"].idxmax()
            ][['reviewer_id', 'prev_item_id', "review_time(unix)", "time_bin", "user_mean_time"]]
        )

        result_df = empty_df.merge(
            user_time_mapping, 
            on = ['reviewer_id'], 
            how='left'
        )

        features = self.featured_df[[
        'gmap_id','isin_category_restaurant', 'isin_category_park', 'isin_category_store',
        'lon_bin_0', 'lon_bin_1', 'lon_bin_2', 'lon_bin_3', 'lon_bin_4',
        'lon_bin_5', 'lon_bin_6', 'lon_bin_7', 'lon_bin_8', 'lon_bin_9',
        'lon_bin_10', 'lon_bin_11', 'lon_bin_12', 'lon_bin_13', 'lon_bin_14',
        'lon_bin_15', 'lon_bin_16', 'lon_bin_17', 'lon_bin_18', 'lon_bin_19',
        'lat_bin_0', 'lat_bin_1', 'lat_bin_2', 'lat_bin_3', 'lat_bin_4',
        'lat_bin_5', 'lat_bin_6', 'lat_bin_7', 'lat_bin_8', 'lat_bin_9',
        'lat_bin_10', 'lat_bin_11', 'lat_bin_12', 'lat_bin_13', 'lat_bin_14',
        'lat_bin_15', 'lat_bin_16', 'lat_bin_17', 'lat_bin_18', 'lat_bin_19',
        'closed_on_weekend', 'weekly_operating_hours']].iloc[0]

        # append feature to the dataframe
        # all the entries will have the same values
        for feature in features.index:
            values = np.repeat(features[feature], len(result_df))
            result_df[feature] = values

        # rating does not exist for testing data, however, it is used in model input
        # just give some random invalid values
        result_df["rating"] = np.repeat(999, len(result_df))

        return result_df

    @staticmethod
    def fpmc_df_to_tf(dataframe):
        """change featuers from data frame to tensorfloe styles"""
        dataframe["reviewer_id"] = dataframe["reviewer_id"].astype(str)
        dataframe["prev_item_id"] = dataframe["prev_item_id"].astype(str)
        dataframe["gmap_id"] = dataframe["gmap_id"].astype(str)

        user_lookup = tf.keras.layers.StringLookup(
            vocabulary=dataframe["reviewer_id"].unique(), mask_token=None
        )
        item_lookup = tf.keras.layers.StringLookup(
            vocabulary=dataframe["gmap_id"].unique(), mask_token=None
        )

        return tf.data.Dataset.from_tensor_slices(
            {
                "reviewer_id": user_lookup(dataframe["reviewer_id"]),
                "prev_item_id": item_lookup(dataframe["prev_item_id"]),
                "next_item_id": item_lookup(dataframe["gmap_id"]),
                "rating": dataframe["rating"].astype(float),
                "isin_category_restaurant": dataframe["isin_category_restaurant"].astype(
                    float
                ),
                "isin_category_park": dataframe["isin_category_park"].astype(float),
                "isin_category_store": dataframe["isin_category_store"].astype(float),
                "closed_on_weekend": dataframe["closed_on_weekend"].astype(float),
                "weekly_operating_hours": dataframe["weekly_operating_hours"].astype(float),
                # Longitude bins
                **{
                    f"lon_bin_{i}": dataframe[f"lon_bin_{i}"].astype(float)
                    for i in range(20)
                    if f"lon_bin_{i}" in dataframe.columns
                },
                # Latitude bins
                **{
                    f"lat_bin_{i}": dataframe[f"lat_bin_{i}"].astype(float)
                    for i in range(20)
                    if f"lat_bin_{i}" in dataframe.columns
                },
            }
        )

    @staticmethod
    def tdlf_df_to_tf(dataframe):
        """change featuers from data frame to tensorfloe styles"""
        return tf.data.Dataset.from_tensor_slices(
            {
                "reviewer_id": dataframe["reviewer_id"].astype(str),
                "gmap_id": dataframe["gmap_id"].astype(str),
                "time": dataframe["review_time(unix)"].astype(float),
                "time_bin": dataframe["time_bin"].astype(float),
                "user_mean_time": dataframe["user_mean_time"],
                "rating": dataframe["rating"],
                "isin_category_restaurant": dataframe["isin_category_restaurant"].astype(
                    float
                ),
                "isin_category_park": dataframe["isin_category_park"].astype(float),
                "isin_category_store": dataframe["isin_category_store"].astype(float),
                "closed_on_weekend": dataframe["closed_on_weekend"].astype(float),
                "weekly_operating_hours": dataframe["weekly_operating_hours"].astype(float),
                # Longitude bins
                **{
                    f"lon_bin_{i}": dataframe[f"lon_bin_{i}"].astype(float)
                    for i in range(20)
                    if f"lon_bin_{i}" in dataframe.columns
                },
                # Latitude bins
                **{
                    f"lat_bin_{i}": dataframe[f"lat_bin_{i}"].astype(float)
                    for i in range(20)
                    if f"lat_bin_{i}" in dataframe.columns
                },
            }
        )

