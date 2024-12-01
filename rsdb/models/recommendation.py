import numpy as np
import pandas as pd
import tensorflow as tf
from tlfm.temporal_dynamic import TemporalDynamicModel
from fpmc.fpmc_v import FPMCVariants

class Recommendation():
    '''
    Recommend potential customers for business owner
    '''
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def recommend(self, gmap_id, location):
        '''
        Given gmap_id, recommend potential customers that are 
        more likely to give high rating to the business
        '''
        df = self.prepare_data(gmap_id, location)

        tf_data = None
        if isinstance(self.model, TemporalDynamicModel):
            tf_data = Recommendation.df_to_tf_dynamic_model(df).batch(1024)
        elif isinstance(self.model, FPMCVariants):
            tf_data = self.df_to_tf_fpmc_model(df).batch(1024)

        # predict ratings for all user for a specific business in a location
        pred_rating = []
        for batch in tf_data:
            pred_rating.extend(self.model(batch))

        df['pred_rating'] = pred_rating

        # rank predicted rating  
        # These are the top users that are more likely to give high rating to the business
        top_users = df.sort_values(by='pred_rating', ascending=False)[['reviewer_id', 'pred_rating']].iloc[:20]
        
        return top_users
    
    def prepare_data(self, gmap_id, location):
        '''
        Prepare data for tensorflow model
        '''
        # TODO: query data for users
        data = self.dataset[self.dataset['...'] == location]
        user_ids = np.unique(data['reviewer_id'])
        gmap_ids = np.repeat(gmap_id, len(user_ids))

        # create user-item dataset 
        empty_df = pd.DataFrame({'reviewer_id': user_ids, 'gmap_id': gmap_ids})

        # find the time of latests review for each user 
        user_time_mapping = (self.dataset[self.dataset['reviewer_id'].isin(user_ids)]
                            .groupby('reviewer_id')['review_time(unix)', 'time_bin', 'user_mean_time']
                            .max())

        result_df = empty_df.merge(
            user_time_mapping, 
            on=['reviewer_id']
        )

        # standardlize data
        time_mean, time_std = result_df["review_time(unix)"].mean(), result_df["review_time(unix)"].std()
        user_mean_time_mean, user_mean_time_std = result_df["user_mean_time"].mean(), result_df["user_mean_time"].std()

        result_df["review_time(unix)"] = (result_df["review_time(unix)"] - time_mean) / time_std
        result_df["user_mean_time"] = (result_df["user_mean_time"] - user_mean_time_mean) / user_mean_time_std

        # TODO: query feature for business
        features = data[data['gmap_id'] == gmap_id].iloc[0][...]

        # append feature to the dataframe
        # all the entries will have the same values
        for feature in features.index:
           values = np.repeat(features[feature], len(result_df))
           result_df[feature] = values
        
        # rating does not exist for testing data, however, it is used in model input
        # just give some random invalid values
        result_df['rating'] = np.repeat(999, len(result_df))

        return result_df
    
    def df_to_tf_fpmc_model(self, dataframe):
        '''
        Convert panda Dataframe to tensforflow data for tf fpmc model 
        '''
        # Instantiate StringLookup layers
        user_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.dataset["reviewer_id"].unique(), mask_token=None
        )
        item_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.dataset["gmap_id"].unique(), mask_token=None
        )
        return tf.data.Dataset.from_tensor_slices({
            "reviewer_id": user_lookup(dataframe["reviewer_id"]),
            "prev_item_id": item_lookup(dataframe["prev_item_id"]),
            "next_item_id": item_lookup(dataframe["gmap_id"]),
            "rating": dataframe["rating"].astype(float),
        })
    
    '''Assume we are using this to process tf data for dynamic model'''
    @staticmethod
    def df_to_tf_dynamic_model(dataframe):
        return tf.data.Dataset.from_tensor_slices({
            "reviewer_id": dataframe["reviewer_id"].astype(str),
            "gmap_id": dataframe["gmap_id"].astype(str),
            "time": dataframe["review_time(unix)"].astype(float),
            "time_bin": dataframe["time_bin"].astype(float),
            "user_mean_time": dataframe["user_mean_time"],
            "rating": dataframe["rating"]
        })



'''

def user_call(gmap_id, dataset):
    User call to recommend business category based on location and hours
    
    # for this dataset, do feature engineering

    # query all the needed info (temporal info + gmap popularity for temporal dynamic model)
    dataset = dataset[dataset['...'] == location & dataset['...'] == hours]
    
    # call recommend function

    
    return ...

'''
