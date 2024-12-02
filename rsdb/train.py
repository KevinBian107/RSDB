# this file is still in process of construction

from preprocess.data_preprocessing import *
from features.featuring import *
from models.tlfm.temporal_dynamic_v import TemporalDynamicVariants
from models.fpmc.fpmc_v import FPMCVariants
import tensorflow as tf

from pathlib import Path
import pandas as pd
import json

def tlfm_df_to_tf_dataset(dataframe):
    '''change featuers from data frame to tensorfloe styles'''
    
    return tf.data.Dataset.from_tensor_slices({
        "reviewer_id": dataframe["reviewer_id"].astype(str),
        "gmap_id": dataframe["gmap_id"].astype(str),
        "time": dataframe["review_time(unix)"].astype(float),
        "time_bin": dataframe["time_bin"].astype(float),
        "user_mean_time": dataframe["user_mean_time"],
        "rating": dataframe["rating"],
        "isin_category_restaurant": dataframe["isin_category_restaurant"].astype(float),
        "isin_category_park": dataframe["isin_category_park"].astype(float),
        "isin_category_store": dataframe["isin_category_store"].astype(float),
        "closed_on_weekend": dataframe["closed_on_weekend"].astype(float),
        "weekly_operating_hours": dataframe["weekly_operating_hours"].astype(float),
        
        # Longitude bins
        **{f"lon_bin_{i}": dataframe[f"lon_bin_{i}"].astype(float) for i in range(20) if f"lon_bin_{i}" in dataframe.columns},
        # Latitude bins
        **{f"lat_bin_{i}": dataframe[f"lat_bin_{i}"].astype(float) for i in range(20) if f"lat_bin_{i}" in dataframe.columns},
    })

def fpmc_df_to_tf_dataset(dataframe):
    '''change featuers from data frame to tensorfloe styles'''
    dataframe["reviewer_id"] = dataframe["reviewer_id"].astype(str)
    dataframe["prev_item_id"] = dataframe["prev_item_id"].astype(str)
    dataframe["gmap_id"] = dataframe["gmap_id"].astype(str)
    
    user_lookup = tf.keras.layers.StringLookup(
        vocabulary=dataframe["reviewer_id"].unique(), mask_token=None
    )
    item_lookup = tf.keras.layers.StringLookup(
        vocabulary=dataframe["gmap_id"].unique(), mask_token=None
    )
    
    return tf.data.Dataset.from_tensor_slices({
        "reviewer_id": user_lookup(dataframe["reviewer_id"]),
        "prev_item_id": item_lookup(dataframe["prev_item_id"]),
        "next_item_id": item_lookup(dataframe["gmap_id"]),
        "rating": dataframe["rating"].astype(float),
        "isin_category_restaurant": dataframe["isin_category_restaurant"].astype(float),
        "isin_category_park": dataframe["isin_category_park"].astype(float),
        "isin_category_store": dataframe["isin_category_store"].astype(float),
        "closed_on_weekend": dataframe["closed_on_weekend"].astype(float),
        "weekly_operating_hours": dataframe["weekly_operating_hours"].astype(float),
        
        # Longitude bins
        **{f"lon_bin_{i}": dataframe[f"lon_bin_{i}"].astype(float) for i in range(20) if f"lon_bin_{i}" in dataframe.columns},
        # Latitude bins
        **{f"lat_bin_{i}": dataframe[f"lat_bin_{i}"].astype(float) for i in range(20) if f"lat_bin_{i}" in dataframe.columns},
    })

def main(model_name):
    # input data
    url = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/review-California_10.json.gz"
    meta_url = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/meta-California.json.gz"

    cleaned_df = get_clean_review_data(url, meta_url)
    featured_df = featuring_engineering(cleaned_df)
    
    data_query = featured_df[['gmap_id', 'reviewer_id', 'rating']]
    train_df = featured_df.sample(frac=0.8, random_state=42)
    test_df = featured_df.drop(train_df.index)
    
    if model_name == "tlfm":
        train_data = tlfm_df_to_tf_dataset(train_df).shuffle(1024).batch(4096)
        test_data = tlfm_df_to_tf_dataset(test_df).batch(4096)
        embedding_dim = 30
        dense_units = 30
        l2_reg = 0.0201
        time_bins= 30
        model = TemporalDynamicVariants(l2_reg, dense_units, embedding_dim, data_query, time_bins)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_root_mean_squared_error", 
            patience=10,
            min_delta=0.001,
            restore_best_weights=True
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2, 
            decay_steps=1000, 
            decay_rate=0.8
        )

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))
        history = model.fit(train_data, epochs=500, validation_data=test_data, callbacks=[early_stopping])
    
    if model_name == "fpmc":
        train_data = fpmc_df_to_tf_dataset(train_df).shuffle(1024).batch(4096)
        test_data = fpmc_df_to_tf_dataset(test_df).batch(4096)
        embedding_dim = 32
        l2_reg = 0.0201
        lr = 1e-3
        model = FPMCVariants(l2_reg=l2_reg, embedding_dim=embedding_dim, data_query=data_query)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_root_mean_squared_error", 
            patience=10,
            min_delta=0.001,
            restore_best_weights=True
        )

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        history = model.fit(
            train_data, 
            validation_data=test_data, 
            epochs=500, 
            callbacks=[early_stopping]
            )
    
    return history

if __name__ == "__main__":
    main('fpmc')
