import gzip
from collections import defaultdict
import math
import numpy as np
import string
import random
import string

import tensorflow as tf
import tensorflow_recommenders as tfrs

import warnings

warnings.filterwarnings("ignore")

class TemporalDynamicVariants(tfrs.Model):
    def __init__(self, l2_reg, dense_units, embedding_dim, dataframe, time_bins):
        super().__init__()

        self.l2_reg = l2_reg
        self.dense_units = dense_units
        self.embedding_dim = embedding_dim
        self.time_bins = time_bins

        # Extract unique identifiers for vocabulary
        self.user_vocab = dataframe["reviewer_id"].astype(str).unique()
        self.item_vocab = dataframe["gmap_id"].astype(str).unique()
        self.time_bin_vocab = list(range(time_bins))

        # Initialize StringLookup layers
        self.user_index = tf.keras.layers.StringLookup(
            vocabulary=self.user_vocab, mask_token=None, num_oov_indices=1
        )
        self.item_index = tf.keras.layers.StringLookup(
            vocabulary=self.item_vocab, mask_token=None, num_oov_indices=1
        )
        self.time_bin_index = tf.keras.layers.IntegerLookup(
            vocabulary=self.time_bin_vocab, mask_token=None, num_oov_indices=1
        )

        # Embedding layers for latent embeddings
        self.user_embedding = tf.keras.layers.Embedding(
            input_dim=self.user_index.vocabulary_size(),
            output_dim=embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
        )
        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=self.item_index.vocabulary_size(),
            output_dim=embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
        )

        # Time bin embedding for dynamic user temporal latent
        self.time_bin_embedding = tf.keras.layers.Embedding(
            input_dim=self.time_bin_index.vocabulary_size(),
            output_dim=embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
        )

        # User and item biases
        self.user_bias = tf.keras.layers.Embedding(
            input_dim=self.user_index.vocabulary_size(), output_dim=1
        )
        self.item_bias = tf.keras.layers.Embedding(
            input_dim=self.item_index.vocabulary_size(), output_dim=1
        )

        # Global bias
        self.global_bias = tf.Variable(initial_value=3.5, trainable=True)

        # Dense layers for interactions
        self.dense_layers = tf.keras.Sequential(
            [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(dense_units, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

        # Rating task
        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def call(self, features):
        # Lookup indices
        user_idx = self.user_index(features["reviewer_id"])
        item_idx = self.item_index(features["gmap_id"])
        time_bin_idx = self.time_bin_index(features["time_bin"])

        # Static embeddings
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)

        # Dynamic user temporal embedding (time-bin specific user latent)
        time_bin_emb = self.time_bin_embedding(time_bin_idx)
        dynamic_user_emb = user_emb + time_bin_emb

        # Biases
        user_bias = self.user_bias(user_idx)
        item_bias = self.item_bias(item_idx)

        # Extract additional features
        category_features = tf.stack(
            [
                features["isin_category_restaurant"],
                features["isin_category_park"],
                features["isin_category_store"],
                features["closed_on_weekend"],
                features["weekly_operating_hours"],
            ],
            axis=1,
        )

        longitude_bins = tf.stack(
            [features[f"lon_bin_{i}"] for i in range(20) if f"lon_bin_{i}" in features],
            axis=1,
        )
        latitude_bins = tf.stack(
            [features[f"lat_bin_{i}"] for i in range(20) if f"lat_bin_{i}" in features],
            axis=1,
        )

        # Combine all features for interaction computation, this is all feature vectors ready for neural network
        interaction_inputs = tf.concat(
            [
                dynamic_user_emb,
                item_emb,
                category_features,
                longitude_bins,
                latitude_bins,
            ],
            axis=1,
        )

        # Interaction score
        interaction_score = self.dense_layers(interaction_inputs)

        # Final prediction
        return (
            tf.squeeze(interaction_score)
            + tf.squeeze(item_bias)
            + tf.squeeze(user_bias)
            + self.global_bias
        )

    def compute_loss(self, features, training=False):
        ratings = features["rating"]
        predictions = self(features)
        return self.rating_task(labels=ratings, predictions=predictions)
    
    def get_config(self):
        # Return a dictionary of the model's configuration
        return {
            "l2_reg": self.l2_reg,
            "dense_units": self.dense_units,
            "embedding_dim": self.embedding_dim,
            "time_bins": self.time_bins,
        }

    @classmethod
    def from_config(cls, config):
        # Create an instance of the model from the config
        return cls(**config)
