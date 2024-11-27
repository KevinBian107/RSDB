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


class TemporalStaticModel(tfrs.Model):
    def __init__(self, l2_reg, dense_units, embedding_dim, data_query, time_bins):
        super().__init__()

        self.l2_reg = l2_reg
        self.dense_units = dense_units
        self.embedding_dim = embedding_dim
        self.time_bins = time_bins

        # Initialize StringLookup layers
        self.user_index = tf.keras.layers.StringLookup(
            vocabulary=data_query["reviewer_id"].unique(),
            mask_token=None,
            num_oov_indices=1
        )
        self.item_index = tf.keras.layers.StringLookup(
            vocabulary=data_query["gmap_id"].unique(),
            mask_token=None,
            num_oov_indices=1
        )
        self.time_bin_index = tf.keras.layers.IntegerLookup(
            vocabulary=list(range(time_bins)),
            mask_token=None,
            num_oov_indices=1
        )

        # Embedding layers for latent embeddings
        self.user_embedding = tf.keras.layers.Embedding(
            input_dim=self.user_index.vocabulary_size(),
            output_dim=embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=self.item_index.vocabulary_size(),
            output_dim=embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.time_bin_bias = tf.keras.layers.Embedding(
            input_dim=self.time_bin_index.vocabulary_size(),
            output_dim=1
        )

        # User and item biases
        self.user_bias = tf.keras.layers.Embedding(
            input_dim=self.user_index.vocabulary_size(),
            output_dim=1
        )
        self.item_bias = tf.keras.layers.Embedding(
            input_dim=self.item_index.vocabulary_size(),
            output_dim=1
        )

        # Dynamic user deviation parameter
        self.user_alpha = tf.keras.layers.Embedding(
            input_dim=self.user_index.vocabulary_size(),
            output_dim=1
        )

        # Global bias
        self.global_bias = tf.Variable(initial_value=3.5, trainable=True)

        # Dense layers for interactions
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(dense_units, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        # Rating task
        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features):
        # Lookup indices
        user_idx = self.user_index(features["reviewer_id"])
        item_idx = self.item_index(features["gmap_id"])
        time_bin_idx = self.time_bin_index(features["time_bin"])

        # Embeddings
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)

        # Biases
        user_bias = self.user_bias(user_idx)
        item_bias = self.item_bias(item_idx)
        time_bias = self.time_bin_bias(time_bin_idx)

        # Temporal deviation
        user_alpha = self.user_alpha(user_idx)
        time = tf.cast(features["time"], tf.float32)
        user_mean_time = tf.cast(features["user_mean_time"], tf.float32)
        deviation = tf.math.sign(time - user_mean_time) * tf.abs(time - user_mean_time) ** 0.5
        temporal_effect = user_bias + user_alpha * tf.expand_dims(deviation, axis=-1)

        # Interaction score
        interaction_score = self.dense_layers(tf.concat([user_emb, item_emb], axis=1))

        # Final prediction
        return (
            tf.squeeze(interaction_score)
            + tf.squeeze(item_bias)
            + tf.squeeze(time_bias)
            + tf.squeeze(temporal_effect)
            + self.global_bias
        )

    def compute_loss(self, features, training=False):
        ratings = features["rating"]
        predictions = self(features)
        return self.rating_task(labels=ratings, predictions=predictions)
