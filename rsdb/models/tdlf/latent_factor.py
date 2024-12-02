import gzip
from collections import defaultdict
import math
import numpy as np
import string
import random
import string

import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import keras_tuner as kt

import warnings

warnings.filterwarnings("ignore")


class LatentFactorModel(tfrs.Model):
    def __init__(self, l2_reg, dense_units, embedding_dim, data_query):
        super().__init__()

        self.l2_reg = l2_reg
        self.dense_units = dense_units
        self.embedding_dim = embedding_dim

        # latent factors using lookup table
        self.b_embed = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=data_query["gmap_id"].unique(),
                    mask_token=None,
                    num_oov_indices=1,
                ),
                tf.keras.layers.Embedding(
                    len(data_query["gmap_id"].unique()) + 2,
                    self.embedding_dim,
                    embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                ),
            ]
        )
        self.r_embed = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=data_query["reviewer_id"].unique(),
                    mask_token=None,
                    num_oov_indices=1,
                ),
                tf.keras.layers.Embedding(
                    len(data_query["reviewer_id"].unique()) + 2,
                    self.embedding_dim,
                    embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                ),
            ]
        )

        # bias terms
        self.b_bias = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=data_query["gmap_id"].unique(),
                    mask_token=None,
                    num_oov_indices=1,
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(data_query["gmap_id"].unique()) + 2, output_dim=1
                ),
            ]
        )
        self.r_bias = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=data_query["reviewer_id"].unique(),
                    mask_token=None,
                    num_oov_indices=1,
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(data_query["reviewer_id"].unique()) + 2, output_dim=1
                ),
            ]
        )

        # start with the average rating for learnable unknown users
        self.global_bias = tf.Variable(initial_value=3.5, trainable=True)

        # dnn neural collaborative filtering
        self.dense_layers = tf.keras.Sequential(
            [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(dense_units, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

        # prediction, encourages on the ordering more, pure wise ranking or BPR?
        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def call(self, features):
        b_embeds = self.b_embed(features["gmap_id"])
        r_embeds = self.r_embed(features["reviewer_id"])
        b_bias = self.b_bias(features["gmap_id"])
        r_bias = self.r_bias(features["reviewer_id"])

        # Concatenate embeddings and pass through dense layers
        concatenated_embeddings = tf.concat([b_embeds, r_embeds], axis=1)
        interaction_score = self.dense_layers(concatenated_embeddings)

        return tf.squeeze(interaction_score) + tf.squeeze(b_bias) + tf.squeeze(r_bias)

    def compute_loss(self, features, training=False):
        ratings = features["rating"]
        rating_predictions = self(features)
        return self.rating_task(labels=ratings, predictions=rating_predictions)
