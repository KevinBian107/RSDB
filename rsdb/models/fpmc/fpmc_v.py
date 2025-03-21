import tensorflow as tf
import tensorflow_recommenders as tfrs

class FPMCVariants(tfrs.Model):
    def __init__(self, l2_reg, embedding_dim, data_query):
        super().__init__()

        self.l2_reg = l2_reg
        self.embedding_dim = embedding_dim
        self.num_users = data_query["reviewer_id"].nunique()
        self.num_items = data_query["gmap_id"].nunique()

        # User-item embeddings
        self.user_to_item_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_users + 1,
            output_dim=embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
        )

        # Item-user embeddings
        self.item_to_user_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_items + 1,
            output_dim=embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
        )

        # Item-item embeddings
        self.item_to_item_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_items + 1,
            output_dim=embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
        )

        # User and item biases
        self.user_bias = tf.keras.layers.Embedding(
            input_dim=self.num_users + 1, output_dim=1
        )
        self.item_bias = tf.keras.layers.Embedding(
            input_dim=self.num_items + 1, output_dim=1
        )

        # Global bias initialized to dataset average
        self.global_bias = tf.Variable(
            initial_value=tf.constant(data_query["rating"].mean(), dtype=tf.float32),
            trainable=True,
        )

        # Additional feature embeddings
        self.category_embedding = tf.keras.layers.Embedding(
            input_dim=3,  # 3 categories: restaurant, park, store
            output_dim=embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
        )

        self.lon_bin_dense = tf.keras.layers.Dense(
            units=embedding_dim,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        )

        self.lat_bin_dense = tf.keras.layers.Dense(
            units=embedding_dim,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        )

        # Mixed DNN architecture for feature integration
        self.mixed_dnn = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.Dense(embedding_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        ])

        # Rating task
        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def preprocess_features(self, features):
        """Preprocess additional features."""
        # Process category embeddings
        category_emb = (
            features["isin_category_restaurant"][:, None] * self.category_embedding(0)
            + features["isin_category_park"][:, None] * self.category_embedding(1)
            + features["isin_category_store"][:, None] * self.category_embedding(2)
        )
        category_emb = tf.reduce_mean(category_emb, axis=1, keepdims=True)

        # Process longitude and latitude bins
        lon_bin_features = tf.concat(
            [
                features[f"lon_bin_{i}"][:, None]
                for i in range(20)
                if f"lon_bin_{i}" in features
            ],
            axis=1,
        )
        lon_bin_emb = self.lon_bin_dense(lon_bin_features)

        lat_bin_features = tf.concat(
            [
                features[f"lat_bin_{i}"][:, None]
                for i in range(20)
                if f"lat_bin_{i}" in features
            ],
            axis=1,
        )
        lat_bin_emb = self.lat_bin_dense(lat_bin_features)

        # Combine all features
        additional_features = tf.concat([category_emb, lon_bin_emb, lat_bin_emb], axis=1)
        return additional_features

    def call(self, features):
        """
        Predict ratings for (user, prev_item, next_item) triplets with additional features.
        """
        # Lookup embeddings
        user_ids = features["reviewer_id"]
        prev_item_ids = features["prev_item_id"]
        next_item_ids = features["next_item_id"]

        user_emb = self.user_to_item_embedding(user_ids)  # γ_{ui}
        next_item_emb_user = self.item_to_user_embedding(next_item_ids)  # γ_{iu}

        prev_item_emb = self.item_to_item_embedding(prev_item_ids)  # γ_{ij}
        next_item_emb_item = self.item_to_item_embedding(next_item_ids)  # γ_{ji}

        # User and next-item compatibility: γ_{ui} ⋅ γ_{iu}
        user_next_score = tf.reduce_sum(user_emb * next_item_emb_user, axis=1)

        # Next-item's compatibility with previous-item: γ_{ij} ⋅ γ_{ji}
        item_next_score = tf.reduce_sum(prev_item_emb * next_item_emb_item, axis=1)

        # Independent biases
        user_bias = tf.squeeze(self.user_bias(user_ids))
        item_bias = tf.squeeze(self.item_bias(next_item_ids))

        # Process additional features
        additional_features = self.preprocess_features(features)

        # Pass additional features through the mixed DNN
        mixed_feature_representation = self.mixed_dnn(additional_features)

        # Final prediction
        return (
            user_next_score
            + item_next_score
            + user_bias
            + item_bias
            + self.global_bias
            + tf.reduce_sum(mixed_feature_representation, axis=1)
        )

    def compute_loss(self, features, training=False):
        """
        Compute MSE loss for predicted ratings.
        """
        ratings = features["rating"]
        predictions = self(features)
        return self.rating_task(labels=ratings, predictions=predictions)

    def get_config(self):
        # Return a dictionary of the model's configuration
        return {
            "l2_reg": self.l2_reg,
            "embedding_dim": self.embedding_dim,
        }

    @classmethod
    def from_config(cls, config):
        # Create an instance of the model from the config
        return cls(**config)
