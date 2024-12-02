import tensorflow as tf
import tensorflow_recommenders as tfrs


class FPMCModel(tfrs.Model):
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

        # Global bias
        self.global_bias = tf.Variable(initial_value=3.5, trainable=True)

        # Rating task
        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def call(self, features):
        """
        Predict ratings for (user, prev_item, next_item) triplets.
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

        # Total predicted rating
        return (
            user_next_score + item_next_score + user_bias + item_bias + self.global_bias
        )

    def compute_loss(self, features, training=False):
        """
        Compute MSE loss for predicted ratings.
        """
        ratings = features["rating"]
        predictions = self(features)
        return self.rating_task(labels=ratings, predictions=predictions)
