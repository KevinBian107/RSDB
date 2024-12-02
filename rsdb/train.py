from preprocess.data_preprocessing import *
from features.featuring import *
from models.tdlf.temporal_dynamic_v import TemporalDynamicVariants
from models.fpmc.fpmc_v import FPMCVariants
import tensorflow as tf
import kerastuner as kt
import argparse

URL = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/review-California_10.json.gz"
METAURL = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/meta-California.json.gz"


def tdlf_df_to_tf_dataset(dataframe):
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


def fpmc_df_to_tf_dataset(dataframe):
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


def train(model_name):
    """Training for botrh TemporalDynamicVariants and FPMCVariants"""

    cleaned_df = get_single_chunk(URL, METAURL)
    featured_df = featuring_engineering(cleaned_df)

    data_query = featured_df[["gmap_id", "reviewer_id", "rating"]]
    train_df = featured_df.sample(frac=0.8, random_state=42)
    test_df = featured_df.drop(train_df.index)

    if model_name == "tdlf":
        train_data = tdlf_df_to_tf_dataset(train_df).shuffle(1024).batch(4096)
        test_data = tdlf_df_to_tf_dataset(test_df).batch(4096)
        embedding_dim = 30
        dense_units = 30
        l2_reg = 0.0201
        time_bins = 30
        model = TemporalDynamicVariants(
            l2_reg, dense_units, embedding_dim, data_query, time_bins
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_root_mean_squared_error",
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2, decay_steps=1000, decay_rate=0.8
        )

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))
        model.fit(
            train_data,
            epochs=500,
            validation_data=test_data,
            callbacks=[early_stopping],
        )

    if model_name == "fpmc":
        train_data = fpmc_df_to_tf_dataset(train_df).shuffle(1024).batch(4096)
        test_data = fpmc_df_to_tf_dataset(test_df).batch(4096)
        embedding_dim = 32
        l2_reg = 0.0201
        lr = 1e-3
        model = FPMCVariants(
            l2_reg=l2_reg, embedding_dim=embedding_dim, data_query=data_query
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_root_mean_squared_error",
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
        )

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        model.fit(
            train_data,
            validation_data=test_data,
            epochs=500,
            callbacks=[early_stopping],
        )

    # Save trained model
    model.save(f"trained_{model_name}_model")
    print(f"Trained {model_name} model saved as 'trained_{model_name}_model'.")


def tune(model_name):
    """Hyperparameter tunning for botrh TemporalDynamicVariants and FPMCVariants"""

    cleaned_df = get_single_chunk(URL, METAURL)
    featured_df = featuring_engineering(cleaned_df)

    data_query = featured_df[["gmap_id", "reviewer_id", "rating"]]
    train_df = featured_df.sample(frac=0.8, random_state=42)
    test_df = featured_df.drop(train_df.index)

    # Data preparation
    train_tdlf_data = tdlf_df_to_tf_dataset(train_df).shuffle(1024).batch(4096)
    test_tdlf_data = tdlf_df_to_tf_dataset(test_df).batch(4096)

    train_fpmc_data = fpmc_df_to_tf_dataset(train_df).shuffle(1024).batch(4096)
    test_fpmc_data = fpmc_df_to_tf_dataset(test_df).batch(4096)

    # Define hyperparameter search space for TemporalDynamicVariants
    def build_tdlf_model(hp):
        l2_reg = hp.Float("l2_reg", min_value=1e-5, max_value=1e-2, sampling="log")
        dense_units = hp.Int("dense_units", min_value=16, max_value=128, step=16)
        embedding_dim = hp.Int("embedding_dim", min_value=8, max_value=64, step=8)
        time_bins = hp.Int("time_bins", min_value=5, max_value=50, step=5)

        model = TemporalDynamicVariants(
            l2_reg, dense_units, embedding_dim, data_query, time_bins
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
            ),
            decay_steps=1000,
            decay_rate=0.8,
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))
        return model

    # Define hyperparameter search space for FPMCVariants
    def build_fpmc_model(hp):
        l2_reg = hp.Float("l2_reg", min_value=1e-5, max_value=1e-2, sampling="log")
        embedding_dim = hp.Int("embedding_dim", min_value=8, max_value=128, step=8)

        model = FPMCVariants(
            l2_reg=l2_reg, embedding_dim=embedding_dim, data_query=data_query
        )

        lr = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        return model

    if model_name == "tdlf":
        print("Starting hypertuning for TemporalDynamicVariants...")
        tdlf_tuner = kt.Hyperband(
            build_tdlf_model,
            objective=kt.Objective("val_root_mean_squared_error", direction="min"),
            max_epochs=50,
            factor=3,
            directory="hypertune_tdlf",
            project_name="tdlf_tuning",
        )

        tdlf_tuner.search(train_tdlf_data, validation_data=test_tdlf_data, epochs=10)
        best_tdlf_hp = tdlf_tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters for TemporalDynamicVariants:")
        print(best_tdlf_hp.values)
        best_model = tdlf_tuner.hypermodel.build(best_tdlf_hp)

    if model_name == "fpmc":
        print("\nStarting hypertuning for FPMCVariants...")
        fpmc_tuner = kt.Hyperband(
            build_fpmc_model,
            objective=kt.Objective("val_root_mean_squared_error", direction="min"),
            max_epochs=50,
            factor=3,
            directory="hypertune_fpmc",
            project_name="fpmc_tuning",
        )

        fpmc_tuner.search(train_fpmc_data, validation_data=test_fpmc_data, epochs=10)
        best_fpmc_hp = fpmc_tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters for FPMCVariants:")
        print(best_fpmc_hp.values)
        best_model = fpmc_tuner.hypermodel.build(best_fpmc_hp)

    # Save trained model
    best_model.save(f"trained_{model_name}_model")
    print(f"Trained {model_name} model saved as 'trained_{model_name}_model'.")
    return best_model


def main():
    parser = argparse.ArgumentParser(
        description="Run train or tune for TemporalDynamicVariants (tdlf) or FPMCVariants (fpmc)."
    )
    parser.add_argument(
        "--action",
        choices=["train", "tune"],
        help="Specify whether to train or tune the model.",
    )
    parser.add_argument(
        "--model",
        choices=["tdlf", "fpmc"],
        help="Specify the model to use (tdlf or fpmc).",
    )
    args = parser.parse_args()

    if args.action == "train":
        print(f"Training {args.model} model...")
        train(args.model)
        print("Training completed. Parameter Saved")

    elif args.action == "tune":
        print(f"Tuning {args.model} model...")
        best_model = tune(args.model)
        print(f"Tuning completed for {args.model}.")
        # Optionally save the best model after tuning
        best_model.save(f"best_{args.model}_model")
        print(f"Best {args.model} model saved as 'best_{args.model}_model'.")


if __name__ == "__main__":
    main()
