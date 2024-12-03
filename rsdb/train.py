from preprocess.data_preprocessing import *
from features.featuring import *
from models.tdlf.temporal_dynamic_v import TemporalDynamicVariants
from models.fpmc.fpmc_v import FPMCVariants
import tensorflow as tf
import kerastuner as kt
import argparse
import yaml

URL = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/review-California_10.json.gz"
METAURL = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/meta-California.json.gz"

def load_config(config_path):
    """Load and validate YAML configuration."""
    def validate_and_cast(value):
        if isinstance(value, str):
            try:
                # Try to cast to float if it's a scientific notation string
                return float(value)
            except ValueError:
                return value
        return value

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Recursively apply the validation and casting function
    def recursive_validate_cast(data):
        if isinstance(data, dict):
            return {k: recursive_validate_cast(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [recursive_validate_cast(item) for item in data]
        else:
            return validate_and_cast(data)

    return recursive_validate_cast(config)

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

def train(model_name, config_path="rsdb/configs/train_config.yaml"):
    """Training for both TemporalDynamicVariants and FPMCVariants"""
    config = load_config(config_path)

    cleaned_df = get_clean_review_data(URL, METAURL)
    featured_df = featuring_engineering(cleaned_df)
    data_query = featured_df[["gmap_id", "reviewer_id", "rating"]]
    train_frac = config["training"]["dataset_split"]["train_frac"]
    random_state = config["training"]["dataset_split"]["random_state"]
    train_df = featured_df.sample(frac=train_frac, random_state=random_state)
    test_df = featured_df.drop(train_df.index)

    shuffle_buffer_size = config["training"]["shuffle_buffer_size"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]
    min_delta = config["training"]["min_delta"]

    # Model-specific configuration
    if model_name == "tdlf":
        embedding_dim = config["tdlf"]["embedding_dim"]
        dense_units = config["tdlf"]["dense_units"]
        l2_reg = config["tdlf"]["l2_reg"]
        time_bins = config["tdlf"]["time_bins"]

        train_data = tdlf_df_to_tf_dataset(train_df).shuffle(shuffle_buffer_size).batch(batch_size)
        test_data = tdlf_df_to_tf_dataset(test_df).batch(batch_size)

        model = TemporalDynamicVariants(
            l2_reg, dense_units, embedding_dim, data_query, time_bins
        )

        lr_schedule_config = config["training"]["learning_rate_schedule"]
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr_schedule_config["initial_learning_rate"],
            decay_steps=lr_schedule_config["decay_steps"],
            decay_rate=lr_schedule_config["decay_rate"],
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_root_mean_squared_error",
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
        )

        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule))

    elif model_name == "fpmc":
        embedding_dim = config["fpmc"]["embedding_dim"]
        l2_reg = config["fpmc"]["l2_reg"]
        learning_rate = config["fpmc"]["learning_rate"]

        train_data = fpmc_df_to_tf_dataset(train_df).shuffle(shuffle_buffer_size).batch(batch_size)
        test_data = fpmc_df_to_tf_dataset(test_df).batch(batch_size)

        model = FPMCVariants(
            l2_reg=l2_reg, embedding_dim=embedding_dim, data_query=data_query
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_root_mean_squared_error",
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
        )

        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate))

    # Training and saving
    model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        callbacks=[early_stopping],
    )

    save_path = config["training"]["model_save_path"].format(model_name=model_name)
    model.save(save_path)
    print(f"Trained {model_name} model saved as '{save_path}'.")
    
    test_metrics = model.evaluate(test_data, return_dict=True)
    print(f"Test RMSE: {test_metrics['root_mean_squared_error']}")


def tune(model_name, config_path="rsdb/configs/tune_config.yaml"):
    """Hyperparameter tuning for both TemporalDynamicVariants and FPMCVariants"""
    config = load_config(config_path)

    # Dataset preparation
    cleaned_df = get_clean_review_data(URL, METAURL)
    featured_df = featuring_engineering(cleaned_df)
    data_query = featured_df[["gmap_id", "reviewer_id", "rating"]]

    train_frac = config["tuning"]["train_frac"]
    random_state = config["tuning"]["random_state"]
    train_df = featured_df.sample(frac=train_frac, random_state=random_state)
    test_df = featured_df.drop(train_df.index)

    shuffle_buffer_size = config["tuning"]["shuffle_buffer_size"]
    batch_size = config["tuning"]["batch_size"]

    train_tdlf_data = tdlf_df_to_tf_dataset(train_df).shuffle(shuffle_buffer_size).batch(batch_size)
    test_tdlf_data = tdlf_df_to_tf_dataset(test_df).batch(batch_size)
    train_fpmc_data = fpmc_df_to_tf_dataset(train_df).shuffle(shuffle_buffer_size).batch(batch_size)
    test_fpmc_data = fpmc_df_to_tf_dataset(test_df).batch(batch_size)

    # Tuning setup
    search_epochs = config["tuning"]["search_epochs"]
    max_epochs = config["tuning"]["max_epochs"]
    factor = config["tuning"]["factor"]

    def build_tdlf_model(hp):
        params = config["tdlf_hyperparameters"]
        l2_reg = hp.Float("l2_reg", min_value=params["l2_reg"]["min"], max_value=params["l2_reg"]["max"], sampling=params["l2_reg"]["sampling"])
        dense_units = hp.Int("dense_units", min_value=params["dense_units"]["min"], max_value=params["dense_units"]["max"], step=params["dense_units"]["step"])
        embedding_dim = hp.Int("embedding_dim", min_value=params["embedding_dim"]["min"], max_value=params["embedding_dim"]["max"], step=params["embedding_dim"]["step"])
        time_bins = hp.Int("time_bins", min_value=params["time_bins"]["min"], max_value=params["time_bins"]["max"], step=params["time_bins"]["step"])
        learning_rate = hp.Float("learning_rate", min_value=params["learning_rate"]["min"], max_value=params["learning_rate"]["max"], sampling=params["learning_rate"]["sampling"])

        model = TemporalDynamicVariants(
            l2_reg, dense_units, embedding_dim, data_query, time_bins
        )
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=params["decay_steps"],
            decay_rate=params["decay_rate"],
        )
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule))
        return model


    def build_fpmc_model(hp):
        params = config["fpmc_hyperparameters"]
        l2_reg = hp.Float("l2_reg", min_value=params["l2_reg"]["min"], max_value=params["l2_reg"]["max"], sampling=params["l2_reg"]["sampling"])
        embedding_dim = hp.Int("embedding_dim", min_value=params["embedding_dim"]["min"], max_value=params["embedding_dim"]["max"], step=params["embedding_dim"]["step"])
        learning_rate = hp.Float("learning_rate", min_value=params["learning_rate"]["min"], max_value=params["learning_rate"]["max"], sampling=params["learning_rate"]["sampling"])

        model = FPMCVariants(
            l2_reg=l2_reg, embedding_dim=embedding_dim, data_query=data_query
        )
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate))
        return model


    # Hyperparameter tuning for the selected model
    if model_name == "tdlf":
        print("Starting hypertuning for TemporalDynamicVariants...")
        tdlf_tuner = kt.Hyperband(
            build_tdlf_model,
            objective=kt.Objective("val_root_mean_squared_error", direction="min"),
            max_epochs=max_epochs,
            factor=factor,
            directory=config["tuning"]["directories"]["tdlf"],
            project_name=config["tuning"]["project_names"]["tdlf"],
        )
        tdlf_tuner.search(train_tdlf_data, validation_data=test_tdlf_data, epochs=search_epochs)
        best_tdlf_hp = tdlf_tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters for TemporalDynamicVariants:", best_tdlf_hp.values)
        best_model = tdlf_tuner.hypermodel.build(best_tdlf_hp)

    elif model_name == "fpmc":
        print("Starting hypertuning for FPMCVariants...")
        fpmc_tuner = kt.Hyperband(
            build_fpmc_model,
            objective=kt.Objective("val_root_mean_squared_error", direction="min"),
            max_epochs=max_epochs,
            factor=factor,
            directory=config["tuning"]["directories"]["fpmc"],
            project_name=config["tuning"]["project_names"]["fpmc"],
        )
        fpmc_tuner.search(train_fpmc_data, validation_data=test_fpmc_data, epochs=search_epochs)
        best_fpmc_hp = fpmc_tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters for FPMCVariants:", best_fpmc_hp.values)
        best_model = fpmc_tuner.hypermodel.build(best_fpmc_hp)

    # Save the best model
    save_path = f"trained_{model_name}_model"
    best_model.save(save_path)
    print(f"Trained {model_name} model saved as '{save_path}'.")
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
