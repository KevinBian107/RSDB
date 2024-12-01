import tensorflow as tf
import pandas as pd
import keras_tuner as kt

from tlfm.latent_factor import LatentFactorModel
from tlfm.temporal_static import TemporalStaticModel
from tlfm.temporal_dynamic import TemporalDynamicModel


def latent_factor_tunning(data:pd.DataFrame):
    """
    Perform hyperparameter tuning for the Latent Factor Model.

    Performs train-test splitting, and converts into TensorFlow datasets.
    Then uses Keras Tuner to search for best hyperparameters for the `LatentFactorModel`. 
    The best model's weight are saved for future use.

    Args:
        data: A pandas DataFrame containing columns 'gmap_id', 'reviewer_id', and 'rating'.

    Returns:
        model: LatentFactorModel
    """
    
    data = data[['gmap_id', 'reviewer_id', 'rating']]

    # train-test split
    train_df = data.sample(frac=0.8, random_state=42)
    test_df = data.drop(train_df.index)

    def df_to_tf_dataset(dataframe):
        return tf.data.Dataset.from_tensor_slices({
            "gmap_id": dataframe["gmap_id"].values,
            "reviewer_id": dataframe["reviewer_id"].values,
            "rating": dataframe["rating"].values
        })

    train_data = df_to_tf_dataset(train_df).shuffle(1024).batch(4096)
    test_data = df_to_tf_dataset(test_df).batch(4096)

    '''------------------hyperparameter tunning------------------'''
    def build_model_latent_factor(hp):
        embedding_dim = hp.Choice("embedding_dim", values=[16, 32, 64, 90, 128])
        dense_unit = hp.Choice("dense_unit", values=[16, 32])
        l2_reg = hp.Float("l2_reg", min_value=0.0001, max_value=0.1, step=0.01)
        lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        
        model = LatentFactorModel(l2_reg=l2_reg, embedding_dim=embedding_dim, dense_units=dense_unit, data_query=data)
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr)) # I use legacy version of Adam, change it when needed
        return model

    tuner = kt.RandomSearch(
        build_model_latent_factor,
        objective=kt.Objective("val_root_mean_squared_error", direction="min"),
        max_trials=5,
        executions_per_trial=2,
        directory="latent_hyperparameter_tuning",
        project_name="rating_model_tuning"
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_root_mean_squared_error", 
        patience=10,
        min_delta=0.001,
        restore_best_weights=True
    )

    # start seaching
    tuner.search(train_data, epochs=150, validation_data=test_data, callbacks=[early_stopping])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    print("Best hyperparameter values:")
    for hp_name, hp_value in best_hps.values.items():
        print(f"{hp_name}: {hp_value}")

    model.fit(train_data, epochs=500, validation_data=test_data, callbacks=[early_stopping])

    # save model weight
    model.save_weights("latent_factor_model_weights.tf")

    return model



def temporal_data_split(data: pd.DataFrame) -> pd.DataFrame:
    train_df = data.sample(frac=0.8, random_state=42)
    test_df = data.drop(train_df.index)

    # Calculate mean and std for normalization
    time_mean, time_std = train_df["review_time(unix)"].mean(), train_df["review_time(unix)"].std()
    user_mean_time_mean, user_mean_time_std = train_df["user_mean_time"].mean(), train_df["user_mean_time"].std()

    # Normalize the training and test data
    train_df["review_time(unix)"] = (train_df["review_time(unix)"] - time_mean) / time_std
    test_df["review_time(unix)"] = (test_df["review_time(unix)"] - time_mean) / time_std
    train_df["user_mean_time"] = (train_df["user_mean_time"] - user_mean_time_mean) / user_mean_time_std
    test_df["user_mean_time"] = (test_df["user_mean_time"] - user_mean_time_mean) / user_mean_time_std

    # Function to convert DataFrame to TensorFlow dataset
    def df_to_tf_dataset(dataframe):
        return tf.data.Dataset.from_tensor_slices({
            "reviewer_id": dataframe["reviewer_id"].astype(str),
            "gmap_id": dataframe["gmap_id"].astype(str),
            "time": dataframe["review_time(unix)"].astype(float),
            "time_bin": dataframe["time_bin"].astype(float),
            "user_mean_time": dataframe["user_mean_time"],
            "rating": dataframe["rating"]
        })

    # Create TensorFlow datasets
    train_data = df_to_tf_dataset(train_df).shuffle(1024).batch(4096)
    test_data = df_to_tf_dataset(test_df).batch(4096)

    return train_data, test_data


def temporal_static_tunning(data: pd.DataFrame):
    """
    Perform hyperparameter tuning for the Temporal Static Model.

    Performs train-test splitting, and converts into TensorFlow datasets.
    Then uses Keras Tuner to search for best hyperparameters for the `TemporalStaticModel`. 
    The best model's weight are saved for future use.

    Args:
        data: A pandas DataFrame containing columns 'reviewer_id', 'gmap_id', 'time', 
        'time_bin', 'user_mean_time', and 'rating'.

    Returns:
        model: TemporalStaticModel
    """

    data = data[['reviewer_id', 'gmap_id', 'time', 'time_bin', 'user_mean_time', 'rating']]

    train_data, test_data = temporal_data_split(data)

    '''------------------hyperparameter tunning------------------'''
    def build_model_temporal_static(hp):
        embedding_dim = hp.Choice("embedding_dim", values=[16, 32, 64, 90])
        dense_unit = hp.Choice("dense_unit", values=[16, 32])
        l2_reg = hp.Float("l2_reg", min_value=0.0001, max_value=0.1, step=0.01)
        time_bins= hp.Choice("dense_unit", values=[5, 10, 20])
        lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        
        model = TemporalStaticModel(l2_reg=l2_reg, embedding_dim=embedding_dim, dense_units=dense_unit, time_bins=time_bins, data_query=data)
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr)) # I use legacy version of Adam, change it when needed
        return model

    tuner = kt.RandomSearch(
        build_model_temporal_static,
        objective=kt.Objective("val_root_mean_squared_error", direction="min"),
        max_trials=5,
        executions_per_trial=2,
        directory="latent_hyperparameter_tuning",
        project_name="rating_model_tuning"
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_root_mean_squared_error", 
        patience=10,
        min_delta=0.001,
        restore_best_weights=True
    )

    tuner.search(train_data, epochs=150, validation_data=test_data, callbacks=[early_stopping])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)

    print("Best hyperparameter values:")
    for hp_name, hp_value in best_hps.values.items():
        print(f"{hp_name}: {hp_value}")

    model.fit(train_data, epochs=500, validation_data=test_data, callbacks=[early_stopping])

    # save model weight
    model.save_weights("temporal_static_model_weights.tf")

    return model



def temporal_dynamic_tunning(data: pd.DataFrame):
    """
    Perform hyperparameter tuning for the Temporal Dynamic Model.

    Performs train-test splitting, and converts into TensorFlow datasets.
    Then uses Keras Tuner to search for best hyperparameters for the `TemporalDynamicModel`. 
    The best model's weight are saved for future use.

    Args:
        data: A pandas DataFrame containing columns 'reviewer_id', 'gmap_id', 'time', 
        'time_bin', 'user_mean_time', and 'rating'.

    Returns:
        model: TemporalDynamicModel
    """

    data = data[['reviewer_id', 'gmap_id', 'time', 'time_bin', 'user_mean_time', 'rating']]

    train_data, test_data = temporal_data_split(data)

    '''------------------hyperparameter tunning------------------'''
    def build_model_temporal_dynamic(hp):
        embedding_dim = hp.Choice("embedding_dim", values=[16, 32, 64, 90])
        dense_unit = hp.Choice("dense_unit", values=[16, 32])
        l2_reg = hp.Float("l2_reg", min_value=0.0001, max_value=0.1, step=0.01)
        time_bins= hp.Choice("dense_unit", values=[5, 10, 20])
        lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        
        model = TemporalDynamicModel(l2_reg=l2_reg, embedding_dim=embedding_dim, dense_units=dense_unit, time_bins=time_bins, data_query=data)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        return model

    tuner = kt.RandomSearch(
        build_model_temporal_dynamic,
        objective=kt.Objective("val_root_mean_squared_error", direction="min"),
        max_trials=5,
        executions_per_trial=2,
        directory="latent_hyperparameter_tuning",
        project_name="rating_model_tuning"
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_root_mean_squared_error", 
        patience=10,
        min_delta=0.001,
        restore_best_weights=True
    )

    tuner.search(train_data, epochs=150, validation_data=test_data, callbacks=[early_stopping])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)

    print("Best hyperparameter values:")
    for hp_name, hp_value in best_hps.values.items():
        print(f"{hp_name}: {hp_value}")

    model.fit(train_data, epochs=500, validation_data=test_data, callbacks=[early_stopping])

    # save model weight
    model.save_weights("temporal_dynamic_model_weights.tf")

    return model

