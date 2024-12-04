# the function will takes trained model and return pandas dataframe contain the metrics of the model
import pandas as pd
import numpy as np
from rsdb.recommendation import Recommendation
from sklearn.metrics import recall_score, precision_score, f1_score


def calculate_mse(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    y_true (numpy.ndarray): Array of true values.
    y_pred (numpy.ndarray): Array of predicted values.

    Returns:
    float: The Mean Squared Error value.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("The input arrays must have the same length.")

    # Calculate MSE
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


def calculate_rmse(arr1, arr2):
    """
    Calculate the Root Mean Squared Error (RMSE) between two arrays.

    Parameters:
    arr1 (numpy.ndarray): The first array.
    arr2 (numpy.ndarray): The second array.

    Returns:
    float: The RMSE value.
    """
    if len(arr1) != len(arr2):
        raise ValueError("The input arrays must have the same length.")

    # Compute RMSE
    rmse = np.sqrt(np.mean((arr1 - arr2) ** 2))
    return rmse


def calculate_r2(y_true, y_pred):
    """
    Calculate the R-squared (coefficient of determination) value.

    Parameters:
    y_true (numpy.ndarray): Array of true values.
    y_pred (numpy.ndarray): Array of predicted values.

    Returns:
    float: The R-squared value.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("The input arrays must have the same length.")

    # Residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # R-squared
    r2 = 1 - (ss_res / ss_tot)
    return r2


def calculate_mase(y_true, y_pred):
    """
    Calculate the Mean Absolute Scaled Error (MASE).

    Parameters:
    y_true (numpy.ndarray): Array of true values.
    y_pred (numpy.ndarray): Array of predicted values.

    Returns:
    float: The MASE value.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("The input arrays must have the same length.")
    if len(y_true) < 2:
        raise ValueError(
            "The input array must have at least two elements for MASE calculation."
        )

    # Mean absolute error (numerator)
    mae = np.mean(np.abs(y_true - y_pred))

    # Mean absolute difference (denominator)
    mad = np.mean(np.abs(np.diff(y_true)))

    # Avoid division by zero
    if mad == 0:
        raise ValueError(
            "Mean absolute difference of true values is zero. MASE cannot be calculated."
        )

    # Calculate MASE
    mase = mae / mad
    return mase


def calculate_recall_multiclass(y_true, y_pred):
    """
    Calculate Recall for multi-class ratings.
    Macro-averaging ensures all classes are equally weighted.
    """
    return recall_score(y_true, y_pred, average="macro")


def calculate_precision_multiclass(y_true, y_pred):
    """
    Calculate Precision for multi-class ratings.
    Macro-averaging ensures all classes are equally weighted.
    """
    return precision_score(y_true, y_pred, average="macro")


def calculate_f1_multiclass(y_true, y_pred):
    """
    Calculate F1 Score for multi-class ratings.
    Macro-averaging ensures all classes are equally weighted.
    """
    return f1_score(y_true, y_pred, average="macro")


def eval_result(models: list, tf_test_datas: list) -> pd.DataFrame:
    """
    evaluate the result
    """
    columns = ["mse", "rmse", "r2", "mase", "recall", "precision", "f1", "acc"]

    result_list = []
    # eval the perforamnce of each model
    for model, tf_test_data in zip(models, tf_test_datas):
        # convert tensorflow data into array
        rating_column_batched = tf_test_data.unbatch().map(lambda x: x["rating"])
        actual_rating = np.array(list(rating_column_batched.as_numpy_iterator()))

        metrics = [model.name]
        prediction = np.round(model.predict(tf_test_data))
        metrics.extend(
            [                
                calculate_mse(actual_rating, prediction),
                calculate_rmse(actual_rating, prediction),
                calculate_r2(actual_rating, prediction),
                calculate_mase(actual_rating, prediction),
                calculate_recall_multiclass(actual_rating, prediction),
                calculate_precision_multiclass(actual_rating, prediction),
                calculate_f1_multiclass(actual_rating, prediction),
                np.mean(np.abs(actual_rating == prediction)),
            ]
        )
        result_list.append(metrics)

    return pd.DataFrame(
        [row[1:] for row in result_list],
        columns=columns,
        index=[row[0] for row in result_list],
    )


def eval_downstream(models: list, clean_df: pd.DataFrame, model_names: list):
    # Define a function that evaluates recommendation models by comparing their predictions against a subset of data.
    # models: list of trained recommendation models.
    # clean_df: pre-processed DataFrame containing relevant data for evaluation.
    # model_names: list of names corresponding to the models, used for reporting.

    print("please make sure all the tensorflow data is same")
    # Reminder to ensure consistency in TensorFlow-related data if applicable (likely meant for debugging).

    seed = 1
    random_sample_gmapids = clean_df.sample(5, random_state=seed)["gmap_id"].values
    # Randomly select 5 unique `gmap_id` values from the `clean_df` for evaluation, ensuring reproducibility with `random_state`.

    gmap_ids_property = [
        list(clean_df[clean_df["gmap_id"] == gmap_id]["category"].iloc[0])
        for gmap_id in random_sample_gmapids
    ]
    # Retrieve the first "category" value for each of the sampled `gmap_id`s, storing them for reference.
    # Assumes each `gmap_id` is associated with a single "category".

    for model, model_name in zip(models, model_names):
        # Iterate through each model and its corresponding name for evaluation.

        recomender = Recommendation(model, clean_df, model_name)
        # Instantiate a `Recommendation` object with the current model, dataset, and name.

        recommend_dfs = [
            recomender.recommend(sample_gmap) for sample_gmap in random_sample_gmapids
        ]
        # Generate recommendations for each sampled `gmap_id` using the current model.
        # `recommend()` presumably returns a DataFrame of recommendations.

        for recommend_df in recommend_dfs:
            recommend_df["reviewer_id"] = recommend_df["reviewer_id"].astype(float)
            # Cast the `reviewer_id` column to float, potentially to align data types for later operations.

        recommend_dfs = [
            recommend_df.merge(
                clean_df[clean_df["gmap_id"] != gmap_id], on="reviewer_id"
            )
            for recommend_df, gmap_id in zip(recommend_dfs, random_sample_gmapids)
        ]
        # For each recommendation DataFrame, merge it with `clean_df` entries that do not match the original `gmap_id`.
        # Likely to exclude the businesses already associated with the sampled `gmap_id`.

        print(f"for model: {model_name}")
        # Display the name of the model being evaluated.

        for recommend_df, gmap_prop in zip(recommend_dfs, gmap_ids_property):
            # Iterate over each recommendation DataFrame and the corresponding property of the sampled `gmap_id`.

            print(f"The given gmapids has the property {gmap_prop}")
            # Print the category of the sampled `gmap_id`.

            print(
                "Of all the recommended user, their categorical visited business are in these categories"
            )
            print(
                recommend_df["category"].explode().value_counts(normalize=True).iloc[:3]
            )
            # Display the top 3 categories of businesses visited by users in the recommendations,
            # along with their normalized frequencies.

            print("\n")

            print("Their average rating is:")
            print(recommend_df.groupby("reviewer_id")["rating"].mean())
            # Compute and print the average rating given by each user in the recommendations.

            print("*----------------------------*")
            # Separator for clarity in the output.

        print("*----------------------------*")
        # Additional separator for separating results of different models.


def main():
    pass


if __name__ == "__main__":
    main()
