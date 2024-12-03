# the function will takes trained model and return pandas dataframe contain the metrics of the model
import pandas as pd
import numpy as np


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


def eval_result(models: list, tf_test_data) -> pd.DataFrame:
    """
    evaluate the result
    """
    columns = ["mse", "rmse", "r2", "mase"]

    # convert tensorflow data into array
    rating_column_batched = tf_test_data.unbatch().map(lambda x: x["rating"])
    actual_rating = np.array(list(rating_column_batched.as_numpy_iterator()))

    result_list = []
    # eval the perforamnce of each model
    for model in models:
        metrics = [model.name]
        prediction = model.predict(tf_test_data)
        metrics.extend(
            [
                calculate_mse(actual_rating, prediction),
                calculate_rmse(actual_rating, prediction),
                calculate_r2(actual_rating, prediction),
                calculate_mase(actual_rating, prediction),
            ]
        )
        result_list.append(metrics)

    return pd.DataFrame(
        [row[1:] for row in result_list],
        columns=columns,
        index=[row[0] for row in result_list],
    )


def main():
    pass


if __name__ == "__main__":
    main()
