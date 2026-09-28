import numpy as np


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Square Error (RMSE).

    Parameters
    ----------
    y_true : array-like
        Reference depth values.

    y_pred : array-like
        Predicted depth values.

    Returns
    -------
    float
        RMSE.
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    validate_inputs(y_true, y_pred)

    return float(
        np.sqrt(
            np.mean(
                (y_pred - y_true) ** 2
            )
        )
    )


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE).
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    validate_inputs(y_true, y_pred)

    return float(
        np.mean(
            np.abs(y_pred - y_true)
        )
    )


def calculate_r2(y_true, y_pred):
    """
    Calculate coefficient of determination (R²).
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    validate_inputs(y_true, y_pred)

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot == 0:
        raise ValueError(
            "R² is undefined when "
            "all reference values are identical."
        )

    return float(
        1.0 - ss_res / ss_tot
    )


def calculate_bias(y_true, y_pred):
    """
    Calculate mean prediction bias.

    Positive value:
        predictions are deeper than reference values.

    Negative value:
        predictions are shallower than reference values.
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    validate_inputs(y_true, y_pred)

    return float(
        np.mean(
            y_pred - y_true
        )
    )


def validate_inputs(y_true, y_pred):
    """
    Validate metric inputs.
    """

    if y_true.ndim != 1:
        raise ValueError(
            f"y_true must be 1D, "
            f"got shape {y_true.shape}"
        )

    if y_pred.ndim != 1:
        raise ValueError(
            f"y_pred must be 1D, "
            f"got shape {y_pred.shape}"
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must "
            "have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Input arrays must not be empty."
        )

    if not np.all(np.isfinite(y_true)):
        raise ValueError(
            "y_true contains NaN or infinite values."
        )

    if not np.all(np.isfinite(y_pred)):
        raise ValueError(
            "y_pred contains NaN or infinite values."
        )


def calculate_all_metrics(y_true, y_pred):
    """
    Calculate all standard SDB regression metrics.

    Returns
    -------
    dict
        Dictionary containing RMSE, MAE, R², and Bias.
    """

    return {
        "RMSE": calculate_rmse(
            y_true,
            y_pred
        ),
        "MAE": calculate_mae(
            y_true,
            y_pred
        ),
        "R2": calculate_r2(
            y_true,
            y_pred
        ),
        "Bias": calculate_bias(
            y_true,
            y_pred
        ),
    }


if __name__ == "__main__":

    print(
        "========== Metrics Test =========="
    )

    # Synthetic test data.
    # The expected values can be calculated manually.
    y_true = np.array(
        [-1.0, -2.0, -3.0, -4.0]
    )

    y_pred = np.array(
        [-1.1, -1.8, -3.2, -3.9]
    )

    metrics = calculate_all_metrics(
        y_true,
        y_pred
    )

    print(
        f"RMSE: {metrics['RMSE']:.6f}"
    )

    print(
        f"MAE: {metrics['MAE']:.6f}"
    )

    print(
        f"R²: {metrics['R2']:.6f}"
    )

    print(
        f"Bias: {metrics['Bias']:.6f}"
    )

    print("\nMetrics test passed.")