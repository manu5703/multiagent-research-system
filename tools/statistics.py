import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


REQUIRED_COLUMNS = [
    "age",
    "BMI",
    "baseline_HbA1c",
    "followup_HbA1c",
    "eGFR",
    "diabetes_duration",
    "medication_class"
]


def load_data(path: str):

    df = pd.read_csv(path)

    return df


def validate_columns(df):

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    return True


def calculate_treatment_response(df):

    df = df.copy()

    df["HbA1c_change"] = (
        df["followup_HbA1c"]
        - df["baseline_HbA1c"]
    )

    return df


def cohort_summary(df):

    summary = {

        "n_patients": int(
            len(df)
        ),

        "missing_values": (
            df.isnull()
            .sum()
            .to_dict()
        ),

        "medication_counts": (
            df[
                "medication_class"
            ]
            .value_counts()
            .to_dict()
        )
    }

    return summary


def medication_group_analysis(df):

    result = {}

    grouped = df.groupby(
        "medication_class"
    )

    for medication, group in grouped:

        result[medication] = {

            "n": int(len(group)),

            "mean_baseline_HbA1c": float(
                group[
                    "baseline_HbA1c"
                ].mean()
            ),

            "mean_followup_HbA1c": float(
                group[
                    "followup_HbA1c"
                ].mean()
            ),

            "mean_HbA1c_change": float(
                group[
                    "HbA1c_change"
                ].mean()
            ),

            "std_HbA1c_change": float(
                group[
                    "HbA1c_change"
                ].std()
            )
        }

    return result


def regression_analysis(df):

    features = [
        "age",
        "BMI",
        "baseline_HbA1c",
        "eGFR",
        "diabetes_duration"
    ]

    analysis_df = df.dropna(
        subset=features + [
            "HbA1c_change"
        ]
    ).copy()

    if len(analysis_df) < 10:

        return {
            "error": (
                "Not enough complete observations "
                "for regression."
            )
        }

    X = analysis_df[
        features
    ]

    y = analysis_df[
        "HbA1c_change"
    ]

    model = LinearRegression()

    model.fit(
        X,
        y
    )

    predictions = model.predict(X)

    coefficients = {}

    for feature, coefficient in zip(
        features,
        model.coef_
    ):

        coefficients[feature] = float(
            coefficient
        )

    return {

        "n": int(
            len(analysis_df)
        ),

        "features": features,

        "coefficients": coefficients,

        "intercept": float(
            model.intercept_
        ),

        "r_squared": float(
            r2_score(
                y,
                predictions
            )
        )
    }


def run_statistics(path: str):

    df = load_data(path)

    validate_columns(df)

    df = calculate_treatment_response(
        df
    )

    summary = cohort_summary(
        df
    )

    medication_results = (
        medication_group_analysis(
            df
        )
    )

    regression_results = (
        regression_analysis(
            df
        )
    )

    return {

        "cohort_summary": summary,

        "medication_group_analysis":
            medication_results,

        "regression_analysis":
            regression_results
    }