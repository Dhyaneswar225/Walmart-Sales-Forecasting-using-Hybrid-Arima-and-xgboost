import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

MERGED_PATH = "F:\\Advanced Analytics\\Project\\WalmartSalesForecasting\\data\\merged_walmart.csv"
ARIMA_RESIDUALS_PATH = "F:\\Advanced Analytics\\Project\\WalmartSalesForecasting\\results\\arima_insample_residuals.csv"
RESULTS_DIR = "F:\\Advanced Analytics\\Project\\WalmartSalesForecasting\\results\\"

STORE_ID = 1
DEPT_ID = 1
LAGS = [1, 2, 52]
ROLLING_WINDOWS = [4, 8, 52]

os.makedirs(RESULTS_DIR, exist_ok=True)

def create_features(df):
    for lag in LAGS:
        df[f"lag_{lag}"] = df["Weekly_Sales"].shift(lag)

    for win in ROLLING_WINDOWS:
        df[f"roll_mean_{win}"] = df["Weekly_Sales"].rolling(win).mean().shift(1)

    return df


def main():
    merged = pd.read_csv(MERGED_PATH, parse_dates=["Date"])
    arima_res = pd.read_csv(ARIMA_RESIDUALS_PATH, parse_dates=["Date"])

    df = merged[(merged["Store"] == STORE_ID) & (merged["Dept"] == DEPT_ID)].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df = df.merge(arima_res, on="Date", how="left")

    print("\nMerged with residuals:")
    print(df.head())

    df = create_features(df)
    feature_cols = [
        "lag_1","lag_2","lag_52",
        "roll_mean_4","roll_mean_8","roll_mean_52",
        "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "IsHoliday", "Size"
    ]

    df = df.dropna().reset_index(drop=True)

    X = df[feature_cols]
    y = df["residual"]

    test_size = 12  # match ARIMA horizon
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    print("\nTraining XGBoost model...")
    model.fit(X_train, y_train)

    y_pred_residuals = model.predict(X_test)

    # Save predictions
    out_df = pd.DataFrame({
        "Date": df["Date"].iloc[-test_size:],
        "residual_pred": y_pred_residuals,
        "actual_residual": y_test.values
    })

    out_df.to_csv(os.path.join(RESULTS_DIR, "ml_residual_predictions.csv"), index=False)

    importance = model.feature_importances_
    imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importance})
    imp_df = imp_df.sort_values("Importance", ascending=False)

    plt.figure(figsize=(10,6))
    plt.barh(imp_df["Feature"], imp_df["Importance"])
    plt.title("XGBoost Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ml_feature_importance.png"))
    plt.close()

    print("\nML Residual Model Ready for Hybrid Step.")

if __name__ == "__main__":
    main()