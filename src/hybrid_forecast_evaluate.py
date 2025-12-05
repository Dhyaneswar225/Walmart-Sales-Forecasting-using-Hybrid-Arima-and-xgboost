# hybrid_forecast_evaluate.py
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- external libs
try:
    import pmdarima as pm
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error
except Exception as e:
    raise ImportError("Please install pmdarima, statsmodels, xgboost, scikit-learn") from e

# -------------------------
# USER CONFIG
# -------------------------
MERGED_PATH = "F:\\Advanced Analytics\\Project\\WalmartSalesForecasting\\data\\merged_walmart.csv"
RESULTS_DIR = "F:\\Advanced Analytics\\Project\\WalmartSalesForecasting\\results\\"
STORE_ID = 1
DEPT_ID = 1
SEASONAL_M = 52        # weekly -> yearly seasonality
HORIZON = 12           # holdout length (weeks)
AUTO_MAX_P = 5
AUTO_MAX_Q = 5
AUTO_MAX_P_SEAS = 2
AUTO_MAX_Q_SEAS = 2
RANDOM_STATE = 42
# -------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------
# Utility / metrics
# -------------------------
def rmse(y, yhat): return np.sqrt(mean_squared_error(y, yhat))
def mae(y, yhat): return mean_absolute_error(y, yhat)
def mape(y, yhat): return np.mean(np.abs((y - yhat) / np.where(y==0, 1e-8, y))) * 100
def smape(y, yhat): return 100/len(y) * np.sum(2 * np.abs(yhat - y) / (np.abs(y) + np.abs(yhat) + 1e-8))
def mase(y, yhat, y_train, seasonality=1):
    denom = np.mean(np.abs(np.diff(y_train, seasonality)))
    return np.mean(np.abs(y - yhat)) / (denom + 1e-8)
def wmape(y, yhat): return np.sum(np.abs(y - yhat)) / np.sum(np.abs(y)) * 100

# -------------------------
# Feature engineering (safe, non-leaky)
# -------------------------
def create_lag_roll_features(df, target_col="Weekly_Sales", lags=[1,2,52], rolls=[4,8,52]):
    # df must be sorted by Date ascending and contain the target column
    X = df.copy()
    for lag in lags:
        X[f"lag_{lag}"] = X[target_col].shift(lag)
    for r in rolls:
        X[f"rmean_{r}"] = X[target_col].rolling(window=r).mean().shift(1)
    # time features
    X["week"] = X["Date"].dt.isocalendar().week.astype(int)
    X["month"] = X["Date"].dt.month
    X["year"] = X["Date"].dt.year
    # holiday numeric
    X["IsHoliday"] = X["IsHoliday"].astype(int)
    # keep external features if present (Temperature, Fuel_Price, CPI, Unemployment, MarkDown*)
    ext_cols = ["Temperature","Fuel_Price","CPI","Unemployment","MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]
    for c in ext_cols:
        if c not in X.columns:
            continue
    # drop rows with NA created by shifting
    return X

# -------------------------
# Main
# -------------------------
def main():
    if not os.path.exists(MERGED_PATH):
        raise FileNotFoundError(f"Merged file not found at {MERGED_PATH}. Run Phase 2 and save merged_walmart.csv there.")
    df = pd.read_csv(MERGED_PATH, parse_dates=["Date"])
    print("Merged dataset loaded successfully.")
    # filter
    series = df[(df["Store"]==STORE_ID) & (df["Dept"]==DEPT_ID)].sort_values("Date").reset_index(drop=True)
    if len(series) < (HORIZON + 20):
        raise ValueError("Not enough data for the chosen horizon. Reduce HORIZON or pick a longer series.")

    # split train/test (last HORIZON weeks as test)
    train = series.iloc[:-HORIZON].copy().reset_index(drop=True)
    test = series.iloc[-HORIZON:].copy().reset_index(drop=True)

    # --- ARIMA: fit on train, forecast H steps
    print("Fitting auto_arima on train...")
    auto = pm.auto_arima(train["Weekly_Sales"],
                         seasonal=True, m=SEASONAL_M,
                         max_p=AUTO_MAX_P, max_q=AUTO_MAX_Q,
                         max_P=AUTO_MAX_P_SEAS, max_Q=AUTO_MAX_Q_SEAS,
                         stepwise=True, trace=False,
                         error_action='ignore', suppress_warnings=True)
    order = auto.order
    sorder = auto.seasonal_order
    print("auto_arima selected:", order, sorder)
    # statsmodels fit for diagnostics & forecasting
    model = sm.tsa.statespace.SARIMAX(train["Weekly_Sales"],
                                      order=order, seasonal_order=sorder,
                                      enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fitted_in = res.fittedvalues
    # out-of-sample forecast
    fc = res.get_forecast(steps=HORIZON)
    arima_forecast = fc.predicted_mean.values
    arima_ci = fc.conf_int()
    # index align
    test_dates = test["Date"].values
    arima_df = pd.DataFrame({"Date": test["Date"].values, "arima_forecast": arima_forecast})
    arima_df.to_csv(os.path.join(RESULTS_DIR, "phase6_arima_forecast.csv"), index=False)
    print("ARIMA forecast saved.")

    # --- Prepare features for ML
    # concat train+test to compute lag features easily, but careful to avoid leaking target values.
    # We'll compute lag/rolling from the actual series (so test features use true history up to that test date).
    combined = pd.concat([train, test], axis=0).reset_index(drop=True)
    combined = create_lag_roll_features(combined, target_col="Weekly_Sales",
                                        lags=[1,2,52], rolls=[4,8,52])
    # Extract ML rows: for training ML on residuals we use the train portion only (rows that still have lag features)
    combined = combined.reset_index(drop=True)
    # create a boolean mask for rows belonging to train / test
    combined["is_train"] = combined["Date"].isin(train["Date"])
    combined["is_test"] = combined["Date"].isin(test["Date"])

    # drop rows with NA in lag features (they occur at start)
    ml_ready = combined.dropna().reset_index(drop=True)

    # split back
    ml_train = ml_ready[ml_ready["is_train"]].copy().reset_index(drop=True)
    ml_test = ml_ready[ml_ready["is_test"]].copy().reset_index(drop=True)

    # --- ARIMA residuals on train (in-sample residuals)
    # Need ARIMA fitted values aligned: res.fittedvalues has index aligned to train index positions
    # create a train fitted series aligned to ml_train by Date
    fitted_series = pd.Series(fitted_in, index=train.index)  # index positions
    # Build a df to merge by Date
    train_fit_df = pd.DataFrame({
        "Date": train["Date"].values,
        "arima_fitted": fitted_series.values
    })
    # Merge into ml_train
    ml_train = ml_train.merge(train_fit_df, on="Date", how="left")
    # Compute residual = actual - arima_fitted
    ml_train["residual"] = ml_train["Weekly_Sales"] - ml_train["arima_fitted"]

    # For ml_test we need an "arima_fitted" for corresponding dates: use arima forecast as 'arima_fitted' for test
    arima_test_df = arima_df.rename(columns={"arima_forecast":"arima_fitted"})
    ml_test = ml_test.merge(arima_test_df[["Date","arima_fitted"]], on="Date", how="left")

    # Feature columns for ML
    feature_cols = [c for c in ml_train.columns if c.startswith("lag_") or c.startswith("rmean_")]
    # also add external features if present
    extra = ["IsHoliday","Temperature","Fuel_Price","CPI","Unemployment","MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]
    for c in extra:
        if c in ml_train.columns:
            feature_cols.append(c)
    feature_cols = list(dict.fromkeys(feature_cols))  # unique preserve order

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found for ML. Check merged data and feature generation.")

    X_train = ml_train[feature_cols]
    y_train_res = ml_train["residual"]

    X_test = ml_test[feature_cols]
    # Note: ml_test rows correspond to the test horizon; some features use true test sales (lag_1 etc.) — see caveat above.

    # --- Train XGBoost on residuals
    print("Training XGBoost on ARIMA residuals...")
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=4, random_state=RANDOM_STATE)
    # small randomized search for good params (speed conscious)
    param_dist = {
        "n_estimators":[100,200,400],
        "learning_rate":[0.01,0.05,0.1],
        "max_depth":[3,5,7],
        "subsample":[0.6,0.8,1.0]
    }
    tscv = TimeSeriesSplit(n_splits=3)
    rs = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=8, cv=tscv,
                            scoring='neg_root_mean_squared_error', n_jobs=4, random_state=RANDOM_STATE, verbose=0)
    rs.fit(X_train.fillna(0), y_train_res.fillna(0))
    best_xgb = rs.best_estimator_
    # save model
    import joblib
    joblib.dump(best_xgb, os.path.join(RESULTS_DIR, "xgb_residual_model.pkl"))
    print("XGBoost residual model trained & saved.")

    # Predict residuals for test
    pred_res_test = best_xgb.predict(X_test.fillna(0))

    # Hybrid forecast
    hybrid_forecast = ml_test["arima_fitted"].values + pred_res_test

    # --- ML-only baseline: train XGBoost directly to predict Weekly_Sales using same features
    print("Training ML-only XGBoost baseline (predict Weekly_Sales directly)...")
    X_train_direct = ml_train[feature_cols].fillna(0)
    y_train_direct = ml_train["Weekly_Sales"].fillna(0)
    xgb_direct = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=4, random_state=RANDOM_STATE,
                                  n_estimators=200, learning_rate=0.05, max_depth=5)
    xgb_direct.fit(X_train_direct, y_train_direct)
    joblib.dump(xgb_direct, os.path.join(RESULTS_DIR, "xgb_direct_model.pkl"))
    direct_pred_test = xgb_direct.predict(X_test.fillna(0))

    # --- Evaluation: compare ARIMA (arima_df), ML-only (direct_pred_test), Hybrid
    y_true = ml_test["Weekly_Sales"].values  # true sales for test rows
    arima_pred = ml_test["arima_fitted"].values
    ml_only_pred = direct_pred_test
    hybrid_pred = hybrid_forecast

    metrics = {}
    for name, yhat in [("ARIMA", arima_pred), ("ML-only", ml_only_pred), ("Hybrid", hybrid_pred)]:
        metrics[name] = {
            "RMSE": rmse(y_true, yhat),
            "MAE": mae(y_true, yhat),
            "MAPE": mape(y_true, yhat),
            "sMAPE": smape(y_true, yhat),
            "MASE": mase(y_true, yhat, train["Weekly_Sales"].values, seasonality=SEASONAL_M),
            "wMAPE": wmape(y_true, yhat)
        }

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "phase6_metrics.csv"))
    print(metrics_df)

    # save forecasts
    out_df = pd.DataFrame({
        "Date": ml_test["Date"].values,
        "actual": y_true,
        "arima": arima_pred,
        "ml_only": ml_only_pred,
        "hybrid": hybrid_pred
    })
    out_df.to_csv(os.path.join(RESULTS_DIR, "phase6_forecasts_compare.csv"), index=False)

    # feature importance plot from residual model
    try:
        fi = best_xgb.feature_importances_
        plt.figure(figsize=(8,5))
        inds = np.argsort(fi)[::-1]
        names = np.array(feature_cols)[inds]
        vals = fi[inds]
        plt.barh(range(len(vals)), vals[::-1])
        plt.yticks(range(len(vals)), names[::-1])
        plt.title("XGBoost (residual model) feature importances")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "phase6_resid_feature_importance.png"))
        plt.close()
    except Exception:
        pass

    print("\nHybrid forecast + evaluation complete.")
if __name__ == "__main__":
    main()
