# visualization.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import joblib
import numpy as np

RESULTS_DIR = "F:\\Advanced Analytics\\Project\\WalmartSalesForecasting\\results\\"
OUT_DIR = RESULTS_DIR
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading required files...")

compare_path = os.path.join(RESULTS_DIR, "phase6_forecasts_compare.csv")
forecast_compare = pd.read_csv(compare_path, parse_dates=["Date"])

print("Forecast comparison loaded:")
print(forecast_compare.head())

resid_path = os.path.join(RESULTS_DIR, "arima_insample_residuals.csv")
arima_resid = pd.read_csv(resid_path, parse_dates=["Date"])

xgb_model_path = os.path.join(RESULTS_DIR, "xgb_residual_model.pkl")
xgb_model = joblib.load(xgb_model_path)

plt.figure(figsize=(12,6))
plt.plot(forecast_compare["Date"], forecast_compare["actual"], label="Actual", linewidth=2)
plt.plot(forecast_compare["Date"], forecast_compare["arima"], label="ARIMA", linestyle="--")
plt.plot(forecast_compare["Date"], forecast_compare["hybrid"], label="Hybrid", linestyle="--")

plt.title("Actual vs ARIMA vs Hybrid Forecast")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "actual_arima_hybrid.png"))
plt.close()

plt.figure(figsize=(10,5))
plot_acf(arima_resid["residual"], lags=40)
plt.title("ACF of ARIMA Residuals")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "acf_residuals.png"))
plt.close()

fi = xgb_model.feature_importances_
features = xgb_model.get_booster().feature_names

inds = np.argsort(fi)[::-1]
plt.figure(figsize=(10,6))
plt.barh(np.array(features)[inds], fi[inds])
plt.title("XGBoost Residual Model — Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "xgb_feature_importance.png"))
plt.close()

arima_fc_path = os.path.join(RESULTS_DIR, "arima_forecast.csv")
arima_fc = pd.read_csv(arima_fc_path)

# If missing date → rebuild date index
if "Date" not in arima_fc.columns:
    print("ARIMA forecast has no Date. Reconstructing timeline...")
    last_date = forecast_compare["Date"].iloc[-1]
    new_dates = pd.date_range(start=last_date, periods=len(arima_fc)+1, freq="W")[1:]
    arima_fc["Date"] = new_dates

arima_fc["Date"] = pd.to_datetime(arima_fc["Date"])

plt.figure(figsize=(12,6))
plt.plot(arima_fc["Date"], arima_fc["forecast"], label="ARIMA Forecast")
plt.fill_between(arima_fc["Date"], arima_fc["lower_ci"], arima_fc["upper_ci"], alpha=0.3)
plt.title("ARIMA Forecast with Confidence Interval")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "arima_confidence_band.png"))
plt.close()

print("\nVisualizations COMPLETE.")