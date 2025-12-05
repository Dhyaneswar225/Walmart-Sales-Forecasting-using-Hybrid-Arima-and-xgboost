# arima_modeling.py
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- external libs: statsmodels & pmdarima
try:
    import pmdarima as pm
    import statsmodels.api as sm
    import statsmodels.graphics.gofplots as sm_gof
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
except Exception as e:
    raise ImportError("Please install pmdarima and statsmodels (pip install pmdarima statsmodels)") from e

# ---------- User-configurable parameters ----------
MERGED_PATH = "F:\\Advanced Analytics\\Project\\WalmartSalesForecasting\\data\\merged_walmart.csv"
RESULTS_DIR = "F:\\Advanced Analytics\\Project\\WalmartSalesForecasting\\results\\"
STORE_ID = 1
DEPT_ID = 1
SEASONAL_M = 52          # weekly data -> yearly seasonality
HORIZON = 12             # forecast horizon (weeks)
AUTO_MAX_P = 5
AUTO_MAX_Q = 5
AUTO_MAX_P_SEAS = 2
AUTO_MAX_Q_SEAS = 2
AUTO_STEPWISE = True
# -------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)

def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def to_datetime_index_safe(df, date_col="Date"):
    """
    Ensure df has a DatetimeIndex and return df (not a copy).
    If date_col is present as a column it will be set as index.
    """
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        # index might already be datetime-like; try to coerce
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    # make sure it's a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df

def main():
    if not os.path.exists(MERGED_PATH):
        raise FileNotFoundError(f"Merged dataset not found at {MERGED_PATH}. Run Phase 2 and save to this path.")
    df = pd.read_csv(MERGED_PATH, parse_dates=["Date"])
    print("Merged dataset loaded successfully.")
    print(df.head())

    ts = df[(df["Store"] == STORE_ID) & (df["Dept"] == DEPT_ID)].copy()
    ts = ts.sort_values("Date").reset_index(drop=True)
    ts = ts.set_index("Date")
    ts.index = pd.DatetimeIndex(ts.index)
    ts = ts.asfreq("W-FRI")

    # Set Date as datetime index (safe)
    ts = to_datetime_index_safe(ts, date_col="Date")

    # keep only the target series (Weekly_Sales) but preserve index
    if "Weekly_Sales" not in ts.columns:
        raise KeyError("Weekly_Sales column not found in merged dataset.")
    ts = ts[["Weekly_Sales"]].dropna()
    print(f"\nFiltered series (Store={STORE_ID}, Dept={DEPT_ID}) length: {len(ts)}")

    print("\nRunning ADF test...")
    adf_res = adfuller(ts["Weekly_Sales"])
    adf_out = {
        "ADF Statistic": adf_res[0],
        "p-value": adf_res[1],
        "Used Lag": adf_res[2],
        "Num Observations": adf_res[3],
    }
    crit = adf_res[4]
    adf_text = "ADF Test Results:\n"
    for k, v in adf_out.items():
        adf_text += f"{k}: {v}\n"
    adf_text += "Critical Values:\n"
    for k, v in crit.items():
        adf_text += f"  {k}: {v}\n"
    print(adf_text)
    save_text(os.path.join(RESULTS_DIR, "adf_results.txt"), adf_text)

    print("\nRunning auto_arima (this may take a while)...")
    auto = pm.auto_arima(ts["Weekly_Sales"],
                         seasonal=True, m=SEASONAL_M,
                         max_p=AUTO_MAX_P, max_q=AUTO_MAX_Q,
                         max_P=AUTO_MAX_P_SEAS, max_Q=AUTO_MAX_Q_SEAS,
                         stepwise=AUTO_STEPWISE,
                         trace=True,
                         error_action='ignore',
                         suppress_warnings=True,
                         n_jobs=1)
    print("\nauto_arima suggested order:", auto.order, ", seasonal_order:", auto.seasonal_order)
    save_text(os.path.join(RESULTS_DIR, "auto_arima_summary.txt"), str(auto.summary()))

    order = auto.order
    sorder = auto.seasonal_order  # (P, D, Q, m)
    print("\nFitting SARIMAX via statsmodels with order", order, "seasonal_order", sorder)
    model = sm.tsa.statespace.SARIMAX(ts["Weekly_Sales"],
                                      order=order,
                                      seasonal_order=sorder,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
    res = model.fit(disp=False)
    print("\nModel fitted. Summary:\n")
    print(res.summary())
    save_text(os.path.join(RESULTS_DIR, "arima_summary.txt"), res.summary().as_text())

    fitted = res.fittedvalues  # aligned with ts index
    residuals = ts["Weekly_Sales"] - fitted
    insample_df = pd.DataFrame({
        "actual": ts["Weekly_Sales"].values,
        "fitted": fitted.values,
        "residual": residuals.values
    }, index=ts.index)
    insample_df.to_csv(os.path.join(RESULTS_DIR, "arima_insample.csv"))
    plt.figure(figsize=(10,5))
    plt.plot(ts.index, ts["Weekly_Sales"], label="Actual")
    plt.plot(fitted.index, fitted.values, label="Fitted (in-sample)")
    plt.legend()
    plt.title(f"Actual vs Fitted — Store {STORE_ID} Dept {DEPT_ID}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "forecast_insample.png"))
    plt.close()

    # residual time series
    plt.figure(figsize=(10,4))
    plt.plot(residuals.index, residuals.values)
    plt.title("Residuals (in-sample)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "resid_ts.png"))
    plt.close()

    # resid ACF & PACF
    plot_acf(residuals.dropna(), lags=40)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "resid_acf.png"))
    plt.close()

    plot_pacf(residuals.dropna(), lags=40, method="ywm")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "resid_pacf.png"))
    plt.close()

    # QQ plot of residuals
    sm_gof.qqplot(residuals.dropna(), line='s')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "resid_qq.png"))
    plt.close()

    print(f"\nGenerating {HORIZON}-step ahead forecast...")
    forecast_res = res.get_forecast(steps=HORIZON)
    fc_mean = forecast_res.predicted_mean
    fc_ci = forecast_res.conf_int(alpha=0.05)  # 95% CI

    # Build forecast index robustly using DatetimeIndex
    # If ts.index is PeriodIndex, convert last period to timestamp
    last_idx = ts.index[-1]
    try:
        # try to get a Timestamp for last index
        if isinstance(last_idx, pd.Period):
            last_ts = last_idx.to_timestamp(how="end")
        else:
            last_ts = pd.to_datetime(last_idx)
    except Exception:
        last_ts = pd.to_datetime(last_idx)

    fc_index = pd.date_range(start=last_ts + pd.Timedelta(days=7), periods=HORIZON, freq='W')  # weekly freq
    # fallback if lengths mismatch
    if len(fc_index) != HORIZON:
        fc_index = pd.Index([last_ts + pd.Timedelta(days=7*(i+1)) for i in range(HORIZON)])

    fc_df = pd.DataFrame({
        "forecast": fc_mean.values,
        "lower_ci": fc_ci.iloc[:, 0].values,
        "upper_ci": fc_ci.iloc[:, 1].values
    }, index=fc_index)
    fc_df.to_csv(os.path.join(RESULTS_DIR, "arima_forecast.csv"))

    # plot forecast
    plt.figure(figsize=(10,6))
    plt.plot(ts.index, ts["Weekly_Sales"], label="Actual")
    plt.plot(fitted.index, fitted.values, label="Fitted (in-sample)")
    plt.plot(fc_df.index, fc_df["forecast"], label="Forecast", color="tab:orange")
    plt.fill_between(fc_df.index, fc_df["lower_ci"], fc_df["upper_ci"], color="orange", alpha=0.2)
    plt.legend()
    plt.title(f"SARIMAX Forecast — Store {STORE_ID} Dept {DEPT_ID}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "forecast.png"))
    plt.close()
    residuals_df = insample_df[["residual"]].reset_index().rename(columns={"index": "Date"})
    residuals_df.to_csv(os.path.join(RESULTS_DIR, "arima_insample_residuals.csv"), index=False)

    print("\n(ARIMA/SARIMA) complete.")

if __name__ == "__main__":
    main()
