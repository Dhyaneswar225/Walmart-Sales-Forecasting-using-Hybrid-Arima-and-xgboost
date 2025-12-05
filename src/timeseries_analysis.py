import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import os

RESULTS_DIR = "F:\\Advanced Analytics\\Project\\WalmartSalesForecasting\\results\\"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Saving all plots to:", RESULTS_DIR)

df = pd.read_csv("F:\\Advanced Analytics\\Project\\WalmartSalesForecasting\\data\\merged_walmart.csv", parse_dates=["Date"])
print("Merged dataset loaded successfully.")
print(df.head())

store_id = 1
dept_id = 1

ts = df[(df["Store"] == store_id) & (df["Dept"] == dept_id)].copy()
ts = ts.sort_values("Date").reset_index(drop=True)

print("\nFiltered Time Series Preview:")
print(ts.head())
print("Total rows:", len(ts))

plt.figure(figsize=(12,5))
plt.plot(ts["Date"], ts["Weekly_Sales"], label="Weekly Sales")
plt.title("Weekly Sales Over Time (Store 1, Dept 1)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR + "01_weekly_sales_timeseries.png")
plt.close()

ts["rolling_mean_12"] = ts["Weekly_Sales"].rolling(window=12).mean()

plt.figure(figsize=(12,5))
plt.plot(ts["Date"], ts["Weekly_Sales"], alpha=0.4, label="Original")
plt.plot(ts["Date"], ts["rolling_mean_12"], color="red", label="12-Week Rolling Mean")
plt.title("Trend Analysis (Store 1 Dept 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR + "02_rolling_mean.png")
plt.close()

ts["Year"] = ts["Date"].dt.year
ts["Week"] = ts["Date"].dt.isocalendar().week.astype(int)

plt.figure(figsize=(12,6))
sns.lineplot(data=ts, x="Week", y="Weekly_Sales", hue="Year", marker="o")
plt.title("Seasonality Comparison Across Years")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR + "03_seasonality_yearly_comparison.png")
plt.close()

fig_acf = plt.figure(figsize=(12,4))
plot_acf(ts["Weekly_Sales"], lags=40)
plt.tight_layout()
fig_acf.savefig(RESULTS_DIR + "04_acf_plot.png")
plt.close()

fig_pacf = plt.figure(figsize=(12,4))
plot_pacf(ts["Weekly_Sales"], lags=40, method="ywm")
plt.tight_layout()
fig_pacf.savefig(RESULTS_DIR + "05_pacf_plot.png")
plt.close()

print("\nADF Test Results:")

adf_result = adfuller(ts["Weekly_Sales"])

print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")

crit_vals = adf_result[4]

print("Critical Values:")
for key, val in crit_vals.items():
    print(f"   {key}: {val}")

# Save ADF test results to text file
with open(RESULTS_DIR + "06_adf_test_results.txt", "w") as f:
    f.write(f"ADF Statistic: {adf_result[0]}\n")
    f.write(f"p-value: {adf_result[1]}\n")
    f.write("Critical Values:\n")
    for key, val in crit_vals.items():
        f.write(f"   {key}: {val}\n")

print("\nTime Series Analysis completed successfully!")
