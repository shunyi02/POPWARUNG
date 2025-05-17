import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score

# Load data
order_items = pd.read_csv('data/Order_Items.csv')
products = pd.read_csv('data/Product.csv')
sales = pd.read_csv('data/Sales.csv')
users = pd.read_csv('data/User.csv')
inventory = pd.read_csv('data/Inventory.csv')

# Merge datasets
df = pd.merge(order_items, sales, on='SalesID')
df = pd.merge(df, users, on='UserID')
df = pd.merge(df, products, on='ProductID', how='left')

# Convert date column
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Aggregate daily sales per product
daily_sales_per_product = df.groupby(['Order Date', 'ProductID'])['Quantity'].sum().reset_index()
daily_sales_per_product.columns = ['ds', 'ProductID', 'y']

# Aggregate weekly sales per product
weekly_sales_per_product = df.groupby([
    pd.Grouper(key='Order Date', freq='W-MON'), 'ProductID'
])['Quantity'].sum().reset_index()
weekly_sales_per_product.columns = ['ds', 'ProductID', 'y']

# Create a dictionary of current stock levels from the inventory CSV
product_stock = {row['productID']: row['CurrentStock'] for _, row in inventory.iterrows()}

# Get list of all product IDs from Product.csv
product_ids = products['ProductID'].unique()

# Forecast horizons
forecast_days_list = [1, 3, 5, 7, 30]
forecast_weeks_list = [1, 2, 3, 4, 12]

# Dictionary to store forecasts
forecast_data = []
evaluation_results = []

# Function to forecast and plot (handles both daily and weekly)
def plot_forecast_subplots(product_id, data, horizons, freq='D', label='Day'):
    df_p = data[data['ProductID'] == product_id]
    if len(df_p) < 2:
        print(f"Not enough data for {product_id} ({freq})")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    alerts = []

    for idx, horizon in enumerate(horizons):
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=(freq == 'D'),
            daily_seasonality=False,
            changepoint_prior_scale=0.2,
            seasonality_prior_scale=15
        )
        model.add_seasonality(name='monthly', period=30.5 if freq == 'D' else 4, fourier_order=5 if freq == 'D' else 3)
        model.add_country_holidays(country_name='MY')
        model.fit(df_p[['ds', 'y']])
        
        future = model.make_future_dataframe(periods=horizon, freq=freq)
        forecast = model.predict(future)

        forecast_subset = forecast[forecast['ds'] > df_p['ds'].max()].head(horizon)
        total_forecast = round(forecast_subset['yhat'].sum())
        stock = product_stock.get(product_id, 0)

        status = "Restock Needed" if stock < total_forecast else "OK"
        alert_msg = f"{horizon} {label}(s): {total_forecast} units ({status})"
        alerts.append(alert_msg)

        forecast_data.append({
            'ProductID': product_id,
            f'{label}s': horizon,
            'Forecasted': total_forecast,
            'Current Stock': stock,
            'Stock Status': status
        })

        ax = axes[idx]
        ax.plot(df_p['ds'], df_p['y'], 'o-', label='Historical Sales')
        ax.plot(forecast['ds'], forecast['yhat'], 'r--', label='Forecast')
        ax.axvline(x=df_p['ds'].max(), color='gray', linestyle=':', label='Forecast Start')
        ax.set_title(f'{horizon}-{label} Forecast')
        ax.grid(True)
        ax.legend()

    for idx in range(len(horizons), len(axes)):
        fig.delaxes(axes[idx])

    full_alert = "\n".join(alerts)
    fig.suptitle(f"{label}ly Sales Forecast for {product_id}\n\n{full_alert}", fontsize=14, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.show()

# Walk-forward validation
def walk_forward_validation(df, product_id, forecast_horizons, freq='W-MON', train_window=20, step=1):
    df_p = df[df['ProductID'] == product_id].sort_values('ds')
    results = []
    for start in range(0, len(df_p) - train_window - max(forecast_horizons) + 1, step):
        train = df_p.iloc[start:start+train_window]
        test_start = start + train_window
        for horizon in forecast_horizons:
            test = df_p.iloc[test_start:test_start+horizon]
            if len(test) < 1:
                continue
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=(freq == 'D'),
                daily_seasonality=False,
                changepoint_prior_scale=0.2,
                seasonality_prior_scale=15
            )
            model.add_seasonality(name='monthly', period=30.5 if freq == 'D' else 4, fourier_order=5 if freq == 'D' else 3)
            model.fit(train[['ds', 'y']])
            future = model.make_future_dataframe(periods=horizon, freq=freq)
            forecast = model.predict(future)
            y_pred = forecast['yhat'][-horizon:].values
            y_true = test['y'].values
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred) if len(test) > 1 else float('nan')
            results.append({'horizon': horizon, 'mae': mae, 'rmse': rmse, 'r2': r2})
    return pd.DataFrame(results)

# Regression report
def regression_report(results_df):
    print("Regression Performance Report")
    print("="*30)
    for horizon in sorted(results_df['horizon'].unique()):
        subset = results_df[results_df['horizon'] == horizon]
        print(f"\nHorizon: {horizon} period(s)")
        print(f"  MAE : {subset['mae'].mean():.4f}")
        print(f"  RMSE: {subset['rmse'].mean():.4f}")
        print(f"  R²  : {subset['r2'].mean():.4f}")
    print("\nOverall:")
    print(f"  MAE : {results_df['mae'].mean():.4f}")
    print(f"  RMSE: {results_df['rmse'].mean():.4f}")
    print(f"  R²  : {results_df['r2'].mean():.4f}")

# Run forecasts and evaluations
for pid in product_ids:
    # Daily forecast
    plot_forecast_subplots(pid, daily_sales_per_product, forecast_days_list, freq='D', label='Day')
    # Weekly forecast
    plot_forecast_subplots(pid, weekly_sales_per_product, forecast_weeks_list, freq='W-MON', label='Week')
    # Walk-forward validation (using weekly data for stability)
    if pid in weekly_sales_per_product['ProductID'].unique():
        wfv_results = walk_forward_validation(weekly_sales_per_product, pid, forecast_weeks_list, freq='W-MON')
        if not wfv_results.empty:
            evaluation_results.append(wfv_results)

# Convert forecast data to DataFrame
forecast_df = pd.DataFrame(forecast_data)

# Save forecast to CSV
output_file = 'product_forecasts.csv'
forecast_df.to_csv(output_file, index=False)
print(f"\nForecast saved to '{os.path.abspath(output_file)}'")

# Combine evaluation results
combined_results = pd.concat(evaluation_results) if evaluation_results else pd.DataFrame()

# Save evaluation results to CSV
if not combined_results.empty:
    combined_results.to_csv('forecast_evaluation.csv', index=False)
    regression_report(combined_results)

# Save forecast_df and combined_results to .pkl file
results_dict = {
    'forecast_df': forecast_df,
    'evaluation_results': combined_results if not combined_results.empty else None
}
pkl_output_file = 'forecast_results.pkl'
with open(pkl_output_file, 'wb') as f:
    pickle.dump(results_dict, f)
print(f"\nForecast and evaluation results saved to '{os.path.abspath(pkl_output_file)}'")