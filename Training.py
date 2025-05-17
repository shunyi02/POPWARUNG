import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import os
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

# Create a dictionary of current stock levels from the inventory CSV
product_stock = {}
for _, row in inventory.iterrows():
    product_stock[row['productID']] = row['CurrentStock']

# Get list of all product IDs from Product.csv (not just those with sales)
product_ids = products['ProductID'].unique()

# Forecast horizons
forecast_days_list = [1, 3, 5, 7, 30]

# Dictionary to store forecasts
forecast_data = []

# Function to forecast and plot one product with separate subplots
def plot_forecast_subplots(product_id):
    df_p = daily_sales_per_product[daily_sales_per_product['ProductID'] == product_id]
    
    if len(df_p) < 2:
        print(f"Not enough data for {product_id}")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows x 3 cols
    axes = axes.flatten()

    alerts = []

    for idx, days in enumerate(forecast_days_list):
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.add_country_holidays(country_name='MY')  # Malaysia holidays
        model.fit(df_p[['ds', 'y']])
        
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)

        forecast_subset = forecast[forecast['ds'] > df_p['ds'].max()].head(days)
        total_forecast = round(forecast_subset['yhat'].sum())
        stock = product_stock.get(product_id, 0)

        # Alert message
        status = "Restock Needed" if stock < total_forecast else " OK"
        alert_msg = f"{days} day(s): {total_forecast} units ({status})"
        alerts.append(alert_msg)

        # Store forecast data
        forecast_data.append({
            'ProductID': product_id,
            'Days': days,
            'Forecasted': total_forecast,
            'Current Stock': stock,
            'Stock Status': 'Restock Needed' if stock < total_forecast else 'OK'
        })

        # Plot historical + forecast
        ax = axes[idx]
        ax.plot(df_p['ds'], df_p['y'], 'o-', label='Historical Sales')
        ax.plot(forecast['ds'], forecast['yhat'], 'r--', label='Forecast')
        ax.axvline(x=df_p['ds'].max(), color='gray', linestyle=':', label='Forecast Start')
        ax.set_title(f'{days}-Day Forecast')
        ax.grid(True)
        ax.legend()

    # Hide extra subplot if any
    for idx in range(len(forecast_days_list), len(axes)):
        fig.delaxes(axes[idx])

    # Add summary title
    full_alert = "\n".join(alerts)
    fig.suptitle(f"Sales Forecast for {product_id}\n\n{full_alert}", fontsize=14, y=0.95)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.show()



# Plot forecast for each product in a single window with subplots
for pid in product_ids:
    plot_forecast_subplots(pid)

# Convert forecast data to DataFrame
forecast_df = pd.DataFrame(forecast_data)

# Save to CSV
output_file = 'product_forecasts.csv'
forecast_df.to_csv(output_file, index=False)

print(f"\n Forecast saved to '{os.path.abspath(output_file)}'")


# --- Place all function definitions here ---
def evaluate_forecast(y_true, y_pred, stock_level):
    # Regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    # Classification: 1 if forecast > stock, else 0
    y_true_class = (y_true > stock_level).astype(int)
    y_pred_class = (y_pred > stock_level).astype(int)
    acc = accuracy_score(y_true_class, y_pred_class)
    f1 = f1_score(y_true_class, y_pred_class)
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'accuracy': acc, 'f1': f1}

def walk_forward_validation(df, product_id, forecast_days_list, train_window=30, step=1):
    df_p = df[df['ProductID'] == product_id].sort_values('ds')
    results = []
    for start in range(0, len(df_p) - train_window - max(forecast_days_list) + 1, step):
        train = df_p.iloc[start:start+train_window]
        test_start = start + train_window
        for horizon in forecast_days_list:
            test = df_p.iloc[test_start:test_start+horizon]
            if len(test) < 1:
                continue
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(train[['ds', 'y']])
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)
            y_pred = forecast['yhat'][-horizon:].values
            y_true = test['y'].values
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            if len(test) > 1:
                r2 = r2_score(y_true, y_pred)
            else:
                r2 = float('nan')  # or None
            results.append({'horizon': horizon, 'mae': mae, 'rmse': rmse, 'r2': r2})
    return pd.DataFrame(results)
# After the main forecasting loop, add:
evaluation_results = []
for pid in product_ids:
    if pid in daily_sales_per_product['ProductID'].unique():
        wfv_results = walk_forward_validation(daily_sales_per_product, pid, forecast_days_list)
        evaluation_results.append(wfv_results)

if evaluation_results:
    combined_results = pd.concat(evaluation_results)
    regression_report(combined_results)
    combined_results.to_csv('forecast_evaluation.csv', index=False)

def regression_report(results_df):
    """
    Print a regression report from a DataFrame of results.
    Expects columns: 'horizon', 'mae', 'rmse', 'r2'
    """
    print("Regression Performance Report")
    print("="*30)
    for horizon in sorted(results_df['horizon'].unique()):
        subset = results_df[results_df['horizon'] == horizon]
        print(f"\nHorizon: {horizon} day(s)")
        print(f"  MAE : {subset['mae'].mean():.4f}")
        print(f"  RMSE: {subset['rmse'].mean():.4f}")
        print(f"  R²  : {subset['r2'].mean():.4f}")
    print("\nOverall:")
    print(f"  MAE : {results_df['mae'].mean():.4f}")
    print(f"  RMSE: {results_df['rmse'].mean():.4f}")
    print(f"  R²  : {results_df['r2'].mean():.4f}")