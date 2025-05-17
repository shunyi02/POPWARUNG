import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import os

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
        status = "⚠ Restock" if stock < total_forecast else "✅ OK"
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


# Get list of unique product IDs
product_ids = daily_sales_per_product['ProductID'].unique()

# Plot forecast for each product in a single window with subplots
for pid in product_ids:
    plot_forecast_subplots(pid)

# Convert forecast data to DataFrame
forecast_df = pd.DataFrame(forecast_data)

# Save to CSV
output_file = 'product_forecasts.csv'
forecast_df.to_csv(output_file, index=False)

print(f"\n✅ Forecast saved to '{os.path.abspath(output_file)}'")