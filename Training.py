import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io
import contextlib

# Context manager to suppress Stan output
class suppress_stdout_stderr(contextlib.ContextDecorator):
    def __enter__(self):
        self.out = io.StringIO()
        self.err = io.StringIO()
        self.stdout_ctx = contextlib.redirect_stdout(self.out)
        self.stderr_ctx = contextlib.redirect_stderr(self.err)
        self.stdout_ctx.__enter__()
        self.stderr_ctx.__enter__()
        return self
    
    def __exit__(self, *exc):
        self.stdout_ctx.__exit__(*exc)
        self.stderr_ctx.__exit__(*exc)
        self.out.close()
        self.err.close()
        return False

# Enhanced forecasting function
def generate_forecast(product_id, data, horizons):
    df_p = data[data['ProductID'] == product_id]
    if len(df_p) < 14:  # Minimum 2 weeks of data
        print(f"Skipping {product_id} - insufficient data ({len(df_p)} records)")
        return None
    
    stock = product_stock.get(product_id, 0)
    
    try:
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            holidays_prior_scale=10,
            seasonality_mode='additive'
        )
        model.add_country_holidays(country_name='MY')
        
        with suppress_stdout_stderr():
            model.fit(df_p[['ds', 'y']])
        
        future = model.make_future_dataframe(periods=max(horizons))
        forecast = model.predict(future)
        
        # Generate predictions for all horizons
        predictions = {
            days: forecast[forecast['ds'] > df_p['ds'].max()].head(days)
            for days in horizons
        }
        
        # Stock analysis and alerts
        for days in horizons:
            forecast_subset = predictions[days]
            daily_forecast = forecast_subset['yhat'].values
            cumulative_stock = stock - np.cumsum(daily_forecast)
            
            stockout_days = np.where(cumulative_stock < 0)[0]
            status = "Restock Needed" if len(stockout_days) > 0 else "OK"
            
            forecast_data.append({
                'ProductID': product_id,
                'HorizonDays': days,
                'TotalForecast': round(sum(daily_forecast)),
                'CurrentStock': stock,
                'Status': status,
                'FirstStockout': stockout_days[0]+1 if len(stockout_days) > 0 else None
            })
        
        # Plotting
        fig = model.plot(forecast)
        plt.title(f"Product {product_id} - Daily Sales Forecast")
        plt.savefig(f'forecast_plots/product_{product_id}.png')
        plt.close()
        
        return predictions
        
    except Exception as e:
        print(f"Error processing {product_id}: {str(e)}")
        return None

# Walk-forward validation function
def walk_forward_validation(df, product_id, horizons, train_window=60, step=7):
    df_p = df[df['ProductID'] == product_id].sort_values('ds')
    if len(df_p) < train_window + max(horizons):
        return pd.DataFrame()
    
    results = []
    try:
        for start in range(0, len(df_p) - train_window - max(horizons) + 1, step):
            train = df_p.iloc[start:start+train_window]
            
            for days in horizons:
                test = df_p.iloc[start+train_window:start+train_window+days]
                if len(test) < days:
                    continue
                
                if test['y'].var() < 1e-6:
                    continue
                
                model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10,
                    holidays_prior_scale=10,
                    seasonality_mode='additive'
                )
                model.add_country_holidays(country_name='MY')
                
                with suppress_stdout_stderr():
                    model.fit(train[['ds', 'y']])
                
                future = model.make_future_dataframe(periods=days)
                forecast = model.predict(future)
                
                y_pred = forecast['yhat'][-days:].values
                y_true = test['y'].values
                
                y_mean = np.mean(y_true)
                ss_tot = np.sum((y_true - y_mean)**2)
                if ss_tot < 1e-6:
                    r2 = float('nan')
                else:
                    r2 = max(r2_score(y_true, y_pred), -10)
                
                results.append({
                    'horizon': days,
                    'mae': mean_absolute_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'r2': r2
                })
                
    except Exception as e:
        print(f"Validation error for {product_id}: {str(e)}")
    
    return pd.DataFrame(results)

# ------------ Main Execution ------------
# Load and process data
order_items = pd.read_csv('data/Order_Items.csv')
products = pd.read_csv('data/Product.csv')
sales = pd.read_csv('data/Sales.csv')
users = pd.read_csv('data/User.csv')
inventory = pd.read_csv('data/Inventory.csv')

df = pd.merge(order_items, sales, on='SalesID')
df = pd.merge(df, users, on='UserID')
df = pd.merge(df, products, on='ProductID', how='left')
df['Order Date'] = pd.to_datetime(df['Order Date'])

daily_sales_per_product = df.groupby(['Order Date', 'ProductID'])['Quantity'].sum().reset_index()
daily_sales_per_product.columns = ['ds', 'ProductID', 'y']

product_stock = {row['productID']: row['CurrentStock'] for _, row in inventory.iterrows()}
product_ids = products['ProductID'].unique()
forecast_days_list = [1, 3, 5, 7, 30]

# Initialize outputs
forecast_data = []
evaluation_results = []
os.makedirs('forecast_plots', exist_ok=True)

# Process each product
for pid in product_ids:
    forecasts = generate_forecast(pid, daily_sales_per_product, forecast_days_list)
    if forecasts is not None:
        val_results = walk_forward_validation(daily_sales_per_product, pid, forecast_days_list)
        if not val_results.empty:
            evaluation_results.append(val_results)

# Save results
forecast_df = pd.DataFrame(forecast_data)
forecast_df.to_csv('product_forecasts.csv', index=False)

if evaluation_results:
    combined_results = pd.concat(evaluation_results)
    combined_results.to_csv('forecast_evaluation.csv', index=False)
    print("\nValidation Metrics:")
    print(combined_results.groupby('horizon').agg({'mae':'mean', 'rmse':'mean', 'r2':'mean'}))

with open('forecast_results.pkl', 'wb') as f:
    pickle.dump({
        'forecasts': forecast_df,
        'evaluation': combined_results if evaluation_results else None
    }, f)

print("\nProcess completed successfully!")
print(f"Forecasts saved to: {os.path.abspath('product_forecasts.csv')}")
print(f"Evaluation saved to: {os.path.abspath('forecast_evaluation.csv')}")